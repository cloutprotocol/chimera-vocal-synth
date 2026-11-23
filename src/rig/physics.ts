/**
 * Soft-tissue secondary motion (fat & jiggle physics) for the body rig: 3-DOF damped spring-mass
 * oscillators on the breasts (`breastL`, `breastR`), buttocks (`buttL`, `buttR`) and abdominal
 * adipose (`belly`).
 *
 * Each region tracks its parent bone's motion in world space (capturing vertical gait bounce,
 * lateral hip sway, pelvic list, thoracic counter-twist and abrupt stops), with static gravity
 * sag that loads and unloads dynamically on heel-strike and push-off. Displacements are constrained
 * in the parent bone's frame so tissue cannot sink into the ribcage or pelvis, and applied to the
 * skinned mesh with smooth falloff and subtle volume-preserving squash-and-stretch.
 *
 * Units: body cm, seconds (gravity 981 cm/s²). Stable under variable frame rates via 4 fixed
 * symplectic substeps per frame.
 */
import * as THREE from "three";
import type { Rig, TissueDef } from "./rig.ts";

export interface SoftTissueSettings {
  /** turn secondary motion on or off */
  enabled: boolean;
  /** overall jiggle amplitude / responsiveness (1: natural; 0: rigid; up to 2: lively) */
  jiggle: number;
  /** downward acceleration in body cm/s² (981: Earth gravity) */
  gravity: number;
}
export const SOFT_TISSUE_SETTINGS: Readonly<SoftTissueSettings> = { enabled: true, jiggle: 1.0, gravity: 981 };

export interface TissueRegionState {
  readonly def: TissueDef;
  readonly boneIndex: number;
  /** sparse influenced body vertices: [vertexIndex, weight, smoothWeight, relX, relY, relZ] */
  readonly verts: { idx: number; w: number; sw: number }[];
  /** current oscillator centre in world cm */
  readonly pos: THREE.Vector3;
  /** rigid parent-bone target centre in world cm */
  readonly target: THREE.Vector3;
  /** current velocity in world cm/s */
  readonly vel: THREE.Vector3;
  /** clamped displacement (world cm) applied to vertices this frame */
  readonly dispWorld: THREE.Vector3;
  /** clamped displacement in parent-bone local frame (cm) */
  readonly dispLocal: THREE.Vector3;
  /** static gravity sag at rest (cm, downward) */
  sag: number;
  /** current displacement magnitude (cm) */
  magnitude: number;
  /** effective tissue softness/compliance factor (0..1+) */
  compliance: number;
  /** previous thigh flexion (for gluteal heel-strike / femur push coupling) */
  prevThighX: number;
}

const SUBSTEPS = 4;
const MAX_DT = 0.05;
const TELEPORT_CM = 35;

/** smooth C1 saturation: linear near 0, asymptotically approaches `limit` */
const softClamp = (x: number, limit: number) => (limit > 1e-6 ? limit * Math.tanh(x / limit) : 0);

export class SoftTissue {
  readonly settings: SoftTissueSettings = { ...SOFT_TISSUE_SETTINGS };
  readonly regions: TissueRegionState[] = [];
  private readonly byName = new Map<string, TissueRegionState>();
  private seeded = false;

  // reusable scratch vectors/quaternions (no per-frame allocations)
  private readonly vAnchor = new THREE.Vector3();
  private readonly vDelta = new THREE.Vector3();
  private readonly vLocal = new THREE.Vector3();
  private readonly vAcc = new THREE.Vector3();
  private readonly vRel = new THREE.Vector3();
  private readonly vOut = new THREE.Vector3();
  private readonly qInv = new THREE.Quaternion();

  constructor(readonly rig: Rig) {
    for (const def of rig.tissueDefs) {
      const weights = rig.tissues[def.name];
      if (!weights) continue;
      const verts: { idx: number; w: number; sw: number }[] = [];
      for (let i = 0; i < rig.bodyVertices; i++) {
        const w = weights[i];
        if (w > 0.008) {
          const sw = w * w * (3 - 2 * w);
          verts.push({ idx: i, w, sw });
        }
      }
      const state: TissueRegionState = {
        def,
        boneIndex: rig.bone(def.bone),
        verts,
        pos: new THREE.Vector3(),
        target: new THREE.Vector3(),
        vel: new THREE.Vector3(),
        dispWorld: new THREE.Vector3(),
        dispLocal: new THREE.Vector3(),
        sag: 0,
        magnitude: 0,
        compliance: 0,
        prevThighX: 0,
      };
      this.regions.push(state);
      this.byName.set(def.name, state);
    }
  }

  region(name: string): TissueRegionState | undefined {
    return this.byName.get(name);
  }

  /** snap every region to its current rigid pose equilibrium (call after sex/preset jumps) */
  reset(): void {
    this.seeded = false;
    for (const r of this.regions) {
      r.vel.set(0, 0, 0);
      r.dispWorld.set(0, 0, 0);
      r.dispLocal.set(0, 0, 0);
      r.magnitude = 0;
    }
  }

  /** inject a world-space velocity impulse (cm/s) into one region (for tests or interactive poke) */
  impulse(name: string, dv: THREE.Vector3): void {
    const r = this.byName.get(name);
    if (r) r.vel.add(dv);
  }

  /** biomechanics of a region from current anatomy (male/female, fat, muscle, breast, butt) */
  private paramsFor(r: TissueRegionState): {
    freq: number;
    zeta: number;
    maxDisp: number;
    maxInward: number;
    sagGain: number;
    compliance: number;
    outwardX: number;
  } {
    const { female, fat, muscle, breast, butt } = this.rig.anatomy;
    const jig = Math.max(0, this.settings.jiggle);
    const name = r.def.name;

    if (name === "breastL" || name === "breastR") {
      // Breasts: minimal on lean male chest; full natural breast dynamics on female / large cup / higher fat.
      // Biomechanical breast natural frequency ~2.3–3.9 Hz (lower with larger mass & higher fat).
      const volume = female * (0.55 + 0.85 * breast) + 0.14 * (1 - female) * (fat + 1.4 * breast) + 0.35 * fat * female;
      const compliance = Math.min(1.6, volume * (1 - 0.18 * muscle)) * jig;
      const freq = Math.max(2.1, 3.85 - 0.95 * breast - 0.45 * fat + 0.3 * muscle);
      const zeta = Math.max(0.16, 0.34 - 0.08 * breast - 0.06 * fat + 0.08 * muscle);
      const maxDisp = (0.75 + 2.5 * Math.min(1.4, volume)) * Math.min(1.4, Math.sqrt(Math.max(0.1, jig)));
      return { freq, zeta, maxDisp, maxInward: 0.75, sagGain: 0.42 * Math.min(1.4, volume), compliance, outwardX: 1 };
    }

    if (name === "buttL" || name === "buttR") {
      // Gluteal soft tissue: active on both sexes, heightened with butt volume, body fat, and female pelvis;
      // firmer and higher-frequency with stronger gluteal muscle tone.
      const volume = 0.32 + 0.75 * butt + 0.65 * fat + 0.25 * female;
      const tone = 1 - 0.28 * muscle;
      const compliance = Math.min(1.5, volume * tone) * jig;
      const freq = Math.max(2.6, 4.35 - 0.65 * butt - 0.7 * fat + 0.55 * muscle);
      const zeta = Math.max(0.18, 0.36 - 0.06 * fat - 0.04 * butt + 0.09 * muscle);
      const maxDisp = (0.65 + 1.85 * Math.min(1.3, volume * tone)) * Math.min(1.4, Math.sqrt(Math.max(0.1, jig)));
      return { freq, zeta, maxDisp, maxInward: 0.65, sagGain: 0.2 * Math.min(1.3, volume * tone), compliance, outwardX: -1 };
    }

    // Belly adipose pad: dormant when lean, soft sway & bounce when fat is raised
    const volume = Math.max(0, fat - 0.1) * (1 - 0.45 * muscle);
    const compliance = Math.min(1.4, volume * 1.15) * jig;
    const freq = Math.max(2.2, 3.5 - 0.7 * fat + 0.4 * muscle);
    const zeta = Math.max(0.2, 0.38 - 0.08 * fat + 0.1 * muscle);
    const maxDisp = (0.4 + 2.2 * volume) * Math.min(1.3, Math.sqrt(Math.max(0.1, jig)));
    return { freq, zeta, maxDisp, maxInward: 0.7, sagGain: 0.32 * volume, compliance, outwardX: 1 };
  }

  /** current rest anchor point of a tissue region in body space (cm) */
  private restAnchor(r: TissueRegionState, out: THREE.Vector3): THREE.Vector3 {
    const f = this.rig.anatomy.female;
    const cm = r.def.centerMale, cf = r.def.centerFemale;
    out.set(cm[0] + (cf[0] - cm[0]) * f, cm[1] + (cf[1] - cm[1]) * f, cm[2] + (cf[2] - cm[2]) * f);
    // slight outward shift with breast/butt/fat volume so the spring centre tracks the morph
    if (r.def.name === "breastL" || r.def.name === "breastR") {
      out.x += 2.2 * this.rig.anatomy.breast + 0.6 * this.rig.anatomy.fat;
      out.y -= 0.7 * this.rig.anatomy.breast;
    } else if (r.def.name === "buttL" || r.def.name === "buttR") {
      out.x -= 1.6 * this.rig.anatomy.butt + 0.8 * this.rig.anatomy.fat;
    } else if (r.def.name === "belly") {
      out.x += 1.8 * this.rig.anatomy.fat;
    }
    return out;
  }

  /** advance soft-tissue oscillators by `dt` seconds (call after `rig.pose(...)`) */
  step(dt: number): void {
    const rig = this.rig;
    if (!this.settings.enabled || this.settings.jiggle <= 1e-4) {
      for (const r of this.regions) {
        r.dispWorld.set(0, 0, 0);
        r.dispLocal.set(0, 0, 0);
        r.magnitude = 0;
        r.compliance = 0;
      }
      this.seeded = false;
      return;
    }

    const clampedDt = Math.max(0, Math.min(MAX_DT, dt));
    const subDt = clampedDt / SUBSTEPS;

    for (const r of this.regions) {
      const bi = r.boneIndex;
      const p = this.paramsFor(r);
      r.compliance = p.compliance;

      // Rigid target centre in world space: S[bi] * (anchor - H[bi]) + P[bi]
      this.restAnchor(r, this.vAnchor);
      r.target.copy(this.vAnchor).sub(rig.H[bi]).applyQuaternion(rig.S[bi]).add(rig.P[bi]);

      const omega = 2 * Math.PI * p.freq;
      const kSpring = omega * omega;
      const cDamp = 2 * p.zeta * omega;
      // Static equilibrium sag under gravity: g_eff / omega^2, scaled by tissue softness
      const gEff = this.settings.gravity * p.sagGain * Math.min(1.2, p.compliance);
      const restSag = gEff / kSpring;
      r.sag = restSag;

      // Seed cleanly on first frame or after a teleport (pose switch)
      if (!this.seeded || r.pos.distanceTo(r.target) > TELEPORT_CM) {
        r.pos.copy(r.target);
        r.pos.y -= restSag;
        r.vel.set(0, 0, 0);
        if (r.def.name === "buttL" || r.def.name === "buttR") {
          const th = rig.bone(r.def.side < 0 ? "thigh.L" : "thigh.R");
          r.prevThighX = rig.P[rig.bone(r.def.side < 0 ? "shin.L" : "shin.R")].x - rig.P[th].x;
        }
      }

      // Gluteal femur coupling: when the ipsilateral thigh kicks back/forward, its acceleration
      // gently excites that glute cheek independently of the contralateral side.
      let gluteKick = 0;
      if ((r.def.name === "buttL" || r.def.name === "buttR") && clampedDt > 1e-4) {
        const side = r.def.side < 0 ? "L" : "R";
        const th = rig.bone(`thigh.${side}`), kn = rig.bone(`shin.${side}`);
        const thighX = rig.P[kn].x - rig.P[th].x;
        const dThigh = (thighX - r.prevThighX) / clampedDt;
        r.prevThighX = thighX;
        gluteKick = Math.max(-180, Math.min(180, dThigh * 0.45)) * p.compliance;
      }

      // Integrate 4 symplectic substeps (unconditionally stable for omega * subDt < 2)
      if (subDt > 0 && p.compliance > 1e-4) {
        for (let s = 0; s < SUBSTEPS; s++) {
          this.vAcc.copy(r.target).sub(r.pos).multiplyScalar(kSpring);
          this.vAcc.addScaledVector(r.vel, -cDamp);
          this.vAcc.y -= gEff;
          if (gluteKick) {
            this.vAcc.y += 0.35 * Math.abs(gluteKick);
            this.vAcc.x -= 0.25 * gluteKick;
          }
          r.vel.addScaledVector(this.vAcc, subDt);
          r.pos.addScaledVector(r.vel, subDt);
        }
      } else if (p.compliance <= 1e-4) {
        r.pos.copy(r.target);
        r.vel.set(0, 0, 0);
      }

      // Convert raw world displacement (pos - target) into parent-bone local frame for anatomical constraints
      this.vDelta.copy(r.pos).sub(r.target);
      this.qInv.copy(rig.S[bi]).invert();
      this.vLocal.copy(this.vDelta).applyQuaternion(this.qInv);

      // Scale response by compliance (keeps flat male chest quiet while female/curvy/fat moves freely)
      const scale = Math.min(1.35, p.compliance);
      this.vLocal.multiplyScalar(scale);

      // Anatomical inward collision limit (cannot collapse deep into ribcage or pelvis)
      const outward = this.vLocal.x * p.outwardX;
      if (outward < -p.maxInward) {
        const clampedOut = -p.maxInward - 0.15 * Math.tanh((-outward - p.maxInward) / 0.5);
        this.vLocal.x = clampedOut * p.outwardX;
      }

      // Smooth radial saturation at maxDisp so rapid motions never invert quads
      const mag = this.vLocal.length();
      if (mag > 1e-6) {
        const limited = softClamp(mag, p.maxDisp);
        this.vLocal.multiplyScalar(limited / mag);
      }

      r.dispLocal.copy(this.vLocal);
      r.dispWorld.copy(this.vLocal).applyQuaternion(rig.S[bi]);
      r.magnitude = r.dispWorld.length();
    }

    this.seeded = true;
  }

  /**
   * Apply current soft-tissue displacements to a skinned vertex buffer (`n × 3`, in-place),
   * with smooth boundary falloff and volume-preserving squash-and-stretch.
   */
  apply(skinned: Float32Array): void {
    if (!this.settings.enabled || this.settings.jiggle <= 1e-4) return;
    const rig = this.rig;
    for (const r of this.regions) {
      if (r.magnitude < 1e-4 || r.compliance < 1e-4) continue;
      const bi = r.boneIndex;
      const S = rig.S[bi];
      const dx = r.dispWorld.x, dy = r.dispWorld.y, dz = r.dispWorld.z;
      const outSign = r.def.name === "buttL" || r.def.name === "buttR" ? -1 : 1;
      // Inward compression drives subtle radial bulge (Poisson-like volume preservation)
      const compress = Math.max(0, -(r.dispLocal.x * outSign));
      const bulge = 0.28 * compress / Math.max(3, r.def.radius);
      // Downward drop adds subtle forward/outward lower-pole pendulum swing
      const pend = -r.dispLocal.y * 0.025 * outSign;
      const cx = r.target.x, cy = r.target.y, cz = r.target.z;

      for (let k = 0; k < r.verts.length; k++) {
        const { idx, sw } = r.verts[k];
        const o = 3 * idx;
        let ox = sw * dx;
        let oy = sw * dy;
        let oz = sw * dz;
        if (bulge > 1e-4 || Math.abs(pend) > 1e-4) {
          this.vRel.set(skinned[o] - cx, skinned[o + 1] - cy, skinned[o + 2] - cz);
          this.qInv.copy(S).invert();
          this.vRel.applyQuaternion(this.qInv);
          // radial expansion in tangent plane (y, z) on inward squash + pendulum tilt on vertical bounce
          this.vOut.set(pend * Math.max(0, -this.vRel.y), bulge * this.vRel.y, bulge * this.vRel.z).applyQuaternion(S);
          ox += sw * this.vOut.x;
          oy += sw * this.vOut.y;
          oz += sw * this.vOut.z;
        }
        skinned[o] += ox;
        skinned[o + 1] += oy;
        skinned[o + 2] += oz;
      }
    }
  }
}
