/**
 * Male external pelvic anatomy (penis shaft, coronal glans & scrotum) rendered in CHIMERA's
 * phosphor-blue quad wireframe style (`WIRE_VERT` / `WIRE_FRAG`), anchored to the `hips` bone
 * at the pubic symphysis with continuous size scaling (`0`: hidden, `0.45`: S, `0.75`: M,
 * `1.1`: L, `1.45`: XL) and damped pendulum soft-tissue jiggle physics.
 */
import * as THREE from "three";
import type { Rig } from "./rig.ts";
import { CHIMERA_LOOK, WIRE_FRAG, WIRE_VERT, type Look } from "./view.ts";

export const PENIS_SIZES: Readonly<Record<string, number>> = {
  none: 0,
  S: 0.45,
  M: 0.78,
  L: 1.12,
  XL: 1.48,
};

const RAD_SEGS = 14;
const LEN_SEGS = 12;
const SCROTUM_U = 12;
const SCROTUM_V = 8;
const SUBSTEPS = 4;
const MAX_DT = 0.05;

/** shaft radial profile along normalized length u in [0, 1] (root → shaft → corona → rounded glans) */
function shaftProfile(u: number): number {
  if (u < 0.14) return 1.05 - 0.12 * (u / 0.14);
  if (u < 0.74) return 0.93 + 0.04 * Math.sin(((u - 0.14) / 0.6) * Math.PI);
  if (u < 0.84) {
    // subtle anatomical corona ridge
    const t = (u - 0.74) / 0.1;
    return 0.93 + 0.13 * Math.sin(t * Math.PI);
  }
  // smooth hemispherical glans dome tapering cleanly to rounded tip
  const t = (u - 0.84) / 0.16;
  return 0.96 * Math.sqrt(Math.max(0.008, 1 - t * t));
}

export class MaleAnatomy {
  readonly group = new THREE.Group();
  size = 0;
  jiggle = 1;
  gravity = 981;

  /** live pendulum tip displacement magnitude (cm) for telemetry */
  magnitude = 0;

  private readonly geo = new THREE.BufferGeometry();
  private readonly mat: THREE.ShaderMaterial;
  private readonly backMat: THREE.ShaderMaterial;
  private readonly front: THREE.Mesh;
  private readonly back: THREE.Mesh;

  // Topology: quad list & buffers
  private readonly nVerts: number;
  private readonly nQuads: number;
  private readonly quads: Uint32Array;
  private readonly corner: Uint32Array;
  private readonly pos: Float32Array;
  private readonly norm: Float32Array;

  // Pendulum oscillator state (world-space tip mass relative to pubic anchor)
  private readonly anchorWorld = new THREE.Vector3();
  private readonly tipTarget = new THREE.Vector3();
  private readonly tipPos = new THREE.Vector3();
  private readonly tipVel = new THREE.Vector3();
  private readonly dispLocal = new THREE.Vector3();
  private seeded = false;

  // Reusable scratch objects
  private readonly vAnchor = new THREE.Vector3();
  private readonly vAcc = new THREE.Vector3();
  private readonly vDelta = new THREE.Vector3();
  private readonly qInv = new THREE.Quaternion();
  private readonly vDir = new THREE.Vector3();
  private readonly vSide = new THREE.Vector3();
  private readonly vUp = new THREE.Vector3();
  private readonly vPt = new THREE.Vector3();

  constructor(readonly rig: Rig) {
    // Vertices: shaft (LEN_SEGS + 1) * RAD_SEGS + tip pole + scrotum (SCROTUM_V + 1) * SCROTUM_U
    const shaftVerts = (LEN_SEGS + 1) * RAD_SEGS;
    const scrotumOff = shaftVerts;
    const scrotumVerts = (SCROTUM_V + 1) * SCROTUM_U;
    this.nVerts = shaftVerts + scrotumVerts;

    const qList: number[] = [];
    // Shaft tube quads
    for (let r = 0; r < LEN_SEGS; r++) {
      for (let s = 0; s < RAD_SEGS; s++) {
        const s1 = (s + 1) % RAD_SEGS;
        const a = r * RAD_SEGS + s;
        const b = r * RAD_SEGS + s1;
        const c = (r + 1) * RAD_SEGS + s1;
        const d = (r + 1) * RAD_SEGS + s;
        qList.push(a, b, c, d);
      }
    }
    // Scrotum bilobed quad shell
    for (let v = 0; v < SCROTUM_V; v++) {
      for (let u = 0; u < SCROTUM_U; u++) {
        const u1 = (u + 1) % SCROTUM_U;
        const a = scrotumOff + v * SCROTUM_U + u;
        const b = scrotumOff + v * SCROTUM_U + u1;
        const c = scrotumOff + (v + 1) * SCROTUM_U + u1;
        const d = scrotumOff + (v + 1) * SCROTUM_U + u;
        qList.push(a, b, c, d);
      }
    }

    this.nQuads = qList.length / 4;
    this.quads = Uint32Array.from(qList);
    this.pos = new Float32Array(3 * this.nVerts);
    this.norm = new Float32Array(3 * this.nVerts);

    const N = 6 * this.nQuads;
    this.corner = new Uint32Array(N);
    const QC = new Float32Array(2 * N);
    const AT = [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]];
    for (let f = 0; f < this.nQuads; f++) {
      [0, 1, 2, 0, 2, 3].forEach((c, j) => {
        this.corner[6 * f + j] = this.quads[4 * f + c];
        QC.set(AT[j], 2 * (6 * f + j));
      });
    }

    this.geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * N), 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(3 * N), 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("q", new THREE.BufferAttribute(QC, 2));
    this.geo.setAttribute("heat", new THREE.BufferAttribute(new Float32Array(N), 1));
    this.geo.setAttribute("tint", new THREE.BufferAttribute(new Float32Array(3 * N).fill(0.4), 3));
    this.geo.setAttribute("tone", new THREE.BufferAttribute(new Float32Array(N).fill(1), 1));

    const uniforms = () => ({
      uFill: { value: new THREE.Vector3(...CHIMERA_LOOK.fill) },
      uLine: { value: new THREE.Vector3(...CHIMERA_LOOK.line) },
      uEdge: { value: new THREE.Vector3(...CHIMERA_LOOK.edge) },
      uMode: { value: 0 },
      uBack: { value: 0 },
      uAlpha: { value: 0.92 },
    });
    this.mat = new THREE.ShaderMaterial({ vertexShader: WIRE_VERT, fragmentShader: WIRE_FRAG, uniforms: uniforms(), transparent: true, side: THREE.FrontSide });
    this.backMat = new THREE.ShaderMaterial({ vertexShader: WIRE_VERT, fragmentShader: WIRE_FRAG, uniforms: uniforms(), transparent: true, side: THREE.BackSide, depthWrite: false });
    this.backMat.uniforms.uBack.value = 1;
    this.back = new THREE.Mesh(this.geo, this.backMat);
    this.front = new THREE.Mesh(this.geo, this.mat);
    this.back.renderOrder = -3;
    this.front.renderOrder = -2;
    this.back.frustumCulled = this.front.frustumCulled = false;
    this.group.add(this.back, this.front);
    this.group.visible = false;
  }

  set look(l: Look) {
    for (const m of [this.mat, this.backMat]) {
      m.uniforms.uFill.value.set(...l.fill);
      m.uniforms.uLine.value.set(...l.line);
      m.uniforms.uEdge.value.set(...l.edge);
    }
  }
  set alpha(a: number) {
    for (const m of [this.mat, this.backMat]) m.uniforms.uAlpha.value = a;
  }

  reset(): void {
    this.seeded = false;
    this.tipVel.set(0, 0, 0);
    this.dispLocal.set(0, 0, 0);
    this.magnitude = 0;
  }

  /** pubic symphysis rest anchor (body cm), adapting cleanly to male/female pelvis + fat */
  private restAnchor(out: THREE.Vector3): THREE.Vector3 {
    const { female, fat } = this.rig.anatomy;
    // Male pubic root: [10.1, 5.1, 0]; Female pubic root shifts down ~9.5 cm with pelvis
    const x = 10.1 + 0.5 * female + 1.1 * fat;
    const y = 5.1 - 9.6 * female;
    return out.set(x, y, 0);
  }

  /** advance pendulum soft-tissue physics and rebuild quad mesh for the current frame */
  update(dt: number): void {
    const s = Math.max(0, Math.min(1.6, this.size));
    if (s <= 0.01) {
      this.group.visible = false;
      this.magnitude = 0;
      this.seeded = false;
      return;
    }
    this.group.visible = true;

    const rig = this.rig;
    const hipsIdx = rig.bone("hips");
    const S = rig.S[hipsIdx], P = rig.P[hipsIdx], H = rig.H[hipsIdx];

    // Anatomical dimensions (cm) scaled smoothly by size s in [0.2..1.5]
    const scaleL = 0.55 + 0.72 * s;
    const scaleR = 0.65 + 0.52 * s;
    const length = 11.2 * scaleL;
    const radius = 1.65 * scaleR;

    // World position of pubic root
    this.restAnchor(this.vAnchor);
    this.anchorWorld.copy(this.vAnchor).sub(H).applyQuaternion(S).add(P);

    // Rest resting-flaccid tip offset in hips-local frame (curves slightly forward & hangs down)
    const restTipLocalX = 3.2 * scaleL;
    const restTipLocalY = -length * 0.88;
    this.tipTarget.set(restTipLocalX, restTipLocalY, 0).applyQuaternion(S).add(this.anchorWorld);

    const clampedDt = Math.max(0, Math.min(MAX_DT, dt));
    const jig = Math.max(0, this.jiggle);

    if (!this.seeded || this.tipPos.distanceTo(this.tipTarget) > 35 || jig <= 1e-4) {
      this.tipPos.copy(this.tipTarget);
      this.tipVel.set(0, 0, 0);
      this.seeded = true;
    } else if (clampedDt > 0) {
      // Natural pendulum frequency ~2.3–3.2 Hz (slower/heavier when larger)
      const freq = Math.max(2.0, 3.3 - 0.65 * s);
      const omega = 2 * Math.PI * freq;
      const kSpring = omega * omega;
      const zeta = 0.24;
      const cDamp = 2 * zeta * omega;
      const subDt = clampedDt / SUBSTEPS;
      for (let i = 0; i < SUBSTEPS; i++) {
        this.vAcc.copy(this.tipTarget).sub(this.tipPos).multiplyScalar(kSpring);
        this.vAcc.addScaledVector(this.tipVel, -cDamp);
        this.vAcc.y -= this.gravity * 0.08 * scaleL;
        this.tipVel.addScaledVector(this.vAcc, subDt);
        this.tipPos.addScaledVector(this.tipVel, subDt);
      }
    }

    // Local displacement relative to hips frame, with thigh/pelvis collision bounds
    this.vDelta.copy(this.tipPos).sub(this.tipTarget);
    this.qInv.copy(S).invert();
    this.dispLocal.copy(this.vDelta).applyQuaternion(this.qInv).multiplyScalar(Math.min(1.4, jig));

    // Gentle thigh-swing clearance during walking: thighs push tip slightly forward & sideways
    const thL = rig.P[rig.bone("shin.L")].x - rig.P[rig.bone("thigh.L")].x;
    const thR = rig.P[rig.bone("shin.R")].x - rig.P[rig.bone("thigh.R")].x;
    const thighPush = Math.max(0, Math.max(thL, thR) - 4) * 0.08 * scaleL;
    this.dispLocal.x = Math.max(-1.0 * scaleL, this.dispLocal.x + thighPush);
    const maxDisp = 3.2 * scaleL * Math.min(1.3, Math.sqrt(Math.max(0.1, jig)));
    const dMag = this.dispLocal.length();
    if (dMag > 1e-5) {
      const lim = maxDisp * Math.tanh(dMag / maxDisp);
      this.dispLocal.multiplyScalar(lim / dMag);
    }
    this.magnitude = this.dispLocal.length();

    // Build all-quad shaft + coronal glans in hips-local space, then transform by S + anchorWorld
    let vi = 0;
    const dx = this.dispLocal.x, dy = this.dispLocal.y, dz = this.dispLocal.z;
    for (let r = 0; r <= LEN_SEGS; r++) {
      const u = r / LEN_SEGS;
      // Smooth pendulous curve from pubic root (+x forward) arching down (-y) + live swing displacement
      const flex = u * u * 0.7 + u * 0.3;
      const cx = (3.4 * Math.sin(u * Math.PI * 0.62) + 0.8 * u) * scaleL + dx * flex;
      const cy = (-length * Math.pow(u, 1.15)) + dy * flex;
      const cz = dz * flex;
      // Analytical tangent for smooth ring orientation
      const u2 = Math.min(1, u + 0.02);
      const flex2 = u2 * u2 * 0.7 + u2 * 0.3;
      const nx = (3.4 * Math.sin(u2 * Math.PI * 0.62) + 0.8 * u2) * scaleL + dx * flex2;
      const ny = (-length * Math.pow(u2, 1.15)) + dy * flex2;
      const nz = dz * flex2;
      this.vDir.set(nx - cx, ny - cy, nz - cz).normalize();
      this.vSide.set(0, 0, 1);
      this.vUp.crossVectors(this.vSide, this.vDir).normalize();
      this.vSide.crossVectors(this.vDir, this.vUp).normalize();

      const rad = radius * shaftProfile(u);
      // Slight elliptical cross-section (anatomically subtly wider/flatter at base)
      const rx = rad * 0.96, rz = rad * 1.04;
      for (let k = 0; k < RAD_SEGS; k++) {
        const th = (2 * Math.PI * k) / RAD_SEGS;
        const cs = Math.cos(th), sn = Math.sin(th);
        this.vPt.set(
          cx + this.vUp.x * rx * cs + this.vSide.x * rz * sn,
          cy + this.vUp.y * rx * cs + this.vSide.y * rz * sn,
          cz + this.vUp.z * rx * cs + this.vSide.z * rz * sn,
        ).applyQuaternion(S).add(this.anchorWorld);
        this.pos[3 * vi] = this.vPt.x;
        this.pos[3 * vi + 1] = this.vPt.y;
        this.pos[3 * vi + 2] = this.vPt.z;
        vi++;
      }
    }

    // Build smooth bilobed scrotum tucked cleanly beneath shaft root
    const scX = -1.6 * scaleR + 0.32 * dx;
    const scY = -3.4 * scaleR + 0.42 * dy;
    const scZ = 0.38 * dz;
    const scRx = 2.45 * scaleR;
    const scRy = 2.85 * scaleR;
    const scRz = 2.85 * scaleR;
    for (let v = 0; v <= SCROTUM_V; v++) {
      const phi = (Math.PI * v) / SCROTUM_V; // 0 (top) to PI (bottom)
      const sp = Math.sin(phi), cp = Math.cos(phi);
      // Teardrop hang: slightly narrower at upper attachment neck, fuller at lower pole
      const hang = 0.72 + 0.34 * (v / SCROTUM_V);
      for (let u = 0; u < SCROTUM_U; u++) {
        const th = (2 * Math.PI * u) / SCROTUM_U;
        const cs = Math.cos(th), sn = Math.sin(th);
        // Subtle anatomical bilobed raphe indentation along z=0 at the bottom
        const cleft = 1 - 0.14 * Math.exp(-Math.pow(sn * 3.2, 2)) * Math.sin(phi);
        const lx = scX + scRx * sp * cs * hang;
        const ly = scY + scRy * cp * cleft;
        const lz = scZ + scRz * sp * sn * hang;
        this.vPt.set(lx, ly, lz).applyQuaternion(S).add(this.anchorWorld);
        this.pos[3 * vi] = this.vPt.x;
        this.pos[3 * vi + 1] = this.vPt.y;
        this.pos[3 * vi + 2] = this.vPt.z;
        vi++;
      }
    }

    // Area-weighted quad normals & GPU upload
    this.norm.fill(0);
    for (let f = 0; f < this.nQuads; f++) {
      const a = 3 * this.quads[4 * f], b = 3 * this.quads[4 * f + 1], c = 3 * this.quads[4 * f + 2], d = 3 * this.quads[4 * f + 3];
      const ux = this.pos[c] - this.pos[a], uy = this.pos[c + 1] - this.pos[a + 1], uz = this.pos[c + 2] - this.pos[a + 2];
      const vx = this.pos[d] - this.pos[b], vy = this.pos[d + 1] - this.pos[b + 1], vz = this.pos[d + 2] - this.pos[b + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      for (const o of [a, b, c, d]) {
        this.norm[o] += nx;
        this.norm[o + 1] += ny;
        this.norm[o + 2] += nz;
      }
    }
    for (let i = 0; i < this.nVerts; i++) {
      const o = 3 * i, l = Math.hypot(this.norm[o], this.norm[o + 1], this.norm[o + 2]) || 1;
      this.norm[o] /= l; this.norm[o + 1] /= l; this.norm[o + 2] /= l;
    }

    const gp = this.geo.getAttribute("position") as THREE.BufferAttribute;
    const gn = this.geo.getAttribute("normal") as THREE.BufferAttribute;
    const A = gp.array as Float32Array, NN = gn.array as Float32Array;
    for (let c = 0; c < this.corner.length; c++) {
      const v = 3 * this.corner[c];
      A[3 * c] = this.pos[v];
      A[3 * c + 1] = this.pos[v + 1];
      A[3 * c + 2] = this.pos[v + 2];
      NN[3 * c] = this.norm[v];
      NN[3 * c + 1] = this.norm[v + 1];
      NN[3 * c + 2] = this.norm[v + 2];
    }
    gp.needsUpdate = gn.needsUpdate = true;
  }

  dispose(): void {
    this.geo.dispose();
    this.mat.dispose();
    this.backMat.dispose();
  }
}
