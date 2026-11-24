/**
 * The rig drawn as a wireframe: the mesh's own quads as glowing lines (each triangle carries its
 * corners' places in their quad, so the diagonal isn't drawn), a glow where the surface turns
 * away, over a dark fill that hides what is behind; the far side seen faintly through the near
 * one. In CHIMERA's blue by default. For finding faults it can colour the lines by a score per
 * quad (`heat`: 1 where a fault starts to show) or by which bones move them (`weights`), and draw
 * the skeleton.
 *
 * The colours are worked in linear light and encoded for the screen here, to sit with the head's
 * lines (display values, as the app's): render with `renderer.outputColorSpace =
 * THREE.LinearSRGBColorSpace`.
 */
import * as THREE from "three";
import { vertexNormals, type Rig } from "./rig.ts";

export const WIRE_VERT = /* glsl */ `
attribute vec2 q;
attribute float heat;
attribute vec3 tint;
attribute float tone;
varying float vTone;
varying vec2 vQ;
varying vec3 vN;
varying vec3 vV;
varying float vHeat;
varying vec3 vTint;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalMatrix * normal;
  vV = -mv.xyz;
  vQ = q;
  vHeat = heat;
  vTint = tint;
  vTone = tone;
  gl_Position = projectionMatrix * mv;
}`;

export const WIRE_FRAG = /* glsl */ `
uniform vec3 uFill;
uniform vec3 uLine;
uniform vec3 uEdge;
uniform float uMode;
uniform float uBack;
uniform float uAlpha;
varying vec2 vQ;
varying vec3 vN;
varying vec3 vV;
varying float vHeat;
varying vec3 vTint;
varying float vTone;
vec3 heatColour(float h) {
  // calm to sore: the look's own line colour, then amber, red, white-hot
  vec3 c = mix(uLine * 0.55, vec3(1.0, 0.62, 0.1), smoothstep(0.35, 0.8, h));
  c = mix(c, vec3(1.0, 0.12, 0.08), smoothstep(0.8, 1.3, h));
  return mix(c, vec3(1.0, 0.9, 0.9), smoothstep(1.6, 2.6, h));
}
void main() {
  vec3 n = normalize(vN), v = normalize(vV);
  float facing = abs(dot(n, v));
  float rim = pow(1.0 - facing, 2.5);
  vec2 fw = max(fwidth(vQ), vec2(1e-6));
  vec2 dq = min(vQ, 1.0 - vQ) / fw;
  float wire = (1.0 - smoothstep(0.35, 1.15, min(dq.x, dq.y))) * (1.0 - smoothstep(0.16, 0.42, max(fw.x, fw.y)));
  vec3 line = uLine, edge = uEdge, fill = uFill;
  if (uMode > 0.5 && uMode < 1.5) {
    line = heatColour(vHeat);
    fill = mix(uFill, vec3(0.35, 0.03, 0.02), smoothstep(0.8, 1.6, vHeat) * 0.6);
  } else if (uMode > 1.5) {
    line = vTint;
    fill = uFill + 0.12 * vTint;
  }
  vec3 col = fill + edge * 1.3 * rim + line * wire * (0.55 + 0.45 * facing) * 1.4 * (uMode > 0.5 ? 1.0 : vTone);
  col = max(col, 0.0);
  col = mix(col * 12.92, 1.055 * pow(col, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, col));
  gl_FragColor = vec4(col, uAlpha * (1.0 - 0.75 * uBack));
}`;

const VERT = WIRE_VERT;
const FRAG = WIRE_FRAG;

export type Mode = "wire" | "heat" | "weights";
export interface Look {
  fill: [number, number, number];
  line: [number, number, number];
  edge: [number, number, number];
}
/** CHIMERA's blue: the app's #4fb6f2 for the lines, its mid blue at the silhouette (linear light) */
export const CHIMERA_LOOK: Readonly<Look> = { fill: [0.003, 0.008, 0.016], line: [0.06, 0.36, 0.68], edge: [0.03, 0.2, 0.45] };

/** Precomputed Catmull-Clark 1× subdivision stencil for real-time 4× quad density on curvy/fat bodies. */
interface SubdivCache {
  readonly n: number;
  readonly nq: number;
  readonly ne: number;
  readonly nSub: number;
  readonly nqSub: number;
  /** edge endpoints and adjacent faces (-1 if open boundary edge) */
  readonly eA: Uint32Array;
  readonly eB: Uint32Array;
  readonly eF0: Int32Array;
  readonly eF1: Int32Array;
  /** per-vertex incident neighbors and faces, packed CSR */
  readonly vNearOff: Uint32Array;
  readonly vNear: Uint32Array;
  readonly vFaceOff: Uint32Array;
  readonly vFace: Uint32Array;
  /** open-edge neighbors per vertex (-1 if interior) */
  readonly vOpen0: Int32Array;
  readonly vOpen1: Int32Array;
  /** subdivided quads (nqSub × 4) and triangle corner indices (6 * nqSub) */
  readonly quads: Uint32Array;
  readonly corner: Uint32Array;
  readonly pos: Float32Array;
  readonly norm: Float32Array;
  /** how far each of the mesh's own vertices was smoothed off its place (xyz), to put it back */
  readonly shift: Float32Array;
  readonly geo: THREE.BufferGeometry;
}

function buildSubdiv(rig: Rig): SubdivCache {
  const { n, nq, quads: Q } = rig;
  const edgeMap = new Map<number, number>();
  const eAList: number[] = [], eBList: number[] = [], eF0List: number[] = [], eF1List: number[] = [];
  const edgeKey = (a: number, b: number) => Math.min(a, b) * 1048576 + Math.max(a, b);
  const quadEdges = new Uint32Array(4 * nq);

  for (let f = 0; f < nq; f++) {
    for (let j = 0; j < 4; j++) {
      const a = Q[4 * f + j], b = Q[4 * f + ((j + 1) % 4)];
      const k = edgeKey(a, b);
      let ei = edgeMap.get(k);
      if (ei === undefined) {
        ei = eAList.length;
        edgeMap.set(k, ei);
        eAList.push(a);
        eBList.push(b);
        eF0List.push(f);
        eF1List.push(-1);
      } else {
        eF1List[ei] = f;
      }
      quadEdges[4 * f + j] = ei;
    }
  }
  const ne = eAList.length;
  const eA = Uint32Array.from(eAList), eB = Uint32Array.from(eBList);
  const eF0 = Int32Array.from(eF0List), eF1 = Int32Array.from(eF1List);

  const nearList: number[][] = Array.from({ length: n }, () => []);
  const faceList: number[][] = Array.from({ length: n }, () => []);
  const vOpen0 = new Int32Array(n).fill(-1), vOpen1 = new Int32Array(n).fill(-1);
  for (let ei = 0; ei < ne; ei++) {
    const a = eA[ei], b = eB[ei];
    nearList[a].push(b);
    nearList[b].push(a);
    if (eF1[ei] < 0) {
      if (vOpen0[a] < 0) vOpen0[a] = b;
      else vOpen1[a] = b;
      if (vOpen0[b] < 0) vOpen0[b] = a;
      else vOpen1[b] = a;
    }
  }
  for (let f = 0; f < nq; f++) for (let j = 0; j < 4; j++) faceList[Q[4 * f + j]].push(f);

  const vNearOff = new Uint32Array(n + 1), vFaceOff = new Uint32Array(n + 1);
  let totalNear = 0, totalFace = 0;
  for (let i = 0; i < n; i++) {
    vNearOff[i] = totalNear;
    totalNear += nearList[i].length;
    vFaceOff[i] = totalFace;
    totalFace += faceList[i].length;
  }
  vNearOff[n] = totalNear;
  vFaceOff[n] = totalFace;
  const vNear = new Uint32Array(totalNear), vFace = new Uint32Array(totalFace);
  for (let i = 0; i < n; i++) {
    vNear.set(nearList[i], vNearOff[i]);
    vFace.set(faceList[i], vFaceOff[i]);
  }

  const nSub = n + nq + ne;
  const nqSub = 4 * nq;
  const quads = new Uint32Array(4 * nqSub);
  for (let f = 0; f < nq; f++) {
    const fv = n + f;
    for (let j = 0; j < 4; j++) {
      const v = Q[4 * f + j];
      const eNext = n + nq + quadEdges[4 * f + j];
      const ePrev = n + nq + quadEdges[4 * f + ((j + 3) % 4)];
      const o = 4 * (4 * f + j);
      quads[o] = v;
      quads[o + 1] = eNext;
      quads[o + 2] = fv;
      quads[o + 3] = ePrev;
    }
  }
  const N = 6 * nqSub;
  const corner = new Uint32Array(N);
  const QC = new Float32Array(2 * N);
  const AT = [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]];
  for (let f = 0; f < nqSub; f++) {
    [0, 1, 2, 0, 2, 3].forEach((c, j) => {
      corner[6 * f + j] = quads[4 * f + c];
      QC.set(AT[j], 2 * (6 * f + j));
    });
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * N), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(3 * N), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("q", new THREE.BufferAttribute(QC, 2));
  geo.setAttribute("heat", new THREE.BufferAttribute(new Float32Array(N), 1).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("tint", new THREE.BufferAttribute(new Float32Array(3 * N), 3));
  geo.setAttribute("tone", new THREE.BufferAttribute(new Float32Array(N), 1));
  return { n, nq, ne, nSub, nqSub, eA, eB, eF0, eF1, vNearOff, vNear, vFaceOff, vFace, vOpen0, vOpen1, quads, corner, pos: new Float32Array(3 * nSub), norm: new Float32Array(3 * nSub), shift: new Float32Array(3 * n), geo };
}

/** evaluate Catmull-Clark stencil on a flat `stride`-component array (1 for tone, 3 for xyz/tint) */
function evalSubdiv(s: SubdivCache, Q: Uint32Array, src: ArrayLike<number>, stride: number, out: Float32Array): void {
  const { n, nq, ne, eA, eB, eF0, eF1, vNearOff, vNear, vFaceOff, vFace, vOpen0, vOpen1 } = s;
  // 1. Face points
  for (let f = 0; f < nq; f++) {
    const a = Q[4 * f] * stride, b = Q[4 * f + 1] * stride, c = Q[4 * f + 2] * stride, d = Q[4 * f + 3] * stride;
    const o = (n + f) * stride;
    for (let k = 0; k < stride; k++) out[o + k] = 0.25 * (src[a + k] + src[b + k] + src[c + k] + src[d + k]);
  }
  // 2. Edge points
  for (let ei = 0; ei < ne; ei++) {
    const a = eA[ei] * stride, b = eB[ei] * stride, o = (n + nq + ei) * stride;
    if (eF1[ei] >= 0) {
      const f0 = (n + eF0[ei]) * stride, f1 = (n + eF1[ei]) * stride;
      for (let k = 0; k < stride; k++) out[o + k] = 0.25 * (src[a + k] + src[b + k] + out[f0 + k] + out[f1 + k]);
    } else {
      for (let k = 0; k < stride; k++) out[o + k] = 0.5 * (src[a + k] + src[b + k]);
    }
  }
  // 3. Original vertices
  for (let v = 0; v < n; v++) {
    const o = v * stride;
    if (vOpen0[v] >= 0) {
      const a = vOpen0[v] * stride, b = (vOpen1[v] >= 0 ? vOpen1[v] : vOpen0[v]) * stride;
      for (let k = 0; k < stride; k++) out[o + k] = 0.75 * src[o + k] + 0.125 * (src[a + k] + src[b + k]);
      continue;
    }
    const n0 = vNearOff[v], n1 = vNearOff[v + 1], kVal = n1 - n0;
    const f0 = vFaceOff[v], f1 = vFaceOff[v + 1], nf = f1 - f0;
    if (kVal === 0 || nf === 0) {
      for (let c = 0; c < stride; c++) out[o + c] = src[o + c];
      continue;
    }
    const invK = 1 / kVal, invK2 = invK * invK, invKF = invK / nf;
    const selfW = (kVal - 3) * invK + kVal * invK2;
    for (let c = 0; c < stride; c++) {
      let acc = selfW * src[o + c];
      for (let j = n0; j < n1; j++) acc += invK2 * src[vNear[j] * stride + c];
      for (let j = f0; j < f1; j++) acc += invKF * out[(n + vFace[j]) * stride + c];
      out[o + c] = acc;
    }
  }
}

/**
 * Put the subdivided surface back through the mesh it was made from. Catmull-Clark smooths a
 * mesh inwards: a full breast's apex drew 1.4 cm inside its own vertices, the nose's tip 0.75,
 * and the lips' cut 0.2 cm off the lips the head draws, while the head's lines, the muscles and
 * the tissue's rings stayed on the mesh. Each of the mesh's vertices goes back where it was, and
 * each new point moves by the mean of its corners' moves, so the surface stays as smooth.
 */
function throughMesh(s: SubdivCache, Q: Uint32Array, src: ArrayLike<number>, out: Float32Array): void {
  const { n, nq, ne, eA, eB, shift } = s;
  for (let i = 0; i < 3 * n; i++) {
    shift[i] = src[i] - out[i];
    out[i] = src[i];
  }
  for (let f = 0; f < nq; f++) {
    const a = 3 * Q[4 * f], b = 3 * Q[4 * f + 1], c = 3 * Q[4 * f + 2], d = 3 * Q[4 * f + 3], o = 3 * (n + f);
    for (let k = 0; k < 3; k++) out[o + k] += 0.25 * (shift[a + k] + shift[b + k] + shift[c + k] + shift[d + k]);
  }
  for (let ei = 0; ei < ne; ei++) {
    const a = 3 * eA[ei], b = 3 * eB[ei], o = 3 * (n + nq + ei);
    for (let k = 0; k < 3; k++) out[o + k] += 0.5 * (shift[a + k] + shift[b + k]);
  }
}

function quadNormals(n: number, nq: number, Q: Uint32Array, P: Float32Array, out: Float32Array): void {
  out.fill(0);
  for (let f = 0; f < nq; f++) {
    const a = 3 * Q[4 * f], b = 3 * Q[4 * f + 1], c = 3 * Q[4 * f + 2], d = 3 * Q[4 * f + 3];
    const ux = P[c] - P[a], uy = P[c + 1] - P[a + 1], uz = P[c + 2] - P[a + 2];
    const vx = P[d] - P[b], vy = P[d + 1] - P[b + 1], vz = P[d + 2] - P[b + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    out[a] += nx; out[a + 1] += ny; out[a + 2] += nz;
    out[b] += nx; out[b + 1] += ny; out[b + 2] += nz;
    out[c] += nx; out[c + 1] += ny; out[c + 2] += nz;
    out[d] += nx; out[d + 1] += ny; out[d + 2] += nz;
  }
  for (let i = 0; i < n; i++) {
    const o = 3 * i, l = Math.hypot(out[o], out[o + 1], out[o + 2]) || 1;
    out[o] /= l; out[o + 1] /= l; out[o + 2] /= l;
  }
}

export class RigView {
  readonly group = new THREE.Group();
  readonly posed: Float32Array<ArrayBufferLike>;
  readonly normals: Float32Array<ArrayBufferLike>;
  private readonly geo = new THREE.BufferGeometry();
  private readonly front: THREE.Mesh;
  private readonly back: THREE.Mesh;
  private readonly mat: THREE.ShaderMaterial;
  private readonly backMat: THREE.ShaderMaterial;
  private readonly bones: THREE.LineSegments;
  private readonly joints: THREE.Points;
  /** each corner's vertex */
  private readonly corner: Uint32Array;
  private sub: SubdivCache | null = null;
  private subdivLevel: 0 | 1 = 0;
  private lastWeightBone = -1;

  constructor(readonly rig: Rig) {
    const nq = rig.nq, N = 6 * nq;
    this.posed = rig.rest.slice();
    this.normals = new Float32Array(3 * rig.n);
    this.corner = new Uint32Array(N);
    const QC = new Float32Array(2 * N);
    const AT = [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]];
    for (let f = 0; f < nq; f++) {
      [0, 1, 2, 0, 2, 3].forEach((c, j) => {
        this.corner[6 * f + j] = rig.quads[4 * f + c];
        QC.set(AT[j], 2 * (6 * f + j));
      });
    }
    this.geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * N), 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(3 * N), 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("q", new THREE.BufferAttribute(QC, 2));
    this.geo.setAttribute("heat", new THREE.BufferAttribute(new Float32Array(N), 1).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("tint", new THREE.BufferAttribute(new Float32Array(3 * N), 3));
    this.geo.setAttribute("tone", new THREE.BufferAttribute(Float32Array.from(this.corner, (v) => rig.tone[v]), 1));
    const uniforms = () => ({
      uFill: { value: new THREE.Vector3() },
      uLine: { value: new THREE.Vector3() },
      uEdge: { value: new THREE.Vector3() },
      uMode: { value: 0 },
      uBack: { value: 0 },
      uAlpha: { value: 0.92 },
    });
    this.mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: uniforms(), transparent: true, side: THREE.FrontSide });
    this.backMat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: uniforms(), transparent: true, side: THREE.BackSide, depthWrite: false });
    this.backMat.uniforms.uBack.value = 1;
    this.back = new THREE.Mesh(this.geo, this.backMat);
    this.front = new THREE.Mesh(this.geo, this.mat);
    this.back.renderOrder = -3;
    this.front.renderOrder = -2;
    this.back.frustumCulled = this.front.frustumCulled = false;
    this.group.add(this.back, this.front);
    const bg = new THREE.BufferGeometry();
    bg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6 * rig.B), 3));
    this.bones = new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color: 0x7fffd4, depthTest: false, transparent: true, opacity: 0.9 }));
    this.bones.renderOrder = 10;
    this.bones.frustumCulled = false;
    const jg = new THREE.BufferGeometry();
    jg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * rig.B), 3));
    this.joints = new THREE.Points(jg, new THREE.PointsMaterial({ color: 0xffffff, size: 5, sizeAttenuation: false, depthTest: false }));
    this.joints.renderOrder = 11;
    this.joints.frustumCulled = false;
    this.bones.visible = this.joints.visible = false;
    this.group.add(this.bones, this.joints);
    this.look = CHIMERA_LOOK;
    this.weightsOf(-1);
  }

  set look(l: Look) {
    for (const m of [this.mat, this.backMat]) {
      m.uniforms.uFill.value.set(...l.fill);
      m.uniforms.uLine.value.set(...l.line);
      m.uniforms.uEdge.value.set(...l.edge);
    }
  }
  set mode(m: Mode) {
    for (const mat of [this.mat, this.backMat]) mat.uniforms.uMode.value = m === "wire" ? 0 : m === "heat" ? 1 : 2;
  }
  set showBones(on: boolean) {
    this.bones.visible = this.joints.visible = on;
  }
  set backFaces(on: boolean) {
    this.back.visible = on;
  }
  set alpha(a: number) {
    for (const m of [this.mat, this.backMat]) m.uniforms.uAlpha.value = a;
  }
  /** 0: base mesh; 1: real-time 4× Catmull-Clark subdivision for smooth high-polygon curves */
  get subdivide(): 0 | 1 {
    return this.subdivLevel;
  }
  set subdivide(level: 0 | 1) {
    if (level === this.subdivLevel) return;
    this.subdivLevel = level;
    if (level === 1 && !this.sub) {
      this.sub = buildSubdiv(this.rig);
      this.toneChanged();
      this.weightsOf(this.lastWeightBone);
    }
    const g = level === 1 ? this.sub!.geo : this.geo;
    this.front.geometry = this.back.geometry = g;
  }
  /** the subdivided surface as last drawn (xyz; its first `rig.n` points are the mesh's own vertices), or null when the mesh is drawn as it is */
  get subdivided(): Float32Array | null {
    return this.subdivLevel === 1 && this.sub ? this.sub.pos : null;
  }
  /** current rendered quad count (4× when subdivided) */
  get quadCount(): number {
    return this.subdivLevel === 1 && this.sub ? this.sub.nqSub : this.rig.nq;
  }

  /** free the GPU's copies of its geometry and materials (take `group` out of the scene too) */
  dispose(): void {
    this.geo.dispose();
    this.sub?.geo.dispose();
    for (const m of [this.mat, this.backMat]) m.dispose();
    for (const o of [this.bones, this.joints]) {
      o.geometry.dispose();
      (o.material as THREE.Material).dispose();
    }
  }

  /** the lines' brightness again from the rig's tone (after it changed) */
  toneChanged(): void {
    const T = this.geo.getAttribute("tone") as THREE.BufferAttribute, A = T.array as Float32Array;
    for (let c = 0; c < this.corner.length; c++) A[c] = this.rig.tone[this.corner[c]];
    T.needsUpdate = true;
    if (this.sub) {
      const subTone = new Float32Array(this.sub.nSub);
      evalSubdiv(this.sub, this.rig.quads, this.rig.tone, 1, subTone);
      const ST = this.sub.geo.getAttribute("tone") as THREE.BufferAttribute, SA = ST.array as Float32Array;
      for (let c = 0; c < this.sub.corner.length; c++) SA[c] = subTone[this.sub.corner[c]];
      ST.needsUpdate = true;
    }
  }

  /** the weights view: every bone its own colour (bone < 0), or one bone's weight as brightness */
  weightsOf(bone: number): void {
    this.lastWeightBone = bone;
    const rig = this.rig, T = this.geo.getAttribute("tint") as THREE.BufferAttribute, A = T.array as Float32Array;
    const hue = (b: number) => new THREE.Color().setHSL(((b * 0.61803) % 1 + 1) % 1, 0.85, 0.55);
    const cols = rig.bones.map((_, b) => hue(b));
    const per = new Float32Array(3 * rig.n);
    for (let i = 0; i < rig.n; i++) {
      for (let j = 0; j < rig.INF; j++) {
        const w = rig.skinWeight[rig.INF * i + j], b = rig.skinIndex[rig.INF * i + j];
        if (!w) continue;
        if (bone < 0) {
          per[3 * i] += w * cols[b].r;
          per[3 * i + 1] += w * cols[b].g;
          per[3 * i + 2] += w * cols[b].b;
        } else if (b === bone) {
          const c = new THREE.Color().setHSL(0.66 - 0.66 * w, 1, 0.25 + 0.35 * w);
          per.set([c.r, c.g, c.b], 3 * i);
        }
      }
      if (bone >= 0 && per[3 * i] + per[3 * i + 1] + per[3 * i + 2] === 0) per.set([0.05, 0.05, 0.2], 3 * i);
    }
    for (let c = 0; c < this.corner.length; c++) A.set(per.subarray(3 * this.corner[c], 3 * this.corner[c] + 3), 3 * c);
    T.needsUpdate = true;
    if (this.sub) {
      const subTint = new Float32Array(3 * this.sub.nSub);
      evalSubdiv(this.sub, rig.quads, per, 3, subTint);
      const ST = this.sub.geo.getAttribute("tint") as THREE.BufferAttribute, SA = ST.array as Float32Array;
      for (let c = 0; c < this.sub.corner.length; c++) SA.set(subTint.subarray(3 * this.sub.corner[c], 3 * this.sub.corner[c] + 3), 3 * c);
      ST.needsUpdate = true;
    }
  }

  /** draw the mesh at `posed` (call after rig.skin into view.posed), with each quad's heat */
  update(heat?: Float32Array): void {
    const rig = this.rig, P = this.posed;
    if (this.subdivLevel === 1 && this.sub) {
      const s = this.sub;
      evalSubdiv(s, rig.quads, P, 3, s.pos);
      throughMesh(s, rig.quads, P, s.pos);
      quadNormals(s.nSub, s.nqSub, s.quads, s.pos, s.norm);
      // the mesh's own normals are kept up either way: others read them
      vertexNormals(rig, P, this.normals);
      const gp = s.geo.getAttribute("position") as THREE.BufferAttribute, gn = s.geo.getAttribute("normal") as THREE.BufferAttribute, gh = s.geo.getAttribute("heat") as THREE.BufferAttribute;
      const A = gp.array as Float32Array, NN = gn.array as Float32Array, HH = gh.array as Float32Array;
      for (let c = 0; c < s.corner.length; c++) {
        const v = 3 * s.corner[c];
        A[3 * c] = s.pos[v];
        A[3 * c + 1] = s.pos[v + 1];
        A[3 * c + 2] = s.pos[v + 2];
        NN[3 * c] = s.norm[v];
        NN[3 * c + 1] = s.norm[v + 1];
        NN[3 * c + 2] = s.norm[v + 2];
        if (heat) HH[c] = heat[Math.floor(c / 24)];
      }
      gp.needsUpdate = gn.needsUpdate = true;
      if (heat) gh.needsUpdate = true;
    } else {
      vertexNormals(rig, P, this.normals);
      const gp = this.geo.getAttribute("position") as THREE.BufferAttribute, gn = this.geo.getAttribute("normal") as THREE.BufferAttribute, gh = this.geo.getAttribute("heat") as THREE.BufferAttribute;
      const A = gp.array as Float32Array, NN = gn.array as Float32Array, HH = gh.array as Float32Array;
      for (let c = 0; c < this.corner.length; c++) {
        const v = 3 * this.corner[c];
        A[3 * c] = P[v];
        A[3 * c + 1] = P[v + 1];
        A[3 * c + 2] = P[v + 2];
        NN[3 * c] = this.normals[v];
        NN[3 * c + 1] = this.normals[v + 1];
        NN[3 * c + 2] = this.normals[v + 2];
        if (heat) HH[c] = heat[Math.floor(c / 6)];
      }
      gp.needsUpdate = gn.needsUpdate = true;
      if (heat) gh.needsUpdate = true;
    }
    // the skeleton
    const bp = this.bones.geometry.getAttribute("position") as THREE.BufferAttribute, jp = this.joints.geometry.getAttribute("position") as THREE.BufferAttribute;
    rig.bones.forEach((b, i) => {
      const par = b.parent < 0 ? i : b.parent;
      (bp.array as Float32Array).set([...rig.P[par].toArray(), ...rig.P[i].toArray()], 6 * i);
      (jp.array as Float32Array).set(rig.P[i].toArray(), 3 * i);
    });
    bp.needsUpdate = jp.needsUpdate = true;
  }
}
