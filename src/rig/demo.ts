/**
 * The body rig wearing the CHIMERA head, as a dev page: `bun run dev`, then /rig.html.
 * Demonstrates male & female anatomy, fat/breast/butt soft-tissue jiggle physics, CHIMERA-style
 * vector muscle system with real-time contraction firing & skin bulging, and procedural walking
 * (walk, catwalk, jog) plus static poses and imported Mixamo/BVH/glTF motion clips.
 * Dev only (not in the app's build). See docs/rig.md.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { initChimera } from "../wasm/chimera.ts";
import rigUrl from "./body.rig.bin?url";
import { Motion, readMotion } from "./motion.ts";
import { SoftTissue } from "./physics.ts";
import { MuscleSystem, type MuscleLoads } from "./muscles.ts";
import { GAIT_STYLES, walkFrame, type GaitStyle } from "./walk.ts";
import type { AnatomyParams, Rig as RigT } from "./rig.ts";
import type { RigHead } from "./head.ts";
import type { RigView as RigViewT } from "./view.ts";
import type { VoiceModel as VoiceModelT } from "../ui/analysis.ts";

await initChimera();
// the head reads the engine's tables as it loads, so it comes after the engine
const [{ Rig, vertexNormals }, { PRESETS }, { wearHead, HEAD_LOOKS, HEAD_LOOK }, { RigView }, { VoiceModel }, { PRESETS: BODIES }, { REST_ARTIC }] = await Promise.all([
  import("./rig.ts"),
  import("./pose.ts"),
  import("./head.ts"),
  import("./view.ts"),
  import("../ui/analysis.ts"),
  import("../ui/body.ts"),
  import("../ui/render/anatomy3d.ts"),
]);

interface BodyPreset {
  label: string;
  voice: "male" | "female";
  anatomy: AnatomyParams;
  jiggle: number;
}
const BODY_PRESETS: Record<string, BodyPreset> = {
  "female": { label: "Female — natural", voice: "female", anatomy: { female: 1, fat: 0.35, muscle: 0.45, breast: 0.55, butt: 0.55 }, jiggle: 1.15 },
  "female-curvy": { label: "Female — curvy", voice: "female", anatomy: { female: 1, fat: 0.72, muscle: 0.25, breast: 0.95, butt: 1.0 }, jiggle: 1.45 },
  "female-athletic": { label: "Female — athletic", voice: "female", anatomy: { female: 1, fat: 0.1, muscle: 0.95, breast: 0.35, butt: 0.65 }, jiggle: 0.85 },
  "male": { label: "Male — natural", voice: "male", anatomy: { female: 0, fat: 0.22, muscle: 0.55, breast: 0, butt: 0.25 }, jiggle: 0.9 },
  "male-athletic": { label: "Male — muscular", voice: "male", anatomy: { female: 0, fat: 0.08, muscle: 1.1, breast: 0, butt: 0.45 }, jiggle: 0.7 },
  "male-heavy": { label: "Male — heavy", voice: "male", anatomy: { female: 0, fat: 0.95, muscle: 0.3, breast: 0.1, butt: 0.65 }, jiggle: 1.3 },
};

const VIEW_MODES = ["anatomy", "muscles", "skin", "bones", "weights"] as const;
type ViewMode = (typeof VIEW_MODES)[number];

const canvas = document.querySelector("canvas")!;
const hud = document.getElementById("hud")!;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(32, 1, 5, 5000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(512, 512), 0.25, 0.3, 0.7));
composer.addPass(new OutputPass());

// Scrolling NORAD vector floor grid beneath the walker
const GRID_STEP = 20;
const floorGrid = (() => {
  const half = 220, pts: number[] = [];
  for (let i = -half; i <= half; i += GRID_STEP) {
    pts.push(i, 0, -half, i, 0, half);
    pts.push(-half, 0, i, half, 0, i);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x1f5f8c, transparent: true, opacity: 0.55 }));
  lines.renderOrder = -5;
  return lines;
})();
scene.add(floorGrid);

// Fetch body.rig.bin once so switching male <-> female is instantaneous
const rigBuf = await (await fetch(rigUrl)).arrayBuffer();

let rig!: RigT;
let model!: VoiceModelT;
let head!: RigHead;
let view!: RigViewT;
let tissue!: SoftTissue;
let muscles!: MuscleSystem;
let skinned!: Float32Array;
let skinNormals!: Float32Array;
let floorY = -85;

// Form wiring
const form = document.querySelector("form")!;
const field = (name: string) => form.elements.namedItem(name) as HTMLSelectElement & HTMLInputElement;
const playBtn = form.elements.namedItem("play") as HTMLButtonElement;

for (const [k, p] of Object.entries(BODY_PRESETS)) field("body").append(new Option(p.label, k));
const animOptions = [...GAIT_STYLES, ...Object.keys(PRESETS)];
field("pose").append(...animOptions.map((k) => new Option(GAIT_STYLES.includes(k as GaitStyle) ? `▶ ${k}` : k, k)));
field("view").append(...VIEW_MODES.map((k) => new Option(k, k)));
field("look").append(...Object.keys(HEAD_LOOKS).map((k) => new Option(k, k)));

field("body").value = "female";
field("pose").value = "walk";
field("view").value = "anatomy";
field("look").value = HEAD_LOOK;

const syncSliderLabels = () => {
  for (const el of form.querySelectorAll<HTMLElement>(".val[data-for]")) {
    const key = el.dataset.for!;
    const inp = field(key);
    if (inp) el.textContent = Number(inp.value).toFixed(key === "speed" ? 1 : 2);
  }
};

let currentSex: "male" | "female" = "female";
const readAnatomy = (): AnatomyParams => ({
  female: currentSex === "female" ? 1 : 0,
  fat: Number(field("fat").value),
  muscle: Number(field("muscle").value),
  breast: Number(field("breast").value),
  butt: Number(field("butt").value),
});

const applyViewMode = (mode: ViewMode) => {
  if (!view || !muscles) return;
  if (mode === "anatomy") {
    view.mode = "wire";
    view.showBones = false;
    view.alpha = 0.8;
    muscles.look.gain = 0.92;
    muscles.look.tissueRings = true;
  } else if (mode === "muscles") {
    view.mode = "wire";
    view.showBones = false;
    view.alpha = 0.34;
    muscles.look.gain = 1.25;
    muscles.look.tissueRings = true;
  } else if (mode === "skin") {
    view.mode = "wire";
    view.showBones = false;
    view.alpha = 0.92;
    muscles.look.gain = 0;
    muscles.look.tissueRings = false;
  } else if (mode === "bones") {
    view.mode = "wire";
    view.showBones = true;
    view.alpha = 0.45;
    muscles.look.gain = 0.35;
    muscles.look.tissueRings = false;
  } else {
    view.mode = "weights";
    view.showBones = false;
    view.alpha = 0.92;
    muscles.look.gain = 0;
    muscles.look.tissueRings = false;
  }
};

function buildCharacter(sex: "male" | "female", anatomy: AnatomyParams, keepCamera = false): void {
  if (view) {
    scene.remove(view.group, head.mount, muscles.group);
    view.dispose();
    head.dispose();
    model.dispose();
    muscles.dispose();
  }
  currentSex = sex;
  rig = new Rig(rigBuf.slice(0));
  rig.setAnatomy({ ...anatomy, female: sex === "female" ? 1 : 0 });
  model = new VoiceModel(BODIES[sex], 48000);
  head = wearHead(rig, model, { look: HEAD_LOOKS[field("look").value] });
  view = new RigView(rig);
  tissue = new SoftTissue(rig);
  tissue.settings.jiggle = Number(field("jiggle").value);
  muscles = new MuscleSystem(rig);
  scene.add(view.group, head.mount, muscles.group);
  skinned = new Float32Array(3 * rig.n);
  skinNormals = new Float32Array(3 * rig.n);

  // Calibrate floor level to the lowest foot vertex at anatomical neutral
  rig.pose(PRESETS.neutral);
  rig.skin(skinned);
  let minY = Infinity;
  for (let i = 0; i < rig.bodyVertices; i++) if (skinned[3 * i + 1] < minY) minY = skinned[3 * i + 1];
  floorY = minY;
  floorGrid.position.y = floorY;

  applyViewMode(field("view").value as ViewMode);
  const w = innerWidth, h = innerHeight;
  head.resize(w, h);
  muscles.resize(w, h);

  if (!keepCamera) setCamera("3/4");
}

function bodyMid(): THREE.Vector3 {
  const hips = rig.P[rig.bone("hips")];
  const chest = rig.P[rig.bone("upperchest")];
  return new THREE.Vector3((hips.x + chest.x) * 0.5, (hips.y + chest.y) * 0.5 - 22, (hips.z + chest.z) * 0.5);
}

function setCamera(which: string): void {
  const c = bodyMid();
  controls.target.copy(c);
  const dist = 365;
  if (which === "front") camera.position.copy(c).add(new THREE.Vector3(dist, 22, 0));
  else if (which === "side") camera.position.copy(c).add(new THREE.Vector3(0, 20, dist));
  else if (which === "back") camera.position.copy(c).add(new THREE.Vector3(-dist, 22, 42));
  else camera.position.copy(c).add(new THREE.Vector3(dist * 0.78, 28, dist * 0.6));
  controls.update();
}

for (const btn of form.querySelectorAll<HTMLButtonElement>("button[data-cam]")) {
  btn.onclick = () => setCamera(btn.dataset.cam!);
}

// Initial character: female walking with anatomy/muscles/jiggle active
const initPreset = BODY_PRESETS[field("body").value];
field("fat").value = String(initPreset.anatomy.fat);
field("muscle").value = String(initPreset.anatomy.muscle);
field("breast").value = String(initPreset.anatomy.breast);
field("butt").value = String(initPreset.anatomy.butt);
field("jiggle").value = String(initPreset.jiggle);
syncSliderLabels();
buildCharacter(initPreset.voice, initPreset.anatomy, false);

// Preset body selector (rebuilds head+body cleanly when sex changes)
field("body").onchange = () => {
  const p = BODY_PRESETS[field("body").value];
  if (!p) return;
  field("fat").value = String(p.anatomy.fat);
  field("muscle").value = String(p.anatomy.muscle);
  field("breast").value = String(p.anatomy.breast);
  field("butt").value = String(p.anatomy.butt);
  field("jiggle").value = String(p.jiggle);
  syncSliderLabels();
  if (p.voice !== currentSex) {
    buildCharacter(p.voice, p.anatomy, true);
  } else {
    rig.setAnatomy(p.anatomy);
    tissue.settings.jiggle = p.jiggle;
    muscles.fitAnatomy();
  }
};

// Live body-shaping sliders (zero-lag instant update without rebuilding the head)
for (const name of ["fat", "muscle", "breast", "butt"] as const) {
  field(name).oninput = () => {
    syncSliderLabels();
    rig.setAnatomy(readAnatomy());
  };
}
field("jiggle").oninput = () => {
  syncSliderLabels();
  tissue.settings.jiggle = Number(field("jiggle").value);
};
field("speed").oninput = () => syncSliderLabels();
field("view").onchange = () => applyViewMode(field("view").value as ViewMode);
field("look").onchange = () => {
  head.setLook(HEAD_LOOKS[field("look").value]);
  view.toneChanged();
};

let playing = true;
let walkT = 0;
let floorScroll = 0;
playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? "pause" : "play";
};

// Motion file loader (Mixamo FBX, BVH, glTF — physics & muscles run automatically on clips too!)
let motion: Motion | null = null, motionT = 0, picked = 0;
let file: { data: ArrayBuffer; name: string } | null = null;
const playClip = async (clip: number, read = file && readMotion(file.data, file.name)) => {
  const mine = ++picked;
  try {
    const { root, clips } = (await read)!;
    if (mine === picked) {
      [motion, motionT] = [new Motion(rig, root, clips[clip]), 0];
      tissue.reset();
    }
  } catch (e) {
    if (mine === picked) alert(e instanceof Error ? e.message : String(e));
  }
};
field("motion").onchange = async () => {
  const f = field("motion").files?.[0];
  if (!f) return;
  file = { data: await f.arrayBuffer(), name: f.name };
  const read = readMotion(file.data, file.name);
  const { clips } = await read.catch(() => ({ clips: [] as THREE.AnimationClip[] }));
  const longest = clips.reduce((a, c, i) => (c.duration > clips[a].duration ? i : a), 0);
  field("clip").replaceChildren(...clips.map((c, i) => new Option(`${c.name} (${c.duration.toFixed(1)} s)`, String(i), false, i === longest)));
  field("clip").hidden = clips.length < 2;
  await playClip(longest, read);
};
field("clip").onchange = () => void playClip(Number(field("clip").value));

field("pose").onchange = () => {
  motion = null;
  picked++;
  tissue.reset();
};

function resize(): void {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  head.resize(w, h);
  muscles.resize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

let last = performance.now();
let hudTick = 0;

renderer.setAnimationLoop((now) => {
  const dt = Math.max(0, Math.min(0.1, (now - last) / 1000));
  last = now;
  const simDt = playing ? dt : 0;
  const speed = Number(field("speed").value);

  let loads: MuscleLoads | undefined;
  const sel = field("pose").value;

  if (motion) {
    motion.pose((motionT += simDt * speed));
    const move = bodyMid().sub(controls.target).setY(0);
    controls.target.add(move);
    camera.position.add(move);
  } else if (GAIT_STYLES.includes(sel as GaitStyle)) {
    walkT += simDt;
    const wf = walkFrame(walkT, sel as GaitStyle, rig.anatomy, speed);
    rig.pose(wf.pose);
    loads = wf.loads;
    floorScroll = (floorScroll + wf.speedCmPerSec * simDt) % GRID_STEP;
    floorGrid.position.x = -floorScroll;
  } else {
    rig.pose(PRESETS[sel] ?? PRESETS.neutral);
  }

  controls.update();

  // Soft-tissue physics & muscle biomechanics
  tissue.step(simDt);
  muscles.update(simDt, loads);

  // Skin → muscle bulge → soft-tissue jiggle → CHIMERA head face
  rig.skin(skinned);
  vertexNormals(rig, skinned, skinNormals);
  muscles.deform(skinned, skinNormals);
  tissue.apply(skinned);

  const open = Number(field("mouth").value);
  head.update(camera, { ...REST_ARTIC, jaw: 1.8 * open, lipAp: 1.8 * open }, null, dt);
  head.face(skinned, view.posed);
  view.update();
  muscles.draw(tissue, camera);
  composer.render();

  // Lightweight HUD readout (~12 Hz)
  hudTick += dt;
  if (hudTick > 0.08) {
    hudTick = 0;
    const bL = tissue.region("breastL")?.magnitude ?? 0;
    const bR = tissue.region("breastR")?.magnitude ?? 0;
    const gL = tissue.region("buttL")?.magnitude ?? 0;
    const gR = tissue.region("buttR")?.magnitude ?? 0;
    const topMuscles = [...muscles.muscles]
      .sort((a, b) => b.activation - a.activation)
      .slice(0, 3)
      .filter((m) => m.activation > 0.08)
      .map((m) => `${m.name} ${(m.activation * 100).toFixed(0)}%`)
      .join(" · ");
    hud.innerHTML =
      `<b>CHIMERA RIG</b> · ${currentSex.toUpperCase()} · ${motion ? "clip" : sel}<br>` +
      `<span class="pk">TISSUE Δ</span> breast ${(Math.max(bL, bR) * 10).toFixed(1)} mm · butt ${(Math.max(gL, gR) * 10).toFixed(1)} mm<br>` +
      `<span class="hi">MUSCLES</span> ${topMuscles || "rest"}`;
  }
});
