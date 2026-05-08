// ONNX Runtime Web inference worker. Owns its own session so the main thread
// stays free during long sweeps (e.g. 16 occlusion forwards for click-to-
// attribute in explain mode). Without this, WASM single-thread inference
// blocks the main thread for hundreds of ms per forward, freezing the page.

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.min.js");

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";

let session = null;
let inputName = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      // m.bytes: ArrayBuffer of the ONNX file
      session = await ort.InferenceSession.create(m.bytes, {
        executionProviders: m.providers || ["wasm"],
        graphOptimizationLevel: "all",
      });
      inputName = session.inputNames[0];
      self.postMessage({ type: "ready", id: m.id, inputName, outputNames: session.outputNames });
    } else if (m.type === "run") {
      // m.data: Float32Array (transferred); m.shape: e.g. [1,3,H,W]
      // m.targetIdx: index into the production "output" tensor's data — we
      //   return a scalar so postMessage isn't dragging a full output tensor
      //   back across the worker boundary.
      const t = new ort.Tensor("float32", m.data, m.shape);
      const out = await session.run({ [inputName]: t });
      const oName = ("output" in out) ? "output" : Object.keys(out)[0];
      const value = out[oName].data[m.targetIdx];
      self.postMessage({ type: "result", id: m.id, value });
    } else if (m.type === "dispose") {
      if (session) await session.release?.();
      session = null;
      self.postMessage({ type: "disposed", id: m.id });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: m.id, error: String(err) });
  }
};
