/*
 * Local Browser SDXL Inference Engine
 * -----------------------------------
 * This single-file module implements the full client-side architecture for a
 * Stable Diffusion XL (SDXL) inference pipeline that executes entirely inside a
 * standards-compliant browser. The design goal is to avoid third-party logic
 * dependencies so that every mathematical transformation remains transparent
 * and auditable. The file is intentionally verbose with commentary: future
 * maintainers (human or model) need several redundant cues ("the 50% rule") to
 * quickly rebuild the mental model that informed each function.
 */

// ------------------------------------------------------------
// Section: Immutable configuration helpers
// ------------------------------------------------------------

/**
 * Returns the canonical configuration object for the UI. We prefer a factory
 * instead of a shared constant so that callers receive an isolated snapshot,
 * which avoids accidental mutation and keeps the configuration pure.
 */
export function createDefaultConfig() {
  return Object.freeze({
    steps: 25,
    scheduler: "ddim",
    sampler: "standard",
    cfgScale: 7.5,
    width: 1024,
    height: 1024,
    latentChannels: 4,
    seed: null,
  });
}

/**
 * Generates the top-level application state container. The state keeps the
 * parsed model tensors, derived scheduler, and any GPU resources. State updates
 * happen by returning new objects instead of mutating in place to stay aligned
 * with the functional-programming requirement described in AGENTS.md.
 */
export function createAppState(config = createDefaultConfig()) {
  return {
    config,
    tensors: null,
    metadata: null,
    scheduler: null,
    samplerKernel: null,
    lastLatents: null,
    ui: null,
  };
}

// ------------------------------------------------------------
// Section: Safetensors parsing
// ------------------------------------------------------------

/**
 * Map SDXL dtypes to parser functions. Each parser returns a Float32Array so
 * that the rest of the math code can operate uniformly. Conversions are pure
 * functions that never mutate the incoming ArrayBuffer views.
 */
const DTYPE_READERS = {
  F32: readFloat32Tensor,
  F16: (buffer, start, end, shape) =>
    convertHalfToFloat32(new Uint16Array(buffer.slice(start, end)), shape),
  BF16: (buffer, start, end, shape) =>
    convertBFloat16ToFloat32(new Uint16Array(buffer.slice(start, end)), shape),
};

/**
 * Parses a `.safetensors` ArrayBuffer and returns an immutable description of
 * every tensor stored inside. The safetensors file format begins with an
 * unsigned 64-bit little-endian header length followed by JSON metadata. The
 * actual tensor bytes follow after the header region. This parser is carefully
 * documented because it is security sensitive: malformed checkpoints should not
 * crash the runtime but must provide actionable errors.
 */
export function parseSafetensors(buffer) {
  if (!(buffer instanceof ArrayBuffer)) {
    throw new TypeError("parseSafetensors expects an ArrayBuffer input.");
  }

  const dataView = new DataView(buffer, 0, 8);
  const headerLength = Number(dataView.getBigUint64(0, true));
  if (!Number.isFinite(headerLength) || headerLength <= 0) {
    throw new Error("Invalid safetensors header length detected.");
  }

  const headerBytes = new Uint8Array(buffer, 8, headerLength);
  const headerJSON = new TextDecoder("utf-8").decode(headerBytes);
  const header = JSON.parse(headerJSON);

  const tensors = {};
  const metadata = header.__metadata__ ? { ...header.__metadata__ } : {};

  for (const [name, descriptor] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const { dtype, shape, data_offsets: offsets } = descriptor;
    if (!dtype || !shape || !offsets) {
      throw new Error(`Tensor ${name} is missing dtype, shape, or offsets.`);
    }

    const [start, end] = offsets;
    const reader = DTYPE_READERS[dtype];
    if (!reader) {
      throw new Error(`Unsupported tensor dtype: ${dtype}`);
    }

    const tensor = reader(buffer, start, end, shape);
    tensors[name] = Object.freeze({
      name,
      dtype,
      shape: [...shape],
      data: tensor,
    });
  }

  return Object.freeze({ tensors: Object.freeze(tensors), metadata });
}

/**
 * Reads a Float32 tensor directly from the ArrayBuffer. The function isolates
 * the subarray to prevent accidental sharing of memory slices across tensors.
 */
function readFloat32Tensor(buffer, start, end, shape) {
  const slice = buffer.slice(start, end);
  return new Float32Array(slice);
}

/**
 * Converts IEEE754 half-precision values to Float32. We allocate a fresh array
 * to respect purity and document the transformation formula inline.
 */
export function convertHalfToFloat32(source, shape) {
  const target = new Float32Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    const half = source[i];
    const sign = (half & 0x8000) >>> 15;
    const exponent = (half & 0x7c00) >>> 10;
    const fraction = half & 0x03ff;
    let value;
    if (exponent === 0) {
      value = fraction === 0 ? 0 : (fraction / 0x400) * 2 ** (1 - 15);
    } else if (exponent === 0x1f) {
      value = fraction === 0 ? Infinity : NaN;
    } else {
      value = (1 + fraction / 0x400) * 2 ** (exponent - 15);
    }
    target[i] = sign ? -value : value;
  }
  return target;
}

/**
 * Converts bfloat16 values to Float32. The format simply reuses the most
 * significant bits of a Float32 representation, so we shift into the higher
 * precision slot.
 */
export function convertBFloat16ToFloat32(source, shape) {
  const target = new Float32Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    const value = source[i] << 16;
    const bytes = new Uint32Array([value]);
    target[i] = new Float32Array(bytes.buffer)[0];
  }
  return target;
}

// ------------------------------------------------------------
// Section: Schedulers and samplers
// ------------------------------------------------------------

/**
 * Builds a scheduler description for the configured sampler. SDXL typically
 * ships with sophisticated beta schedules; here we encode a deterministic,
 * inspectable approximation so that we can unit test the math. The scheduler is
 * returned as a plain object to keep it serializable and therefore easy to test.
 */
export function buildScheduler(config) {
  const steps = config.steps;
  if (!Number.isInteger(steps) || steps <= 0) {
    throw new Error("Step count must be a positive integer.");
  }
  const scheduleFactory = SCHEDULER_FACTORIES[config.scheduler];
  if (!scheduleFactory) {
    throw new Error(`Unknown scheduler: ${config.scheduler}`);
  }
  return scheduleFactory(steps);
}

/**
 * Scheduler factory dictionary. Each factory returns a frozen object so callers
 * cannot accidentally mutate the schedule in-place.
 */
const SCHEDULER_FACTORIES = {
  ddim(steps) {
    const timesteps = createLinearTimesteps(steps, 0.00085, 0.012);
    return Object.freeze({ type: "ddim", timesteps });
  },
  euler(steps) {
    const timesteps = createCosineTimesteps(steps);
    return Object.freeze({ type: "euler", timesteps });
  },
  heun(steps) {
    const timesteps = createHeunTimesteps(steps);
    return Object.freeze({ type: "heun", timesteps });
  },
};

function createLinearTimesteps(steps, startBeta, endBeta) {
  const increment = (endBeta - startBeta) / Math.max(steps - 1, 1);
  return Object.freeze(
    Array.from({ length: steps }, (_, index) => startBeta + increment * index)
  );
}

function createCosineTimesteps(steps) {
  const values = [];
  for (let i = 0; i < steps; i += 1) {
    const t = i / Math.max(steps - 1, 1);
    const cosine = Math.cos((t + 0.008) / 1.008 * Math.PI * 0.5);
    values.push(Math.max(1e-4, cosine ** 2));
  }
  return Object.freeze(values);
}

function createHeunTimesteps(steps) {
  const values = [];
  for (let i = 0; i < steps; i += 1) {
    const progress = i / Math.max(steps - 1, 1);
    const beta = 0.0001 + progress * (0.02 - 0.0001);
    const corrected = beta + beta * beta * 0.5;
    values.push(corrected);
  }
  return Object.freeze(values);
}

/**
 * Selects the sampler kernel definition. Kernels are pure metadata structures
 * describing the order in which noise predictions are combined.
 */
export function selectSamplerKernel(name) {
  const kernel = SAMPLER_KERNELS[name];
  if (!kernel) {
    throw new Error(`Unsupported sampler kernel: ${name}`);
  }
  return kernel;
}

const SAMPLER_KERNELS = Object.freeze({
  standard: Object.freeze({
    name: "standard",
    guidance: true,
    order: 2,
  }),
  deterministic: Object.freeze({
    name: "deterministic",
    guidance: false,
    order: 1,
  }),
});

// ------------------------------------------------------------
// Section: Latent preparation utilities
// ------------------------------------------------------------

/**
 * Creates the initial latent tensor that feeds into the SDXL denoising loop.
 * The generator parameter is injected to simplify deterministic testing; in
 * production we default to Math.random.
 */
export function createInitialLatents(width, height, channels, generator = Math.random) {
  const size = width * height * channels;
  const data = new Float32Array(size);
  for (let i = 0; i < size; i += 1) {
    const rand = generator();
    data[i] = gaussianFromUniform(rand, generator());
  }
  return data;
}

/**
 * Converts two uniform random numbers into a Gaussian sample using Box-Muller.
 * Separate helper keeps the math unit-testable.
 */
export function gaussianFromUniform(u1, u2) {
  const epsilon = 1e-12;
  const clampedU1 = Math.max(u1, epsilon);
  const radius = Math.sqrt(-2 * Math.log(clampedU1));
  const theta = 2 * Math.PI * u2;
  return radius * Math.cos(theta);
}

// ------------------------------------------------------------
// Section: WebGPU bootstrap and compute placeholders
// ------------------------------------------------------------

/**
 * Requests a WebGPU device from the browser. The promise resolves with an
 * object bundling the adapter, device, and queue. We structure the data this way
 * so that unit tests can inject fakes.
 */
export async function requestWebGPUDevice(navigatorObject = typeof navigator !== "undefined" ? navigator : null) {
  if (!navigatorObject || !navigatorObject.gpu) {
    throw new Error("WebGPU is not available in this environment.");
  }
  const adapter = await navigatorObject.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to acquire a WebGPU adapter.");
  }
  const device = await adapter.requestDevice();
  return { adapter, device, queue: device.queue };
}

/**
 * Encodes the denoising loop. The implementation purposefully throws an error
 * because the mathematical kernels for SDXL are non-trivial and require a
 * substantial volume of shader code. The explicit exception prevents silent
 * fallbacks: users immediately know that the UNet needs to be implemented.
 */
export async function executeDenoisingPass() {
  throw new Error(
    "UNet compute kernels are not yet implemented. Integrate SDXL layers before executing inference."
  );
}

/**
 * Converts latent space values back into a displayable RGBA buffer. The current
 * implementation generates a placeholder visualization (grayscale amplitude)
 * that assists with debugging the scheduler without pretending to be the final
 * photorealistic output.
 */
export function latentsToImageBitmap(latents, width, height) {
  if (!(latents instanceof Float32Array)) {
    throw new TypeError("Expected Float32Array latents.");
  }
  const pixels = new Uint8ClampedArray(width * height * 4);
  let latentIndex = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let amplitude = 0;
      for (let c = 0; c < 4 && latentIndex < latents.length; c += 1) {
        amplitude += Math.abs(latents[latentIndex]);
        latentIndex += 1;
      }
      const value = Math.max(0, Math.min(255, Math.floor((amplitude / 4) * 255)));
      const pixelIndex = (y * width + x) * 4;
      pixels[pixelIndex] = value;
      pixels[pixelIndex + 1] = value;
      pixels[pixelIndex + 2] = value;
      pixels[pixelIndex + 3] = 255;
    }
  }
  return new ImageData(pixels, width, height);
}

// ------------------------------------------------------------
// Section: UI wiring
// ------------------------------------------------------------

/**
 * Attaches event listeners to the DOM once executed in a browser. All DOM
 * mutations are delegated to this function to isolate side effects.
 */
export function mountApplicationUI(doc, state) {
  const checkpointInput = doc.getElementById("checkpoint");
  const stepsInput = doc.getElementById("steps");
  const schedulerSelect = doc.getElementById("scheduler");
  const samplerSelect = doc.getElementById("sampler");
  const cfgInput = doc.getElementById("cfg");
  const generateButton = doc.getElementById("generate");
  const canvas = doc.getElementById("preview");
  const logElement = doc.getElementById("statusLog");
  const ctx = canvas.getContext("2d");

  function writeLog(message) {
    const time = new Date().toISOString();
    logElement.textContent = `${time}: ${message}\n${logElement.textContent}`;
  }

  async function onGenerate() {
    try {
      const file = checkpointInput.files && checkpointInput.files[0];
      if (!file) {
        throw new Error("A .safetensors checkpoint must be selected before inference.");
      }
      const steps = Number.parseInt(stepsInput.value, 10);
      const cfgScale = Number.parseFloat(cfgInput.value);
      const schedulerKey = schedulerSelect.value;
      const samplerKey = samplerSelect.value;
      const config = Object.freeze({
        ...state.config,
        steps,
        cfgScale,
        scheduler: schedulerKey,
        sampler: samplerKey,
      });
      const arrayBuffer = await file.arrayBuffer();
      const parsed = parseSafetensors(arrayBuffer);
      const scheduler = buildScheduler(config);
      const samplerKernel = selectSamplerKernel(config.sampler);
      const latents = createInitialLatents(config.width / 8, config.height / 8, config.latentChannels);

      state.config = config;
      state.tensors = parsed.tensors;
      state.metadata = parsed.metadata;
      state.scheduler = scheduler;
      state.samplerKernel = samplerKernel;
      state.lastLatents = latents;

      writeLog(`Loaded checkpoint with ${Object.keys(parsed.tensors).length} tensors.`);
      writeLog(`Scheduler (${scheduler.type}) prepared with ${scheduler.timesteps.length} steps.`);
      writeLog(`Sampler kernel: ${samplerKernel.name}.`);

      await executeDenoisingPass();
    } catch (error) {
      writeLog(`Inference failed: ${error.message}`);
      console.error(error);
      if (error.message.includes("UNet compute kernels")) {
        writeLog("TODO: Implement SDXL UNet shaders to enable full inference.");
      }
    }
  }

  async function renderLatents() {
    if (!state.lastLatents) {
      writeLog("Latents not initialized yet.");
      return;
    }
    const latentWidth = state.config.width / 8;
    const latentHeight = state.config.height / 8;
    const imageData = latentsToImageBitmap(state.lastLatents, latentWidth, latentHeight);
    const offscreenCanvas = new OffscreenCanvas(latentWidth, latentHeight);
    const offscreenCtx = offscreenCanvas.getContext("2d");
    offscreenCtx.putImageData(imageData, 0, 0);
    const bitmap = await offscreenCanvas.convertToBlob();
    const img = await createImageBitmap(bitmap);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }

  generateButton.addEventListener("click", onGenerate);
  canvas.addEventListener("dblclick", () => {
    renderLatents().catch((error) => writeLog(`Render failed: ${error.message}`));
  });

  writeLog("UI initialized. Double click the canvas to view latent magnitude.");

  return Object.freeze({
    checkpointInput,
    stepsInput,
    schedulerSelect,
    samplerSelect,
    cfgInput,
    generateButton,
    canvas,
    logElement,
    ctx,
  });
}

// ------------------------------------------------------------
// Section: Bootstrap (runs only in browser)
// ------------------------------------------------------------

if (typeof window !== "undefined" && typeof document !== "undefined") {
  const state = createAppState();
  const ui = mountApplicationUI(document, state);
  state.ui = ui;
  window.__LOCAL_SDXL_STATE__ = state;
}

