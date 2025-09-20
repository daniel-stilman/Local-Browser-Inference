/**
 * Browser-first SDXL-inspired latent diffusion engine.
 *
 * The guiding philosophy is a monolithic, functionally-driven module that keeps
 * every behaviour observable, testable, and comprehensively documented. The
 * emphasis on verbose commentary satisfies the "50% rule" directive: we expose
 * context, intent, and implementation choices to maximise downstream
 * comprehension for both humans and future models.
 */

// ============================================================================
// =                              Math Utilities                              =
// ============================================================================

/**
 * Creates a Mulberry32 pseudo random generator. Mulberry32 is tiny, fast and
 * deterministic, which makes it ideal for regression testing as well as seeded
 * diffusion runs inside the browser where we want reproducibility across
 * engines without importing heavyweight RNG libraries.
 * @param {number} seed - Unsigned 32 bit integer used to initialise the state.
 * @returns {() => number} Function that returns a float in [0, 1).
 */
export function createMulberry32(seed) {
  let x = seed >>> 0;
  return function next() {
    x = (x + 0x6D2B79F5) >>> 0;
    let t = Math.imul(x ^ (x >>> 15), 1 | x);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Applies the SiLU activation (also called swish). The diffusion model relies
 * on smooth, non-monotonic activations; SiLU is a common choice in SDXL and we
 * re-implement it here using pure math primitives to avoid dependencies.
 * @param {number} x - Input scalar.
 * @returns {number} Activated scalar.
 */
export function silu(x) {
  const inv = 1 / (1 + Math.exp(-x));
  return x * inv;
}

/**
 * Encodes a scalar timestep into a sinusoidal embedding. Stable diffusion style
 * networks frequently use sinusoidal timesteps because they allow the network
 * to reason about very large or very small diffusion indexes without losing
 * resolution. We mimic the behaviour with a configurable embedding size.
 * @param {number} timestep - Normalised timestep in [0, 1].
 * @param {number} dimension - Size of the resulting embedding vector.
 * @returns {Float32Array} Embedding vector with alternating sin/cos entries.
 */
export function encodeTimestep(timestep, dimension) {
  const result = new Float32Array(dimension);
  const maxPeriod = 10000;
  for (let i = 0; i < dimension / 2; i++) {
    const frequency = Math.pow(maxPeriod, (2 * i) / dimension);
    const angle = timestep * frequency;
    result[2 * i] = Math.sin(angle);
    result[2 * i + 1] = Math.cos(angle);
  }
  return result;
}

/**
 * Normalises a vector in-place and returns it. Normalisation ensures that text
 * conditioning embeddings have unit magnitude, which stabilises classifier
 * free guidance blending later in the pipeline.
 * @param {Float32Array} vector - Vector that will be normalised.
 * @returns {Float32Array} The same vector instance for chaining.
 */
export function normalizeVector(vector) {
  let sumSquares = 0;
  for (let i = 0; i < vector.length; i++) {
    sumSquares += vector[i] * vector[i];
  }
  const norm = Math.sqrt(Math.max(sumSquares, 1e-8));
  for (let i = 0; i < vector.length; i++) {
    vector[i] /= norm;
  }
  return vector;
}

/**
 * Deterministically hashes a prompt into an embedding vector. This serves as a
 * lightweight stand-in for a full text encoder, giving us reproducible
 * conditioning that still responds to user prompts. The hash strategy uses two
 * accumulators with FNV-like mixing to inject each unicode code point into the
 * embedding space.
 * @param {string} prompt - Arbitrary user provided text.
 * @param {number} dimension - Desired output vector length.
 * @returns {Float32Array} Normalised embedding vector.
 */
export function encodePrompt(prompt, dimension) {
  const vector = new Float32Array(dimension);
  let hashA = 0x811C9DC5;
  let hashB = 0x01000193;
  for (let charIndex = 0; charIndex < prompt.length; charIndex++) {
    const code = prompt.charCodeAt(charIndex);
    hashA ^= code;
    hashA = Math.imul(hashA, 0x01000193);
    hashB += code + ((hashB << 1) >>> 0);
    hashB ^= hashB >>> 7;
    const slot = charIndex % dimension;
    const value = ((hashA >>> 1) ^ hashB) & 0xffffffff;
    vector[slot] += ((value / 0xffffffff) * 2 - 1) * 0.5;
  }
  normalizeVector(vector);
  return vector;
}

// ============================================================================
// =                               File Parsing                               =
// ============================================================================

/**
 * Parses a safetensors buffer into a map of typed arrays. The implementation is
 * intentionally direct: we stream through the JSON header, compute offsets and
 * slice the backing ArrayBuffer without allocating intermediate buffers.
 * @param {ArrayBuffer} buffer - Raw file contents.
 * @returns {{ tensors: Map<string, TypedArray>, metadata: any }}
 */
export function parseSafetensors(buffer) {
  if (!(buffer instanceof ArrayBuffer)) {
    throw new Error('Expected ArrayBuffer when parsing safetensors.');
  }
  const view = new DataView(buffer);
  if (view.byteLength < 8) {
    throw new Error('Safetensors file too small to contain header.');
  }
  const headerSize = Number(view.getBigUint64(0, true));
  const headerStart = 8;
  const headerEnd = headerStart + headerSize;
  if (headerEnd > view.byteLength) {
    throw new Error('Safetensors header length exceeds buffer.');
  }
  const headerBytes = new Uint8Array(buffer, headerStart, headerSize);
  const headerString = new TextDecoder('utf-8').decode(headerBytes);
  const header = JSON.parse(headerString);
  const tensors = new Map();
  for (const [name, descriptor] of Object.entries(header)) {
    const { dtype, shape, data_offsets: dataOffsets } = descriptor;
    if (!Array.isArray(shape) || shape.length === 0) {
      throw new Error(`Tensor "${name}" missing valid shape definition.`);
    }
    const [start, end] = dataOffsets;
    if (typeof start !== 'number' || typeof end !== 'number') {
      throw new Error(`Tensor "${name}" offset malformed.`);
    }
    const byteStart = headerEnd + start;
    const byteEnd = headerEnd + end;
    if (byteEnd > buffer.byteLength) {
      throw new Error(`Tensor "${name}" exceeds file length.`);
    }
    const slice = buffer.slice(byteStart, byteEnd);
    const totalElements = shape.reduce((acc, value) => acc * value, 1);
    let typedArray;
    switch (dtype) {
      case 'F32':
        typedArray = new Float32Array(slice);
        break;
      case 'F16':
        typedArray = convertFloat16To32(new Uint16Array(slice));
        break;
      case 'BF16':
        typedArray = convertBFloat16To32(new Uint16Array(slice));
        break;
      case 'I32':
        typedArray = new Int32Array(slice);
        break;
      default:
        throw new Error(`Unsupported dtype "${dtype}" in tensor "${name}".`);
    }
    if (typedArray.length !== totalElements) {
      throw new Error(`Tensor "${name}" element count mismatch.`);
    }
    tensors.set(name, typedArray);
  }
  const metadata = header.__metadata__ ?? null;
  return { tensors, metadata };
}

/**
 * Converts half precision floats into Float32Array. This helper performs manual
 * IEEE 754 decoding so that we can accept F16 weights while still running our
 * compute kernels in 32-bit precision for simplicity.
 * @param {Uint16Array} values - Raw half precision words.
 * @returns {Float32Array} Converted floating point values.
 */
export function convertFloat16To32(values) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const word = values[i];
    const sign = (word & 0x8000) ? -1 : 1;
    let exponent = (word >> 10) & 0x1f;
    let fraction = word & 0x3ff;
    if (exponent === 0) {
      result[i] = sign * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 31) {
      result[i] = fraction ? NaN : sign * Infinity;
    } else {
      result[i] = sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
  }
  return result;
}

/**
 * Converts brain floating point 16 values (bfloat16) to Float32 precision.
 * @param {Uint16Array} values - Raw 16-bit bfloat values.
 * @returns {Float32Array} Converted Float32 representation.
 */
export function convertBFloat16To32(values) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const word = values[i];
    const sign = (word & 0x8000) ? -1 : 1;
    const exponent = (word >> 7) & 0xff;
    const fraction = word & 0x7f;
    if (exponent === 0xff) {
      result[i] = fraction ? NaN : sign * Infinity;
    } else {
      const mantissa = fraction / 128;
      result[i] = sign * Math.pow(2, exponent - 127) * (1 + mantissa);
    }
  }
  return result;
}

// ============================================================================
// =                          Scheduler + Sampler Logic                        =
// ============================================================================

/**
 * Builds the scheduler used to advance the latent tensor through diffusion
 * steps. The schedule mirrors the concepts of DDIM / Euler samplers without
 * importing the full diffusers stack. Each schedule entry exposes the current
 * alpha value, the variance term and a helper sigma used by Euler samplers.
 * @param {number} totalSteps - Number of refinement iterations requested.
 * @param {'ddim' | 'euler'} schedulerType - Name of scheduler curve to use.
 * @returns {{
 *   alphas: Float32Array,
 *   sigmas: Float32Array,
 *   timesteps: Float32Array,
 *   schedulerType: string
 * }}
 */
export function createScheduler(totalSteps, schedulerType) {
  if (!Number.isInteger(totalSteps) || totalSteps <= 0) {
    throw new Error('Total steps must be a positive integer.');
  }
  const steps = totalSteps;
  const epsilon = 1e-4;
  const alphas = new Float32Array(steps);
  const sigmas = new Float32Array(steps + 1);
  const timesteps = new Float32Array(steps);
  for (let i = 0; i < steps; i++) {
    const progress = (i + 1) / (steps + 1);
    const eased = progress * progress;
    const alpha = Math.max(eased, epsilon);
    alphas[i] = alpha;
    timesteps[i] = 1 - progress;
  }
  sigmas[steps] = 0;
  for (let i = 0; i < steps; i++) {
    const alpha = alphas[i];
    const sigma = Math.sqrt(Math.max((1 - alpha) / alpha, 0));
    sigmas[i] = sigma;
  }
  if (schedulerType === 'euler') {
    for (let i = 0; i < sigmas.length - 1; i++) {
      if (sigmas[i] < sigmas[i + 1]) {
        sigmas[i + 1] = Math.max(sigmas[i], sigmas[i + 1]);
      }
    }
  }
  return { alphas, sigmas, timesteps, schedulerType };
}

/**
 * Executes a single scheduler update. The DDIM branch mirrors the closed form
 * update equations, whereas the Euler branch follows the k-diffusion inspired
 * update expressed in sigma parameter space.
 * @param {Float32Array} latents - Current latent tensor.
 * @param {Float32Array} predictedNoise - Predicted noise tensor.
 * @param {number} stepIndex - Index of the diffusion step.
 * @param {{alphas: Float32Array, sigmas: Float32Array}} scheduler - Scheduler parameters.
 * @param {'ddim'|'euler'} samplerType - Chosen sampler variant.
 * @param {() => number} rng - Random generator for stochasticity.
 * @returns {Float32Array} Updated latent tensor.
 */
export function schedulerStep(latents, predictedNoise, stepIndex, scheduler, samplerType, rng) {
  const { alphas, sigmas } = scheduler;
  const currentAlpha = alphas[stepIndex];
  const nextAlpha = stepIndex + 1 < alphas.length ? alphas[stepIndex + 1] : 1;
  const sqrtCurrentAlpha = Math.sqrt(currentAlpha);
  const sqrtOneMinusCurrentAlpha = Math.sqrt(Math.max(1 - currentAlpha, 0));
  const latentsCopy = new Float32Array(latents.length);
  if (samplerType === 'euler') {
    const sigmaCurr = sigmas[stepIndex];
    const sigmaNext = sigmas[stepIndex + 1] ?? 0;
    const deltaSigma = sigmaNext - sigmaCurr;
    for (let i = 0; i < latents.length; i++) {
      const derivative = predictedNoise[i];
      latentsCopy[i] = latents[i] + derivative * deltaSigma;
    }
    if (sigmaNext > 0) {
      for (let i = 0; i < latentsCopy.length; i++) {
        latentsCopy[i] += rng() * Math.sqrt(Math.max(sigmaNext * sigmaNext - sigmaCurr * sigmaCurr, 0));
      }
    }
    return latentsCopy;
  }
  const sqrtNextAlpha = Math.sqrt(nextAlpha);
  const sqrtOneMinusNextAlpha = Math.sqrt(Math.max(1 - nextAlpha, 0));
  for (let i = 0; i < latents.length; i++) {
    const currentLatent = latents[i];
    const noise = predictedNoise[i];
    const predictedOriginal = (currentLatent - sqrtOneMinusCurrentAlpha * noise) / sqrtCurrentAlpha;
    let nextLatent = sqrtNextAlpha * predictedOriginal + sqrtOneMinusNextAlpha * noise;
    const sigmaCurr = sigmas[stepIndex];
    const sigmaNext = sigmas[stepIndex + 1] ?? 0;
    if (sigmaNext > 0) {
      nextLatent += rng() * sigmaNext;
    }
    latentsCopy[i] = nextLatent;
  }
  return latentsCopy;
}

/**
 * Blends the conditional and unconditional noise predictions using classifier
 * free guidance.
 * @param {Float32Array} unconditional - Unconditional noise prediction.
 * @param {Float32Array} conditional - Conditional noise prediction.
 * @param {number} cfgScale - Guidance scale.
 * @returns {Float32Array} Blended noise tensor.
 */
export function combineGuidance(unconditional, conditional, cfgScale) {
  const result = new Float32Array(unconditional.length);
  for (let i = 0; i < unconditional.length; i++) {
    const delta = conditional[i] - unconditional[i];
    result[i] = unconditional[i] + cfgScale * delta;
  }
  return result;
}

// ============================================================================
// =                          Toy UNet Style Backbone                          =
// ============================================================================

/**
 * Builds a compact latent network backed by the safetensors checkpoint. The
 * weights adhere to a simplified architecture that still captures the core
 * mechanics of SDXL: latent processing, timestep conditioning, prompt
 * conditioning and decoding into RGB pixels.
 * @param {Map<string, TypedArray>} tensors - Parsed tensor dictionary.
 * @returns {{
 *   predict: (latents: Float32Array, timestep: number, conditioning: Float32Array) => Float32Array,
 *   decode: (latents: Float32Array) => Float32Array,
 *   latentShape: { channels: number, height: number, width: number },
 *   imageShape: { height: number, width: number },
 *   conditioningSize: number,
 *   timestepSize: number
 * }}
 */
export function buildToyModel(tensors) {
  const latentWeight = requireTensor(tensors, 'model.latent.weight', Float32Array);
  const latentBias = requireTensor(tensors, 'model.latent.bias', Float32Array);
  const timeWeight = requireTensor(tensors, 'model.time.weight', Float32Array);
  const timeBias = requireTensor(tensors, 'model.time.bias', Float32Array);
  const condWeight = requireTensor(tensors, 'model.cond.weight', Float32Array);
  const condBias = requireTensor(tensors, 'model.cond.bias', Float32Array);
  const outputWeight = requireTensor(tensors, 'model.output.weight', Float32Array);
  const outputBias = requireTensor(tensors, 'model.output.bias', Float32Array);
  const decoderWeight = requireTensor(tensors, 'model.decoder.weight', Float32Array);
  const decoderBias = requireTensor(tensors, 'model.decoder.bias', Float32Array);
  const latentShapeTensor = requireTensor(tensors, 'model.latent_shape', Int32Array);
  const imageShapeTensor = requireTensor(tensors, 'model.image_shape', Int32Array);
  const latentChannels = latentShapeTensor[0];
  const latentHeight = latentShapeTensor[1];
  const latentWidth = latentShapeTensor[2];
  const latentSize = latentChannels * latentHeight * latentWidth;
  const hiddenSize = latentBias.length;
  const timeSize = timeWeight.length / hiddenSize;
  const condSize = condWeight.length / hiddenSize;
  const decoderSize = decoderBias.length;
  if (!Number.isInteger(timeSize)) {
    throw new Error('Time embedding weight dimensions invalid.');
  }
  if (!Number.isInteger(condSize)) {
    throw new Error('Conditioning weight dimensions invalid.');
  }
  if (timeBias.length !== hiddenSize) {
    throw new Error('Time bias length mismatch.');
  }
  if (condBias.length !== hiddenSize) {
    throw new Error('Conditioning bias length mismatch.');
  }
  if (latentWeight.length !== hiddenSize * latentSize) {
    throw new Error('Latent weight shape mismatch.');
  }
  if (timeWeight.length !== hiddenSize * timeSize) {
    throw new Error('Time weight shape mismatch.');
  }
  if (condWeight.length !== hiddenSize * condSize) {
    throw new Error('Conditioning weight shape mismatch.');
  }
  if (outputWeight.length !== latentSize * hiddenSize) {
    throw new Error('Output weight shape mismatch.');
  }
  if (outputBias.length !== latentSize) {
    throw new Error('Output bias length mismatch.');
  }
  if (decoderWeight.length !== decoderSize * latentSize) {
    throw new Error('Decoder weight shape mismatch.');
  }
  const imageHeight = imageShapeTensor[0];
  const imageWidth = imageShapeTensor[1];
  if (decoderSize !== imageHeight * imageWidth * 3) {
    throw new Error('Decoder shape mismatch with image dimensions.');
  }
  const predict = (latents, timestep, conditioning) => {
    if (latents.length !== latentSize) {
      throw new Error('Latent size mismatch in predict call.');
    }
    if (conditioning.length !== condSize) {
      throw new Error('Conditioning vector length mismatch.');
    }
    const timeEmbedding = encodeTimestep(timestep, timeSize);
    const hidden = new Float32Array(hiddenSize);
    for (let h = 0; h < hiddenSize; h++) {
      let sum = latentBias[h];
      const latentOffset = h * latentSize;
      for (let i = 0; i < latentSize; i++) {
        sum += latentWeight[latentOffset + i] * latents[i];
      }
      const timeOffset = h * timeSize;
      for (let j = 0; j < timeSize; j++) {
        sum += timeWeight[timeOffset + j] * timeEmbedding[j];
      }
      const condOffset = h * condSize;
      for (let k = 0; k < condSize; k++) {
        sum += condWeight[condOffset + k] * conditioning[k];
      }
      sum += timeBias[h] + condBias[h];
      hidden[h] = silu(sum);
    }
    const output = new Float32Array(latentSize);
    for (let i = 0; i < latentSize; i++) {
      let sum = outputBias[i];
      const outputOffset = i * hiddenSize;
      for (let h = 0; h < hiddenSize; h++) {
        sum += outputWeight[outputOffset + h] * hidden[h];
      }
      output[i] = sum;
    }
    return output;
  };

  const decode = (latents) => {
    if (latents.length !== latentSize) {
      throw new Error('Latent size mismatch when decoding image.');
    }
    const pixels = new Float32Array(decoderSize);
    for (let p = 0; p < decoderSize; p++) {
      let sum = decoderBias[p];
      const decoderOffset = p * latentSize;
      for (let i = 0; i < latentSize; i++) {
        sum += decoderWeight[decoderOffset + i] * latents[i];
      }
      pixels[p] = Math.tanh(sum);
    }
    return pixels;
  };

  return {
    predict,
    decode,
    latentShape: { channels: latentChannels, height: latentHeight, width: latentWidth },
    imageShape: { height: imageHeight, width: imageWidth },
    conditioningSize: condSize,
    timestepSize: timeSize,
  };
}

/**
 * Utility that retrieves a tensor from the map and validates its constructor.
 * @param {Map<string, TypedArray>} tensors - Tensor dictionary.
 * @param {string} name - Tensor key to locate.
 * @param {Function} ctor - Expected typed array constructor.
 * @returns {TypedArray}
 */
export function requireTensor(tensors, name, ctor) {
  const value = tensors.get(name);
  if (!value) {
    throw new Error(`Missing tensor "${name}" in checkpoint.`);
  }
  if (!(value instanceof ctor)) {
    throw new Error(`Tensor "${name}" expected ${ctor.name}.`);
  }
  return value;
}

// ============================================================================
// =                            Diffusion Orchestration                        =
// ============================================================================

/**
 * Initialises the latent tensor with Gaussian noise.
 * @param {number} size - Number of elements.
 * @param {() => number} rng - Deterministic RNG.
 * @returns {Float32Array}
 */
export function createInitialLatents(size, rng) {
  const latents = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const u1 = rng();
    const u2 = rng();
    const radius = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-12)));
    const theta = 2 * Math.PI * u2;
    latents[i] = radius * Math.cos(theta);
  }
  return latents;
}

/**
 * Converts decoded pixel values in [-1, 1] into an RGBA Uint8 buffer that can
 * be written directly into a canvas ImageData instance.
 * @param {Float32Array} pixels - Decoder output.
 * @param {{height: number, width: number}} shape - Image height and width.
 * @returns {Uint8ClampedArray}
 */
export function pixelsToImageData(pixels, shape) {
  const { height, width } = shape;
  const totalPixels = height * width;
  const rgba = new Uint8ClampedArray(totalPixels * 4);
  for (let i = 0; i < totalPixels; i++) {
    const r = Math.max(-1, Math.min(1, pixels[i * 3] ?? 0));
    const g = Math.max(-1, Math.min(1, pixels[i * 3 + 1] ?? 0));
    const b = Math.max(-1, Math.min(1, pixels[i * 3 + 2] ?? 0));
    rgba[i * 4] = ((r + 1) / 2) * 255;
    rgba[i * 4 + 1] = ((g + 1) / 2) * 255;
    rgba[i * 4 + 2] = ((b + 1) / 2) * 255;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/**
 * Executes the complete diffusion sampling loop.
 * @param {{
 *   model: ReturnType<typeof buildToyModel>,
 *   prompt: string,
 *   negativePrompt: string,
 *   steps: number,
 *   schedulerType: 'ddim' | 'euler',
 *   samplerType: 'ddim' | 'euler',
 *   cfgScale: number,
 *   seed: number
 * }} options - High level inference configuration.
 * @returns {{ latents: Float32Array, image: Uint8ClampedArray }}
 */
export function sampleDiffusion(options) {
  const { model, prompt, negativePrompt, steps, schedulerType, samplerType, cfgScale, seed } = options;
  if (!model) {
    throw new Error('Model must be provided before sampling.');
  }
  const scheduler = createScheduler(steps, schedulerType);
  const rng = createMulberry32(seed >>> 0);
  const latents = createInitialLatents(model.latentShape.channels * model.latentShape.height * model.latentShape.width, rng);
  const cond = encodePrompt(prompt ?? '', model.conditioningSize);
  const uncond = encodePrompt(negativePrompt ?? '', model.conditioningSize);
  let currentLatents = latents;
  for (let stepIndex = 0; stepIndex < steps; stepIndex++) {
    const timestep = scheduler.timesteps[stepIndex];
    const unconditionalNoise = model.predict(currentLatents, timestep, uncond);
    const conditionalNoise = model.predict(currentLatents, timestep, cond);
    const guided = combineGuidance(unconditionalNoise, conditionalNoise, cfgScale);
    currentLatents = schedulerStep(currentLatents, guided, stepIndex, scheduler, samplerType, rng);
  }
  const decoded = model.decode(currentLatents);
  const image = pixelsToImageData(decoded, model.imageShape);
  return { latents: currentLatents, image };
}

// ============================================================================
// =                               Canvas Rendering                            =
// ============================================================================

/**
 * Draws image data onto a canvas element. The function is isolated so that
 * integration tests can stub it out while browser usage still benefits from a
 * straightforward imperative painting routine.
 * @param {HTMLCanvasElement} canvas - Destination canvas.
 * @param {Uint8ClampedArray} image - RGBA pixel buffer.
 * @param {{height: number, width: number}} shape - Image shape used by decoder.
 */
export function renderImageToCanvas(canvas, image, shape) {
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas context unavailable.');
  }
  const { height, width } = shape;
  const imageData = new ImageData(width, height);
  imageData.data.set(image);
  context.putImageData(imageData, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(canvas, 0, 0, width, height, 0, 0, canvas.width, canvas.height);
}

// ============================================================================
// =                                UI Controller                              =
// ============================================================================

/**
 * Wires the DOM to the inference engine. The controller exposes dependency
 * injection hooks which facilitate deterministic testing without compromising
 * the production configuration.
 * @param {Document} doc - Document instance from the browser or a testing DOM.
 * @param {{
 *   loadModelFromFile?: (file: File) => Promise<ReturnType<typeof buildToyModel>>,
 *   runPipeline?: (options: any) => Promise<{ image: Uint8ClampedArray }>,
 *   renderImage?: (canvas: HTMLCanvasElement, image: Uint8ClampedArray, shape: {height:number,width:number}) => void,
 *   reportStatus?: (message: string) => void
 * }} overrides - Optional dependency overrides for tests.
 * @returns {{ getState: () => { modelLoaded: boolean } }}
 */
export function initializeApp(doc, overrides = {}) {
  const form = doc.getElementById('inference-form');
  const checkpointInput = doc.getElementById('checkpoint-input');
  const promptInput = doc.getElementById('prompt-input');
  const negativePromptInput = doc.getElementById('negative-prompt-input');
  const stepsInput = doc.getElementById('steps-input');
  const cfgInput = doc.getElementById('cfg-input');
  const schedulerSelect = doc.getElementById('scheduler-select');
  const samplerSelect = doc.getElementById('sampler-select');
  const seedInput = doc.getElementById('seed-input');
  const generateButton = doc.getElementById('generate-button');
  const statusElement = doc.getElementById('status');
  const canvas = doc.getElementById('preview-canvas');

  if (!(form instanceof doc.defaultView.HTMLFormElement)) {
    throw new Error('Inference form not found.');
  }
  const state = {
    model: null,
  };

  const loadModelFromFile = overrides.loadModelFromFile ?? (async (file) => {
    const buffer = await file.arrayBuffer();
    const { tensors } = parseSafetensors(buffer);
    return buildToyModel(tensors);
  });

  const runPipeline = overrides.runPipeline ?? (async (config) => sampleDiffusion(config));
  const renderImage = overrides.renderImage ?? renderImageToCanvas;
  const reportStatus = overrides.reportStatus ?? ((message) => {
    if (statusElement) {
      statusElement.textContent = message;
    }
  });

  if (checkpointInput) {
    checkpointInput.addEventListener('change', async () => {
      const file = checkpointInput.files?.[0];
      if (!file) {
        reportStatus('No checkpoint selected.');
        return;
      }
      reportStatus('Loading checkpoint…');
      try {
        const model = await loadModelFromFile(file);
        state.model = model;
        reportStatus('Checkpoint loaded. Ready to sample.');
      } catch (error) {
        state.model = null;
        reportStatus(`Failed to load checkpoint: ${error instanceof Error ? error.message : String(error)}`);
        throw error;
      }
    });
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.model) {
      reportStatus('Load a checkpoint before sampling.');
      return;
    }
    generateButton.disabled = true;
    reportStatus('Sampling…');
    const cfgScale = parseFloat(cfgInput.value);
    const steps = parseInt(stepsInput.value, 10);
    const seed = parseInt(seedInput.value, 10) >>> 0;
    try {
      const result = await runPipeline({
        model: state.model,
        prompt: promptInput.value,
        negativePrompt: negativePromptInput.value,
        steps,
        schedulerType: schedulerSelect.value,
        samplerType: samplerSelect.value,
        cfgScale,
        seed,
      });
      renderImage(canvas, result.image, state.model.imageShape);
      reportStatus('Generation complete.');
    } catch (error) {
      reportStatus(`Sampling failed: ${error instanceof Error ? error.message : String(error)}`);
      throw error;
    } finally {
      generateButton.disabled = false;
    }
  });

  return {
    getState: () => ({ modelLoaded: Boolean(state.model) }),
  };
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  initializeApp(document);
}

