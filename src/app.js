import { StableDiffusionXLPipeline, setCacheImpl, ProgressStatus } from '@aislamov/diffusers.js'
import { Tensor } from '@xenova/transformers'
import { unzipSync } from 'fflate'
import seedrandom from 'seedrandom'

const TEXT_DECODER = new TextDecoder('utf-8')

const REQUIRED_MODEL_FILES = [
  'model_index.json',
  'unet/model.onnx',
  'unet/config.json',
  'text_encoder/model.onnx',
  'text_encoder/config.json',
  'text_encoder_2/model.onnx',
  'text_encoder_2/config.json',
  'vae_decoder/model.onnx',
  'vae_decoder/config.json',
  'tokenizer/tokenizer.json',
  'tokenizer_2/tokenizer.json',
  'scheduler/scheduler_config.json',
]

export const SAMPLER_PROFILES = Object.freeze({
  deterministic: { id: 'deterministic', label: 'Deterministic (η=0)', eta: 0 },
  ancestral: { id: 'ancestral', label: 'Ancestral (η=1)', eta: 1 },
  'ancestral-half': { id: 'ancestral-half', label: 'Ancestral (η=0.5)', eta: 0.5 },
})

export const SCHEDULER_CHOICES = Object.freeze({
  pndm: { id: 'pndm', label: 'PNDM (accurate)' },
  ddim: { id: 'ddim', label: 'DDIM (flexible)' },
})

function cloneConfig(config) {
  if (typeof structuredClone === 'function') {
    return structuredClone(config)
  }
  return JSON.parse(JSON.stringify(config))
}

function normalizePath(path) {
  return path.replace(/\\/g, '/').replace(/^\.\/?/, '').replace(/\/+/g, '/').trim()
}

function ensureArrayBuffer(view) {
  if (view instanceof ArrayBuffer) {
    return view
  }
  if (view instanceof Uint8Array) {
    return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)
  }
  if (ArrayBuffer.isView(view)) {
    const typed = new Uint8Array(view.buffer, view.byteOffset, view.byteLength)
    return typed.slice().buffer
  }
  throw new Error('Unsupported binary payload type')
}

/**
 * VirtualModelCache maintains an in-memory view of every artifact that composes a converted SDXL model.
 * The diffusers.js runtime asks for files through the hub interface; by implementing the CacheImpl contract we can
 * intercept those requests and satisfy them with the user's uploaded files instead of performing network fetches.
 */
export class VirtualModelCache {
  constructor() {
    this.models = new Map()
  }

  registerModel(modelId, fileMap) {
    this.models.set(modelId, fileMap)
  }

  unregisterModel(modelId) {
    this.models.delete(modelId)
  }

  hasModel(modelId) {
    return this.models.has(modelId)
  }

  async getModelFile(modelRepoOrPath, fileName, fatal = true, options = {}) {
    const store = this.models.get(modelRepoOrPath)
    if (!store) {
      if (fatal) {
        throw new Error(`Model "${modelRepoOrPath}" has not been loaded into the browser cache.`)
      }
      return null
    }
    const normalized = normalizePath(fileName)
    const payload = store.get(normalized)
    if (!payload) {
      if (fatal) {
        throw new Error(`File "${normalized}" is missing from the provided model package.`)
      }
      return null
    }
    if (options.returnText) {
      return TEXT_DECODER.decode(payload)
    }
    return ensureArrayBuffer(payload)
  }
}

function readFileAsUint8Array(file) {
  return file.arrayBuffer().then(buffer => new Uint8Array(buffer))
}

export function extractArchiveToMap(bytes) {
  const isZip = bytes[0] === 0x50 && bytes[1] === 0x4b && (bytes[2] === 0x03 || bytes[2] === 0x05)
  if (!isZip) {
    throw new Error('The selected file is not a supported archive. Provide a .zip package generated from your checkpoint.')
  }
  const entries = unzipSync(bytes, { consume: true })
  if (!entries || Object.keys(entries).length === 0) {
    throw new Error('The archive is empty or could not be decompressed.')
  }
  let rootPrefix = ''
  for (const entry of Object.keys(entries)) {
    const normalized = normalizePath(entry)
    if (normalized.endsWith('model_index.json')) {
      const parts = normalized.split('/')
      parts.pop()
      rootPrefix = parts.join('/')
      break
    }
  }
  const files = new Map()
  for (const [rawPath, data] of Object.entries(entries)) {
    if (!(data instanceof Uint8Array)) continue
    let normalized = normalizePath(rawPath)
    if (rootPrefix && normalized.startsWith(rootPrefix + '/')) {
      normalized = normalized.slice(rootPrefix.length + 1)
    }
    if (!normalized || normalized.endsWith('/')) {
      continue
    }
    files.set(normalized, data)
  }
  return files
}

export async function fileListToMap(fileList) {
  const files = new Map()
  for (const file of Array.from(fileList)) {
    const relativePath = normalizePath(file.webkitRelativePath || file.name)
    if (!relativePath) continue
    const data = await readFileAsUint8Array(file)
    files.set(relativePath, data)
  }
  return files
}

function validateModelMap(fileMap) {
  const missing = REQUIRED_MODEL_FILES.filter(required => !fileMap.has(required))
  if (missing.length) {
    throw new Error(`The provided package is missing required files: ${missing.join(', ')}`)
  }
}

function describeBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  return `${size.toFixed(unitIndex === 0 ? 0 : 2)} ${units[unitIndex]}`
}

/**
 * ModelManager orchestrates the process of ingesting a user-supplied archive or folder. It validates the contents,
 * streams progress updates for UI feedback, and registers the resulting files with the VirtualModelCache.
 */
export class ModelManager {
  constructor(cache, logger = () => {}) {
    this.cache = cache
    this.logger = logger
    this.currentModelId = null
  }

  async loadModelFromInput(fileList, progressCallback = () => {}) {
    if (!fileList || fileList.length === 0) {
      throw new Error('Select a model archive or directory before attempting to load it.')
    }

    let fileMap
    let totalBytes = 0
    for (const file of Array.from(fileList)) {
      totalBytes += file.size ?? 0
    }

    if (fileList.length === 1 && !fileList[0].webkitRelativePath) {
      const bytes = await readFileAsUint8Array(fileList[0])
      progressCallback({ phase: 'decompress', loaded: bytes.byteLength, total: bytes.byteLength })
      fileMap = extractArchiveToMap(bytes)
    } else {
      const entries = Array.from(fileList)
      const aggregated = new Map()
      let processedBytes = 0
      for (const file of entries) {
        const data = await readFileAsUint8Array(file)
        processedBytes += data.byteLength
        progressCallback({ phase: 'read', loaded: processedBytes, total: totalBytes })
        const relativePath = normalizePath(file.webkitRelativePath || file.name)
        aggregated.set(relativePath, data)
      }
      fileMap = aggregated
    }

    validateModelMap(fileMap)

    const modelId = `local-model-${Date.now()}`
    this.cache.registerModel(modelId, fileMap)
    this.currentModelId = modelId

    const totalSize = Array.from(fileMap.values()).reduce((sum, value) => sum + value.byteLength, 0)
    this.logger(`Loaded ${fileMap.size} files (${describeBytes(totalSize)}) into browser memory as ${modelId}.`)
    return { modelId, fileCount: fileMap.size, totalBytes: totalSize }
  }

  get activeModelId() {
    return this.currentModelId
  }
}

function cloneTensor(tensor) {
  return new Tensor(tensor.type, tensor.data.slice(), tensor.dims.slice())
}

/**
 * Draws a single normally distributed sample using the Box-Muller transform.
 */
function randomNormal(mean, std, rng) {
  let u = 0
  let v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  const mag = Math.sqrt(-2.0 * Math.log(u))
  const z0 = mag * Math.cos(2.0 * Math.PI * v)
  return z0 * std + mean
}

/**
 * Creates a tensor filled with Gaussian noise. A deterministic RNG is used whenever a seed is provided so that
 * repeated generations yield identical trajectories.
 */
function randomNormalTensor(shape, mean = 0, std = 1, type = 'float32', seed = '') {
  const total = shape.reduce((acc, value) => acc * value, 1)
  const data = new Float32Array(total)
  const rng = seed ? seedrandom(seed) : seedrandom()
  for (let i = 0; i < total; i++) {
    data[i] = randomNormal(mean, std, rng)
  }
  return new Tensor(type, data, shape)
}

function computeTimestepsLinear(numTrainSteps, numInferenceSteps) {
  if (numInferenceSteps <= 1) {
    return [numTrainSteps - 1]
  }
  const stepRatio = (numTrainSteps - 1) / (numInferenceSteps - 1)
  const timesteps = []
  for (let i = 0; i < numInferenceSteps; i++) {
    const step = Math.round((numTrainSteps - 1) - i * stepRatio)
    timesteps.push(Math.max(step, 0))
  }
  return timesteps
}

/**
 * BaseScheduler provides diffusion scheduler utilities shared by the custom implementations in this file. It mirrors the
 * responsibilities of diffusers.js' SchedulerBase without importing the package internals (which are not part of the public API).
 */
class BaseScheduler {
  constructor(config) {
    this.config = { ...config }
    this.numTrainTimesteps = this.config.num_train_timesteps
    this.betas = this.computeBetas()
    this.alphas = new Float32Array(this.betas.length)
    this.alphasCumprod = new Float32Array(this.betas.length)
    let cumulative = 1
    for (let i = 0; i < this.betas.length; i++) {
      const alpha = 1 - this.betas[i]
      this.alphas[i] = alpha
      cumulative *= alpha
      this.alphasCumprod[i] = cumulative
    }
    this.finalAlphaCumprod = this.config.set_alpha_to_one ? 1 : this.alphasCumprod[0]
    this.numInferenceSteps = 0
    this.timesteps = new Tensor('int32', new Int32Array(), [0])
    this.initNoiseSigma = 1
  }

  computeBetas() {
    const num = this.config.num_train_timesteps
    const betas = new Float32Array(num)
    if (Array.isArray(this.config.trained_betas) && this.config.trained_betas.length === num) {
      return Float32Array.from(this.config.trained_betas)
    }
    const schedule = this.config.beta_schedule || 'linear'
    if (schedule === 'linear') {
      const start = this.config.beta_start
      const end = this.config.beta_end
      if (num === 1) {
        betas[0] = start
        return betas
      }
      const step = (end - start) / (num - 1)
      for (let i = 0; i < num; i++) {
        betas[i] = start + step * i
      }
      return betas
    }
    if (schedule === 'scaled_linear') {
      const start = Math.sqrt(this.config.beta_start)
      const end = Math.sqrt(this.config.beta_end)
      const step = num === 1 ? 0 : (end - start) / (num - 1)
      for (let i = 0; i < num; i++) {
        const value = start + step * i
        betas[i] = value * value
      }
      return betas
    }
    if (schedule === 'squaredcos_cap_v2') {
      return betasForAlphaBar(num, 0.999, 'cosine')
    }
    throw new Error(`Unsupported beta schedule: ${schedule}`)
  }
}

function betasForAlphaBar(numDiffusionTimesteps, maxBeta = 0.999, alphaTransformType = 'cosine') {
  const betas = new Float32Array(numDiffusionTimesteps)
  const alphaBar = timeStep => {
    if (alphaTransformType === 'cosine') {
      return Math.cos((timeStep + 0.008) / 1.008 * Math.PI / 2) ** 2
    }
    if (alphaTransformType === 'exp') {
      return Math.exp(timeStep * -12)
    }
    throw new Error(`Unsupported alpha transform type: ${alphaTransformType}`)
  }
  for (let i = 0; i < numDiffusionTimesteps; i++) {
    const t1 = i / numDiffusionTimesteps
    const t2 = (i + 1) / numDiffusionTimesteps
    betas[i] = Math.min(1 - alphaBar(t2) / alphaBar(t1), maxBeta)
  }
  return betas
}

/**
 * CustomDDIMScheduler reproduces the behaviour of the DDIM scheduler with optional stochasticity (via eta). The implementation
 * is written from scratch so it can operate in the browser without depending on non-exported internals from diffusers.js.
 */
class CustomDDIMScheduler extends BaseScheduler {
  constructor(config, { eta = 0, seed = '' } = {}) {
    super({ ...config, prediction_type: config.prediction_type || 'epsilon' })
    this.eta = eta
    this.seed = seed || ''
    this.noiseCounter = 0
  }

  updateSampling({ eta, seed }) {
    if (typeof eta === 'number') {
      this.eta = eta
    }
    if (typeof seed === 'string') {
      this.seed = seed
    }
    this.noiseCounter = 0
  }

  setTimesteps(numInferenceSteps) {
    this.numInferenceSteps = numInferenceSteps
    const steps = computeTimestepsLinear(this.config.num_train_timesteps, numInferenceSteps)
    this.timesteps = new Tensor('int32', Int32Array.from(steps), [steps.length])
    this.noiseCounter = 0
  }

  step(modelOutput, timestep, sample) {
    if (typeof timestep !== 'number') {
      timestep = Number(timestep)
    }
    const prevTimestep = timestep - Math.floor(this.config.num_train_timesteps / this.numInferenceSteps)
    const alphaProdT = this.alphasCumprod[timestep]
    const alphaProdTPrev = prevTimestep >= 0 ? this.alphasCumprod[prevTimestep] : this.finalAlphaCumprod
    const sqrtAlphaProdT = Math.sqrt(alphaProdT)
    const sqrtOneMinusAlphaProdT = Math.sqrt(Math.max(1 - alphaProdT, 1e-12))
    let predOriginalSample
    switch (this.config.prediction_type) {
      case 'epsilon':
        predOriginalSample = sample.sub(modelOutput.mul(sqrtOneMinusAlphaProdT))
        break
      case 'sample':
        predOriginalSample = cloneTensor(modelOutput)
        break
      case 'v_prediction':
        predOriginalSample = sample.mul(sqrtAlphaProdT).sub(modelOutput.mul(sqrtOneMinusAlphaProdT))
        break
      default:
        throw new Error(`Unsupported prediction type: ${this.config.prediction_type}`)
    }

    const variance = (1 - alphaProdTPrev) / (1 - alphaProdT) * (1 - alphaProdT / alphaProdTPrev)
    const varianceClamped = Math.max(variance, 0)
    const stdDevT = this.eta * Math.sqrt(varianceClamped)
    const sqrtAlphaProdTPrev = Math.sqrt(alphaProdTPrev)
    const coeff = Math.sqrt(Math.max(0, 1 - alphaProdTPrev - stdDevT * stdDevT))
    let prevSample = predOriginalSample.mul(sqrtAlphaProdTPrev).add(modelOutput.mul(coeff))

    if (this.eta > 0 && stdDevT > 0) {
      const noiseSeed = `${this.seed}:${this.noiseCounter++}`
      const noise = randomNormalTensor(sample.dims, 0, 1, sample.type, noiseSeed)
      prevSample = prevSample.add(noise.mul(stdDevT))
    }

    return prevSample
  }
}

/**
 * PNDMSchedulerCompat reimplements the PLMS solver used by the default Stable Diffusion pipelines. The logic follows the
 * published pseudo numerical method but avoids depending on the unexported SchedulerBase class provided by diffusers.js.
 */
class PNDMSchedulerCompat extends BaseScheduler {
  constructor(config) {
    super({ ...config, skip_prk_steps: typeof config.skip_prk_steps === 'boolean' ? config.skip_prk_steps : true })
    this.pndmOrder = 4
    this.skipPrk = this.config.skip_prk_steps !== false
    this.counter = 0
    this.ets = []
    this.curSample = null
    this.timestepsArray = []
  }

  setTimesteps(numInferenceSteps) {
    this.numInferenceSteps = numInferenceSteps
    const stepRatio = Math.floor(this.config.num_train_timesteps / numInferenceSteps)
    const base = []
    for (let i = 0; i < numInferenceSteps; i++) {
      const step = Math.round(i * stepRatio) + this.config.steps_offset
      base.push(step)
    }
    if (!this.skipPrk) {
      throw new Error('PRK steps are not supported in this browser scheduler implementation.')
    }
    const plms = []
    for (let i = 0; i < base.length - 1; i++) {
      plms.push(base[i])
    }
    if (base.length >= 2) {
      plms.push(base[base.length - 2])
    }
    plms.push(base[base.length - 1])
    plms.reverse()
    this.timestepsArray = plms
    this.timesteps = new Tensor('int32', Int32Array.from(this.timestepsArray), [this.timestepsArray.length])
    this.ets = []
    this.counter = 0
    this.curSample = null
  }

  _getPrevSample(sample, timestep, prevTimestep, modelOutput) {
    const alphaProdT = this.alphasCumprod[timestep]
    const alphaProdTPrev = prevTimestep >= 0 ? this.alphasCumprod[prevTimestep] : this.finalAlphaCumprod
    const betaProdT = 1 - alphaProdT
    const betaProdTPrev = 1 - alphaProdTPrev
    if (this.config.prediction_type === 'v_prediction') {
      modelOutput = modelOutput.mul(Math.sqrt(alphaProdT)).add(sample.mul(Math.sqrt(betaProdT)))
    } else if (this.config.prediction_type !== 'epsilon') {
      throw new Error(`prediction_type ${this.config.prediction_type} must be 'epsilon' or 'v_prediction'`)
    }
    const sampleCoeff = Math.sqrt(alphaProdTPrev / alphaProdT)
    const modelOutputDenomCoeff = alphaProdT * Math.sqrt(betaProdTPrev) + Math.sqrt(alphaProdT * betaProdT * alphaProdTPrev)
    const prevSample = sample.mul(sampleCoeff).sub(modelOutput.mul((alphaProdTPrev - alphaProdT) / modelOutputDenomCoeff))
    return prevSample
  }

  step(modelOutput, timestep, sample) {
    let prevTimestep = timestep - Math.floor(this.config.num_train_timesteps / this.numInferenceSteps)
    if (this.counter !== 1) {
      if (this.ets.length >= 3) {
        this.ets = this.ets.slice(-3)
      }
      this.ets.push(modelOutput)
    } else {
      prevTimestep = timestep
      timestep = timestep + Math.floor(this.config.num_train_timesteps / this.numInferenceSteps)
    }

    let adjustedOutput = modelOutput
    if (this.ets.length === 1 && this.counter === 0) {
      this.curSample = sample
    } else if (this.ets.length === 1 && this.counter === 1) {
      adjustedOutput = modelOutput.add(this.ets[this.ets.length - 1]).div(2)
      sample = this.curSample
      this.curSample = null
    } else if (this.ets.length === 2) {
      adjustedOutput = this.ets[this.ets.length - 1].mul(3).sub(this.ets[this.ets.length - 2]).div(2)
    } else if (this.ets.length === 3) {
      adjustedOutput = this.ets[this.ets.length - 1].mul(23)
        .sub(this.ets[this.ets.length - 2].mul(16))
        .add(this.ets[this.ets.length - 3].mul(5))
        .div(12)
    } else if (this.ets.length >= 4) {
      const last = this.ets.length - 1
      adjustedOutput = this.ets[last].mul(55)
        .sub(this.ets[last - 1].mul(59))
        .add(this.ets[last - 2].mul(37))
        .sub(this.ets[last - 3].mul(9))
        .mul(1 / 24)
    }

    const prevSample = this._getPrevSample(sample, timestep, prevTimestep, adjustedOutput)
    this.counter += 1
    return prevSample
  }
}

/**
 * Factory that selects the appropriate scheduler implementation for the user's configuration.
 */
export function createSchedulerInstance(baseConfig, schedulerKey, options = {}) {
  const config = cloneConfig(baseConfig)
  switch (schedulerKey) {
    case 'pndm':
      return new PNDMSchedulerCompat(config)
    case 'ddim':
      return new CustomDDIMScheduler(config, { eta: options.eta ?? 0, seed: options.seed || '' })
    default:
      throw new Error(`Unsupported scheduler selection: ${schedulerKey}`)
  }
}

/**
 * PipelineController encapsulates the lifecycle of a StableDiffusionXLPipeline instance. It ensures that only one
 * pipeline is resident at a time, allows hot-swapping the scheduler strategy, and exposes a simple generate method
 * for the UI to trigger inference.
 */
class PipelineController {
  constructor(modelManager, cache, logger = () => {}) {
    this.modelManager = modelManager
    this.cache = cache
    this.logger = logger
    this.pipeline = null
    this.pipelineModelId = null
  }

  async ensurePipeline(modelId, progressCallback) {
    if (!modelId) {
      throw new Error('No model has been loaded. Please select a checkpoint archive first.')
    }
    if (!this.cache.hasModel(modelId)) {
      throw new Error('The requested model is not available in memory. Reload the checkpoint and try again.')
    }
    if (this.pipeline && this.pipelineModelId === modelId) {
      return this.pipeline
    }
    if (this.pipeline) {
      try {
        await this.pipeline.release()
      } catch (error) {
        this.logger(`Warning while releasing previous pipeline: ${error.message}`)
      }
    }
    this.logger('Instantiating StableDiffusionXLPipeline with selected model…')
    const pipeline = await StableDiffusionXLPipeline.fromPretrained(modelId, { progressCallback })
    this.pipeline = pipeline
    this.pipelineModelId = modelId
    return pipeline
  }

  async generate({
    modelId,
    prompt,
    negativePrompt,
    steps,
    guidanceScale,
    schedulerKey,
    samplerKey,
    seed,
    progressCallback,
  }) {
    const samplerProfile = SAMPLER_PROFILES[samplerKey]
    if (!samplerProfile) {
      throw new Error(`Unknown sampler option: ${samplerKey}`)
    }

    const pipeline = await this.ensurePipeline(modelId, progressCallback)
    const schedulerConfig = cloneConfig(pipeline.scheduler?.config || {})
    const scheduler = createSchedulerInstance(schedulerConfig, schedulerKey, {
      eta: samplerProfile.eta,
      seed: seed || '',
    })
    scheduler.setTimesteps(steps)
    if (scheduler instanceof CustomDDIMScheduler) {
      scheduler.updateSampling({ eta: samplerProfile.eta, seed: seed || '' })
    }
    pipeline.scheduler = scheduler

    const result = await pipeline.run({
      prompt,
      negativePrompt,
      numInferenceSteps: steps,
      guidanceScale,
      seed,
      progressCallback,
    })
    return result
  }
}

function buildStatusText(payload) {
  if (!payload || !payload.status) return ''
  if (payload.statusText) return payload.statusText
  switch (payload.status) {
    case ProgressStatus.Downloading:
      if (payload.downloadStatus) {
        const pct = ((payload.downloadStatus.downloaded / payload.downloadStatus.size) * 100).toFixed(1)
        return `Downloading ${payload.downloadStatus.file} (${pct}%)`
      }
      return 'Downloading model files'
    case ProgressStatus.EncodingPrompt:
      return 'Encoding prompt tokens'
    case ProgressStatus.RunningUnet:
      return `UNet step ${payload.unetTimestep}/${payload.unetTotalSteps}`
    case ProgressStatus.RunningVae:
      return 'Decoding latents'
    case ProgressStatus.Done:
      return 'Generation complete'
    case ProgressStatus.Ready:
      return 'Pipeline ready'
    default:
      return payload.status
  }
}

async function renderImageToCanvas(canvas, tensor) {
  if (!tensor) return
  const imageData = await tensor.toImageData({ format: 'RGB' })
  canvas.width = imageData.width
  canvas.height = imageData.height
  const ctx = canvas.getContext('2d')
  ctx.putImageData(imageData, 0, 0)
}

/**
 * UIController wires DOM elements to the model loading and generation workflows. It focuses on orchestrating actions
 * rather than business logic so that other modules remain testable.
 */
class UIController {
  constructor(modelManager, pipelineController) {
    this.modelManager = modelManager
    this.pipelineController = pipelineController

    this.modelInput = document.getElementById('model-input')
    this.modelStatus = document.getElementById('model-status')
    this.promptInput = document.getElementById('prompt-input')
    this.negativePromptInput = document.getElementById('negative-prompt-input')
    this.stepsInput = document.getElementById('steps-input')
    this.cfgInput = document.getElementById('cfg-input')
    this.schedulerSelect = document.getElementById('scheduler-select')
    this.samplerSelect = document.getElementById('sampler-select')
    this.seedInput = document.getElementById('seed-input')
    this.generateButton = document.getElementById('generate-button')
    this.statusLine = document.getElementById('status-line')
    this.logOutput = document.getElementById('log-output')
    this.canvas = document.getElementById('result-canvas')

    this.attachEventListeners()
  }

  attachEventListeners() {
    this.modelInput.addEventListener('change', async event => {
      const files = event.target.files
      this.clearStatus()
      try {
        this.setStatus('Loading model into memory…')
        const result = await this.modelManager.loadModelFromInput(files, progress => {
          if (progress.phase === 'read') {
            const pct = ((progress.loaded / progress.total) * 100).toFixed(1)
            this.setStatus(`Reading files (${pct}%)`)
          } else if (progress.phase === 'decompress') {
            this.setStatus('Decompressing archive')
          }
        })
        this.appendLog(`Model ready: ${result.fileCount} files (${describeBytes(result.totalBytes)})`)
        this.modelStatus.textContent = `Active model: ${result.modelId}`
        this.setStatus('Model loaded. Ready to generate.')
      } catch (error) {
        this.handleError(error)
      }
    })

    this.generateButton.addEventListener('click', () => this.handleGenerate())
  }

  clearStatus() {
    this.statusLine.textContent = 'Idle.'
  }

  setStatus(message) {
    this.statusLine.textContent = message
  }

  appendLog(message) {
    const timestamp = new Date().toISOString()
    this.logOutput.textContent += `[${timestamp}] ${message}\n`
    this.logOutput.scrollTop = this.logOutput.scrollHeight
  }

  disableGenerate(disabled) {
    this.generateButton.disabled = disabled
  }

  async handleGenerate() {
    const prompt = this.promptInput.value.trim()
    if (!prompt) {
      this.handleError(new Error('Enter a prompt before starting generation.'))
      return
    }
    const modelId = this.modelManager.activeModelId
    if (!modelId) {
      this.handleError(new Error('Load a converted SDXL checkpoint before generating an image.'))
      return
    }
    const steps = Number.parseInt(this.stepsInput.value, 10)
    if (!Number.isFinite(steps) || steps <= 0) {
      this.handleError(new Error('Step count must be a positive integer.'))
      return
    }
    const guidanceScale = Number.parseFloat(this.cfgInput.value)
    if (!Number.isFinite(guidanceScale) || guidanceScale <= 0) {
      this.handleError(new Error('CFG scale must be a positive number.'))
      return
    }
    const schedulerKey = this.schedulerSelect.value
    const samplerKey = this.samplerSelect.value
    const seed = this.seedInput.value.trim() || undefined

    this.disableGenerate(true)
    this.setStatus('Preparing pipeline…')
    this.appendLog(`Starting generation with ${steps} steps, CFG ${guidanceScale}, scheduler ${schedulerKey}, sampler ${samplerKey}`)

    try {
      const progressCallback = async payload => {
        const text = buildStatusText(payload)
        if (text) {
          this.setStatus(text)
        }
        if (payload.images && payload.images.length) {
          await renderImageToCanvas(this.canvas, payload.images[0])
        }
        if (payload.status === ProgressStatus.Error) {
          throw new Error(payload.statusText || 'Pipeline reported an error state.')
        }
      }

      const images = await this.pipelineController.generate({
        modelId,
        prompt,
        negativePrompt: this.negativePromptInput.value.trim() || undefined,
        steps,
        guidanceScale,
        schedulerKey,
        samplerKey,
        seed,
        progressCallback,
      })

      const image = images && images[0]
      if (!image) {
        throw new Error('The pipeline did not return an image tensor.')
      }
      await renderImageToCanvas(this.canvas, image)
      this.appendLog('Generation complete.')
      this.setStatus('Generation complete. You can tweak settings and run again.')
    } catch (error) {
      this.handleError(error)
    } finally {
      this.disableGenerate(false)
    }
  }

  handleError(error) {
    const message = error instanceof Error ? error.message : String(error)
    this.appendLog(`Error: ${message}`)
    this.setStatus(`Error: ${message}`)
  }
}

function initializeBrowserApp() {
  const cache = new VirtualModelCache()
  setCacheImpl(cache)
  const modelManager = new ModelManager(cache, console.log)
  const pipelineController = new PipelineController(modelManager, cache, console.log)
  new UIController(modelManager, pipelineController)
}

if (typeof window !== 'undefined' && document.readyState !== 'complete') {
  window.addEventListener('DOMContentLoaded', () => {
    initializeBrowserApp()
  })
} else if (typeof window !== 'undefined') {
  initializeBrowserApp()
}

export { CustomDDIMScheduler, PNDMSchedulerCompat, PipelineController }
