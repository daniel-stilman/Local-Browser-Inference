import { describe, it, before } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { promises as fs } from 'node:fs';
import {
  buildToyModel,
  createScheduler,
  encodePrompt,
  initializeApp,
  parseSafetensors,
  sampleDiffusion,
} from '../src/script.js';

function buildTestCheckpoint() {
  const latentShape = [4, 8, 8];
  const latentSize = latentShape[0] * latentShape[1] * latentShape[2];
  const hiddenSize = 16;
  const timeSize = 8;
  const condSize = 8;
  const imageShape = [8, 8];
  const imageSize = imageShape[0] * imageShape[1] * 3;
  const tensors = [];

  const latentWeight = new Float32Array(hiddenSize * latentSize);
  for (let i = 0; i < latentWeight.length; i++) {
    latentWeight[i] = Math.sin(i * 0.01) * 0.05;
  }
  const latentBias = new Float32Array(hiddenSize);
  for (let i = 0; i < latentBias.length; i++) {
    latentBias[i] = Math.cos(i * 0.07) * 0.01;
  }
  const timeWeight = new Float32Array(hiddenSize * timeSize);
  for (let i = 0; i < timeWeight.length; i++) {
    timeWeight[i] = Math.cos(i * 0.02) * 0.04;
  }
  const timeBias = new Float32Array(hiddenSize);
  for (let i = 0; i < timeBias.length; i++) {
    timeBias[i] = Math.sin(i * 0.05) * 0.02;
  }
  const condWeight = new Float32Array(hiddenSize * condSize);
  for (let i = 0; i < condWeight.length; i++) {
    condWeight[i] = Math.sin(i * 0.03) * 0.03;
  }
  const condBias = new Float32Array(hiddenSize);
  for (let i = 0; i < condBias.length; i++) {
    condBias[i] = Math.cos(i * 0.09) * 0.015;
  }
  const outputWeight = new Float32Array(latentSize * hiddenSize);
  for (let i = 0; i < outputWeight.length; i++) {
    outputWeight[i] = Math.cos(i * 0.005) * 0.04;
  }
  const outputBias = new Float32Array(latentSize);
  for (let i = 0; i < outputBias.length; i++) {
    outputBias[i] = Math.sin(i * 0.02) * 0.01;
  }
  const decoderWeight = new Float32Array(imageSize * latentSize);
  for (let i = 0; i < decoderWeight.length; i++) {
    decoderWeight[i] = Math.cos(i * 0.004) * 0.02;
  }
  const decoderBias = new Float32Array(imageSize);
  for (let i = 0; i < decoderBias.length; i++) {
    decoderBias[i] = Math.sin(i * 0.03) * 0.01;
  }
  tensors.push(['model.latent.weight', 'F32', [hiddenSize, latentSize], latentWeight]);
  tensors.push(['model.latent.bias', 'F32', [hiddenSize], latentBias]);
  tensors.push(['model.time.weight', 'F32', [hiddenSize, timeSize], timeWeight]);
  tensors.push(['model.time.bias', 'F32', [hiddenSize], timeBias]);
  tensors.push(['model.cond.weight', 'F32', [hiddenSize, condSize], condWeight]);
  tensors.push(['model.cond.bias', 'F32', [hiddenSize], condBias]);
  tensors.push(['model.output.weight', 'F32', [latentSize, hiddenSize], outputWeight]);
  tensors.push(['model.output.bias', 'F32', [latentSize], outputBias]);
  tensors.push(['model.decoder.weight', 'F32', [imageSize, latentSize], decoderWeight]);
  tensors.push(['model.decoder.bias', 'F32', [imageSize], decoderBias]);
  tensors.push(['model.latent_shape', 'I32', [3], new Int32Array(latentShape)]);
  tensors.push(['model.image_shape', 'I32', [2], new Int32Array(imageShape)]);

  const header = {};
  const buffers = [];
  let offset = 0;
  for (const [name, dtype, shape, data] of tensors) {
    const buffer = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
    header[name] = {
      dtype,
      shape,
      data_offsets: [offset, offset + buffer.byteLength],
    };
    buffers.push(buffer);
    offset += buffer.byteLength;
  }
  const headerBuffer = Buffer.from(JSON.stringify(header), 'utf-8');
  const headerLength = Buffer.alloc(8);
  headerLength.writeBigUInt64LE(BigInt(headerBuffer.length));
  const body = Buffer.concat([headerLength, headerBuffer, ...buffers]);
  return body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength);
}

describe('safetensors parsing', () => {
  it('parses tensors and builds model metadata', () => {
    const buffer = buildTestCheckpoint();
    const { tensors } = parseSafetensors(buffer);
    assert.equal(tensors.size, 12);
    const model = buildToyModel(tensors);
    assert.deepEqual(model.latentShape, { channels: 4, height: 8, width: 8 });
    assert.deepEqual(model.imageShape, { height: 8, width: 8 });
    assert.equal(model.conditioningSize, 8);
  });
});

describe('prompt encoding', () => {
  it('produces deterministic embeddings', () => {
    const first = encodePrompt('glowing nebula', 8);
    const second = encodePrompt('glowing nebula', 8);
    const third = encodePrompt('dark forest', 8);
    assert.deepEqual(Array.from(first), Array.from(second));
    assert.notDeepEqual(Array.from(first), Array.from(third));
  });
});

describe('scheduler', () => {
  it('creates monotonic alphas and sigmas', () => {
    const scheduler = createScheduler(5, 'ddim');
    for (let i = 1; i < scheduler.alphas.length; i++) {
      assert.ok(scheduler.alphas[i] > scheduler.alphas[i - 1]);
    }
    assert.ok(scheduler.sigmas[0] > scheduler.sigmas[4]);
  });
});

describe('diffusion pipeline', () => {
  let model;
  before(() => {
    const buffer = buildTestCheckpoint();
    const { tensors } = parseSafetensors(buffer);
    model = buildToyModel(tensors);
  });

  it('runs sampling deterministically', () => {
    const result = sampleDiffusion({
      model,
      prompt: 'crystalline waterfall',
      negativePrompt: 'blurry',
      steps: 6,
      schedulerType: 'ddim',
      samplerType: 'ddim',
      cfgScale: 4.5,
      seed: 12345,
    });
    assert.equal(result.image.length, model.imageShape.height * model.imageShape.width * 4);
    const checksum = Math.round(result.image.slice(0, 20).reduce((sum, value) => sum + value, 0));
    assert.equal(checksum, 3315);
  });

  it('supports Euler sampling', () => {
    const result = sampleDiffusion({
      model,
      prompt: 'dreamscape',
      negativePrompt: '',
      steps: 5,
      schedulerType: 'euler',
      samplerType: 'euler',
      cfgScale: 3,
      seed: 9876,
    });
    const checksum = Math.round(result.latents.slice(0, 10).reduce((sum, value) => sum + value, 0) * 1e6);
    assert.equal(checksum, 1011803);
  });
});

describe('UI integration', () => {
  it('loads checkpoint and triggers sampling from the form', async () => {
    const html = await fs.readFile('index.html', 'utf-8');
    const dom = new JSDOM(html, {
      runScripts: 'dangerously',
      resources: 'usable',
    });
    const { document } = dom.window;
    const buffer = buildTestCheckpoint();
    const { tensors } = parseSafetensors(buffer);
    const model = buildToyModel(tensors);
    const fakeResult = sampleDiffusion({
      model,
      prompt: 'ui prompt',
      negativePrompt: '',
      steps: 2,
      schedulerType: 'ddim',
      samplerType: 'ddim',
      cfgScale: 4,
      seed: 1,
    });
    const renderCalls = [];
    initializeApp(document, {
      loadModelFromFile: async () => model,
      runPipeline: async () => fakeResult,
      renderImage: (...args) => {
        renderCalls.push(args);
      },
      reportStatus: () => {},
    });
    const fileInput = document.getElementById('checkpoint-input');
    const form = document.getElementById('inference-form');
    const promptInput = document.getElementById('prompt-input');
    promptInput.value = 'ui prompt';
    const file = new dom.window.File([new Uint8Array(buffer)], 'toy.safetensors');
    Object.defineProperty(fileInput, 'files', {
      value: [file],
      configurable: true,
    });
    fileInput.dispatchEvent(new dom.window.Event('change'));
    await Promise.resolve();
    await Promise.resolve();
    form.dispatchEvent(new dom.window.Event('submit'));
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(renderCalls.length, 1);
  });
});
