import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parseSafetensors,
  convertHalfToFloat32,
  convertBFloat16ToFloat32,
  buildScheduler,
  selectSamplerKernel,
  createInitialLatents,
  gaussianFromUniform,
} from '../src/app.js';

function createSafetensorBuffer() {
  let headerLength = 0;
  let headerBytes = Buffer.alloc(0);
  let stable = false;
  let headerJSON = '';
  while (!stable) {
    const offsets = [8 + headerLength, 8 + headerLength + 8];
    const header = {
      tensorA: {
        dtype: 'F32',
        shape: [2],
        data_offsets: offsets,
      },
      __metadata__: { model: 'unit-test' },
    };
    headerJSON = JSON.stringify(header);
    headerBytes = Buffer.from(headerJSON, 'utf-8');
    stable = headerBytes.length === headerLength;
    headerLength = headerBytes.length;
  }
  const buffer = new ArrayBuffer(8 + headerLength + 8);
  const view = new DataView(buffer);
  view.setBigUint64(0, BigInt(headerLength), true);
  new Uint8Array(buffer, 8, headerLength).set(headerBytes);
  view.setFloat32(8 + headerLength, 1.5, true);
  view.setFloat32(8 + headerLength + 4, -2.25, true);
  return buffer;
}

test('parseSafetensors decodes float32 tensors and metadata', () => {
  const buffer = createSafetensorBuffer();
  const result = parseSafetensors(buffer);
  const tensor = result.tensors.tensorA;
  assert.equal(tensor.dtype, 'F32');
  assert.deepEqual(Array.from(tensor.shape), [2]);
  assert.deepEqual(Array.from(tensor.data), [1.5, -2.25]);
  assert.equal(result.metadata.model, 'unit-test');
});

test('convertHalfToFloat32 converts canonical half precision values', () => {
  const source = new Uint16Array([0x3c00, 0xc000, 0x7bff]);
  const floats = convertHalfToFloat32(source);
  const rounded = Array.from(floats).map((value) => Number(value.toFixed(5)));
  assert.deepEqual(rounded, [1, -2, 65504]);
});

test('convertBFloat16ToFloat32 reconstructs float values', () => {
  const source = new Uint16Array([0x3f80, 0xbf80]);
  const floats = convertBFloat16ToFloat32(source);
  const rounded = Array.from(floats).map((value) => Number(value.toFixed(5)));
  assert.deepEqual(rounded, [1, -1]);
});

test('buildScheduler returns deterministic timestep arrays', () => {
  const config = { steps: 5, scheduler: 'ddim' };
  const schedule = buildScheduler(config);
  assert.equal(schedule.type, 'ddim');
  assert.equal(schedule.timesteps.length, 5);
  assert.ok(schedule.timesteps[0] < schedule.timesteps[4]);
});

test('selectSamplerKernel exposes supported kernels', () => {
  const sampler = selectSamplerKernel('deterministic');
  assert.equal(sampler.name, 'deterministic');
  assert.equal(sampler.order, 1);
});

test('createInitialLatents produces deterministic gaussian samples with seeded generator', () => {
  function* sequence() {
    let value = 1;
    while (true) {
      yield (value % 997) / 997;
      value += 1;
    }
  }
  const iter = sequence();
  const generator = () => iter.next().value;
  const latents = createInitialLatents(2, 1, 2, generator);
  assert.equal(latents.length, 4);
  const rounded = Array.from(latents).map((v) => Number(v.toFixed(4)));
  assert.deepEqual(rounded, [3.7158, 3.4066, 3.252, 3.1452]);
});

test('gaussianFromUniform handles edge cases without blowing up', () => {
  const sample = gaussianFromUniform(1e-9, 0.25);
  assert.ok(Number.isFinite(sample));
});
