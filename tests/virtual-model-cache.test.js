import { describe, expect, it } from 'vitest'
import { VirtualModelCache, extractArchiveToMap } from '../src/app.js'
import { zipSync } from 'fflate'

function createSampleArchive() {
  const files = {
    'model_index.json': new TextEncoder().encode('{}'),
    'unet/model.onnx': new Uint8Array([1, 2, 3]),
    'unet/config.json': new TextEncoder().encode('{}'),
    'text_encoder/model.onnx': new Uint8Array([4, 5, 6]),
    'text_encoder/config.json': new TextEncoder().encode('{}'),
    'text_encoder_2/model.onnx': new Uint8Array([7, 8, 9]),
    'text_encoder_2/config.json': new TextEncoder().encode('{}'),
    'vae_decoder/model.onnx': new Uint8Array([10, 11, 12]),
    'vae_decoder/config.json': new TextEncoder().encode('{}'),
    'tokenizer/tokenizer.json': new TextEncoder().encode('{}'),
    'tokenizer_2/tokenizer.json': new TextEncoder().encode('{}'),
    'scheduler/scheduler_config.json': new TextEncoder().encode('{}'),
  }
  return zipSync(files)
}

describe('VirtualModelCache', () => {
  it('stores and retrieves binary payloads', async () => {
    const cache = new VirtualModelCache()
    const archive = createSampleArchive()
    const map = extractArchiveToMap(archive)
    cache.registerModel('local/test', map)

    const buffer = await cache.getModelFile('local/test', 'unet/model.onnx')
    expect(new Uint8Array(buffer)).toEqual(new Uint8Array([1, 2, 3]))

    const text = await cache.getModelFile('local/test', 'scheduler/scheduler_config.json', true, { returnText: true })
    expect(text).toBe('{}')
  })

  it('throws when requested file missing', async () => {
    const cache = new VirtualModelCache()
    cache.registerModel('local/empty', new Map())
    await expect(cache.getModelFile('local/empty', 'missing.bin')).rejects.toThrow(/missing/)
  })
})
