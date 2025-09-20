import { describe, expect, it } from 'vitest'
import { CustomDDIMScheduler, PNDMSchedulerCompat } from '../src/app.js'
import { Tensor } from '@xenova/transformers'

const BASE_CONFIG = {
  beta_start: 0.00085,
  beta_end: 0.012,
  beta_schedule: 'scaled_linear',
  clip_sample: false,
  num_train_timesteps: 1000,
  prediction_type: 'epsilon',
  set_alpha_to_one: false,
  steps_offset: 1,
  trained_betas: null,
}

function makeTensor(value) {
  const data = new Float32Array(16).fill(value)
  return new Tensor('float32', data, [1, 1, 4, 4])
}

describe('CustomDDIMScheduler', () => {
  it('performs a sampling step without throwing', () => {
    const scheduler = new CustomDDIMScheduler(BASE_CONFIG, { eta: 0.1, seed: 'unit' })
    scheduler.setTimesteps(4)
    const sample = makeTensor(0.5)
    const modelOutput = makeTensor(0.1)
    const next = scheduler.step(modelOutput, scheduler.timesteps.data[0], sample)
    expect(next).toBeInstanceOf(Tensor)
    expect(next.dims).toEqual(sample.dims)
  })

  it('allows eta updates', () => {
    const scheduler = new CustomDDIMScheduler(BASE_CONFIG, { eta: 0, seed: 'seed' })
    scheduler.setTimesteps(2)
    scheduler.updateSampling({ eta: 0.5, seed: 'alt' })
    const next = scheduler.step(makeTensor(0.05), scheduler.timesteps.data[0], makeTensor(0.2))
    expect(next).toBeInstanceOf(Tensor)
  })
})

describe('PNDMSchedulerCompat', () => {
  it('executes PLMS step logic', () => {
    const scheduler = new PNDMSchedulerCompat({ ...BASE_CONFIG, skip_prk_steps: true })
    scheduler.setTimesteps(3)
    const sample = makeTensor(0.4)
    const modelOutput = makeTensor(0.1)
    const timestep = scheduler.timesteps.data[0]
    const prev = scheduler.step(modelOutput, timestep, sample)
    expect(prev).toBeInstanceOf(Tensor)
  })
})
