import { describe, expect, it } from 'vitest'
import { planRows, savingsSummary, type PlanResponse } from './model'

const hour = (h: number, rts: number, room: number, price: number) => ({
  start: `2026-10-12T${String(h).padStart(2, '0')}:00:00Z`,
  rts_c: rts, price_sek_per_kwh: price, extreme: false, floor_c: 18,
  flow_target_c: 36, energy_kwh: 2, room_c_end: room, slab_c_end: 25,
  energy_cost_sek: 2 * price, comfort_cost_sek: 0, tail: false, reason: { binding: 'price' },
})

const plan: PlanResponse = {
  computed_at: '2026-10-12T00:00:00Z', mode: 'shadow', age_seconds: 60, comfort_outlook: null,
  plan: {
    hours: [hour(0, 17, 19.6, 1), hour(1, 18.5, 19.7, 3)],
    baseline_hours: [hour(0, 18, 19.8, 1), hour(1, 18, 19.8, 3)],
    objective_sek: 10, baseline_objective_sek: 12, floor_violation_hours: 0,
  },
}

describe('planRows', () => {
  it('joins the plan and the baseline hour by hour', () => {
    const rows = planRows(plan)
    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({ rts: 17, baselineRts: 18, room: 19.6, baselineRoom: 19.8, price: 1 })
    expect(rows[1].rts).toBe(18.5)
  })
})

describe('savingsSummary', () => {
  it('reports the plan advantage as a percentage of the baseline', () => {
    expect(savingsSummary(plan)).toEqual({ savingSek: 2, percent: (2 / 12) * 100 })
  })
  it('is zero when the baseline cost is zero', () => {
    const p = { ...plan, plan: { ...plan.plan, baseline_objective_sek: 0, objective_sek: 0 } }
    expect(savingsSummary(p).percent).toBe(0)
  })
})

import { describeReason, formatAge, fitQuality } from './model'

describe('describeReason', () => {
  it('explains known binding values and passes unknown ones through', () => {
    expect(describeReason('rts_min')).toMatch(/lowest allowed setpoint/i)
    expect(describeReason('flow_cap')).toMatch(/flow/i)
    expect(describeReason('something_new')).toBe('something_new')
  })
})

describe('formatAge', () => {
  it('uses minutes, hours and days', () => {
    expect(formatAge(5)).toBe('5 min')
    expect(formatAge(180)).toBe('3 h')
    expect(formatAge(4333)).toBe('3 days')
  })
})

describe('fitQuality', () => {
  it('compares the model error with just assuming no change', () => {
    expect(fitQuality(0.48, 0.47)).toBe('no better than assuming no change')
    expect(fitQuality(0.3, 0.5)).toBe('better than assuming no change')
  })
})

import { statusTone } from './model'

describe('statusTone', () => {
  it('maps check statuses to colours', () => {
    expect(statusTone('ok')).toBe('good')
    expect(statusTone('warn')).toBe('warn')
    expect(statusTone('fail')).toBe('bad')
  })
})

import { parameterSeries, type ParameterRecord } from './model'

describe('parameterSeries', () => {
  it('picks one parameter over time and skips records without it', () => {
    const history: ParameterRecord[] = [
      { at: '2026-10-10T03:00:00+00:00', kind: 'refit', note: '', params: { gain: 0.4 } },
      { at: '2026-10-11T03:00:00+00:00', kind: 'margin', note: '', params: { margin: 0.25 } },
      { at: '2026-10-12T03:00:00+00:00', kind: 'refit', note: '', params: { gain: 0.45 } },
    ]
    expect(parameterSeries(history, 'gain')).toEqual([
      { label: '2026-10-10', value: 0.4 },
      { label: '2026-10-12', value: 0.45 },
    ])
  })
})

import { outlookText, type ComfortOutlook } from './model'

describe('outlookText', () => {
  const base: ComfortOutlook = { start: '2026-10-13T04:00:00+00:00', room_c: 19.4, target_c: 19.5, deficit_c: 0.1, extra_rate_c_per_h: 0.03, lead_hours: 3.3 }
  it('says when the room is at the target', () => {
    expect(outlookText({ ...base, room_c: 19.6, deficit_c: 0 }).note).toContain('at the target')
  })
  it('treats a negligible shortfall as being at the target', () => {
    expect(outlookText({ ...base, room_c: 19.47, deficit_c: 0.03 }).note).toContain('at the target')
  })
  it('gives the deficit and the lead time', () => {
    const text = outlookText(base)
    expect(text.value).toBe('19.4 °C')
    expect(text.note).toContain('0.1 °C under the target')
    expect(text.note).toContain('about 3 h')
  })
  it('says so when it cannot be recovered', () => {
    expect(outlookText({ ...base, lead_hours: null }).note).toContain('cannot be recovered')
  })
})
