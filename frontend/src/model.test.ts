import { describe, expect, it } from 'vitest'
import { planRows, dayRows, savingsSummary, tabHref, type PlanResponse } from './model'

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

import { warmupText, type WarmupFit } from './model'

describe('warmupText', () => {
  const fit: WarmupFit = { scale: 1.1, se: 0.2, applied_scale: 1, response_c_per_c: 0.15, predicted_c_per_c: 0.14, cooling_response_c_per_c: 0.2, cooling_scale: 2.1, cooling_se: 0.4, windows: 600, step_windows: 80 }
  it('says when the measured speed is not used', () => {
    expect(warmupText(fit)).toContain('1.10 ± 0.20')
    expect(warmupText(fit)).toContain('not used')
  })
  it('says what the planner uses when the difference is real', () => {
    expect(warmupText({ ...fit, scale: 1.4, applied_scale: 1.4 })).toContain('1.40 x the model')
  })
})

describe('tabHref', () => {
  it('stays on the current document whatever the base is', () => {
    expect(tabHref('control', { pathname: '/api/hassio_ingress/abc/', search: '' })).toBe('/api/hassio_ingress/abc/#control')
  })
})

describe('dayRows', () => {
  const hours = ['2026-10-12T14:00:00+00:00', '2026-10-12T15:00:00+00:00', '2026-10-12T16:00:00+00:00'].map((start, i) => ({
    start, past: i < 2, price_sek_per_kwh: 1, floor_c: 18, extreme: false,
    rts_c: 18 + i, room_c: 20 + i, baseline_rts_c: i < 2 ? null : 18, baseline_room_c: i < 2 ? null : 20,
  }))
  const rows = dayRows({ day: '2026-10-12', now: '2026-10-12T16:00:00+00:00', hours })

  it('keeps every hour of the day, not only the ones still to come', () => {
    expect(rows).toHaveLength(3)
    expect(rows[0].label).toMatch(/^\d\d:00$/)
  })

  it('shows what was measured before now and the plan from now on', () => {
    expect(rows[0]).toMatchObject({ measuredRts: 18, plannedRts: null, measuredRoom: 20, baselineRts: null })
    expect(rows[2]).toMatchObject({ measuredRts: null, plannedRts: 20, plannedRoom: 22, baselineRts: 18 })
    expect(rows.map((r) => r.current)).toEqual([false, false, true])
  })
})
