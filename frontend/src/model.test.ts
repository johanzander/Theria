import { describe, expect, it } from 'vitest'
import {
  planRows, roomRows, formatValue, tooltipFormatter, UNITS, sortedRecommendations, savingsSummary, tabHref,
  timelineRows, quarterRows, hasData, modeOf, axisTicks, axisLabel, rangeLabel, savingsTotals,
  type PlanResponse, type DayHour, type DayView,
} from './model'

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

describe('sortedRecommendations', () => {
  const rec = (key: string, status: 'ok' | 'consider' | 'no data') =>
    ({ key, title: key, current: '', observed: '', suggestion: null, reason: '', status })
  it('puts what is worth considering first and keeps the order otherwise', () => {
    const sorted = sortedRecommendations([rec('a', 'ok'), rec('b', 'consider'), rec('c', 'no data'), rec('d', 'consider')])
    expect(sorted.map((r) => r.key)).toEqual(['b', 'd', 'a', 'c'])
  })
})

describe('roomRows', () => {
  const rooms = [
    { room: 'vardagsrum', temperature_c: 22.34, rate_c_per_h: 0.123, state: 'closed', limit_c: 22, rise_c: 0.7, close_at_c: 21.3, valve_fraction: 0 },
    { room: 'bibliotek', temperature_c: null, rate_c_per_h: null, state: 'unknown', limit_c: 22, rise_c: 0, close_at_c: 21.9, valve_fraction: null },
    { room: 'kok', temperature_c: 20, rate_c_per_h: -0.04, state: 'open', limit_c: 22, rise_c: 0.2, close_at_c: 21.8, valve_fraction: 0.6 },
  ]
  const rows = roomRows(rooms)

  it('formats the readings and flags a closed or unreadable room', () => {
    expect(rows[0]).toMatchObject({ temperature: '22.3 °C', rate: '+0.12 °C/h', valves: '0 % open', tone: 'warn' })
    expect(rows[1]).toMatchObject({ temperature: '-', rate: '-', valves: '-', tone: 'bad' })
    expect(rows[2]).toMatchObject({ rate: '-0.04 °C/h', valves: '60 % open', tone: undefined })
  })

  it('shows the level it closes at and the rise it has learned', () => {
    expect(rows[0]).toMatchObject({ closeAt: '21.3 °C', rise: '0.7 °C' })
  })
})

describe('tooltip formatting', () => {
  it('rounds to a few decimals and adds the unit', () => {
    expect(formatValue(19.123456, UNITS.temp)).toBe('19.1 °C')
    expect(formatValue(1.23456, UNITS.price)).toBe('1.23 kr/kWh')
    expect(formatValue(3.97, UNITS.sek)).toBe('4.0 kr')
  })

  it('shows a dash for a value that is not there', () => {
    expect(formatValue(null, UNITS.temp)).toBe('-')
    expect(formatValue(undefined, UNITS.temp)).toBe('-')
    expect(formatValue(NaN, UNITS.temp)).toBe('-')
  })

  it('picks the unit from the series name and keeps the name', () => {
    const format = tooltipFormatter((name) => (name.includes('Price') ? UNITS.price : UNITS.temp))
    expect(format(1.5678, 'Price per quarter (SEK/kWh)')).toEqual(['1.57 kr/kWh', 'Price per quarter (SEK/kWh)'])
    expect(format(18, 'Plan setpoint')).toEqual(['18.0 °C', 'Plan setpoint'])
  })
})

const dayHour = (start: string, extra: Partial<DayHour> = {}): DayHour => ({
  start, past: false, price_sek_per_kwh: 1, price_quarters: [1, 1, 1, 1], floor_c: 18, extreme: false,
  rts_measured: null, rts_planned: 18, rts_baseline: 18, room_measured: null, room_planned: 19.5, room_baseline: 19.5,
  energy_measured_kwh: null, energy_planned_kwh: 1.5, energy_baseline_kwh: 2, saving_planned_sek: 0.5, saving_measured_sek: null,
  flow_planned_c: 35, flow_baseline_c: 36, reason: 'price_trade_off', loops_open: {}, thermostats: {}, ...extra,
})

const day = (date: string, now: string, hours: DayHour[]): DayView => ({ day: date, now, loops_total: {}, hours })

describe('timelineRows', () => {
  const today = day('2026-10-14', '2026-10-14T01:00:00+00:00', [
    dayHour('2026-10-14T00:00:00+00:00', { past: true, rts_measured: 18, room_measured: 19.4, energy_measured_kwh: 1.7, saving_measured_sek: 0.4 }),
    dayHour('2026-10-14T01:00:00+00:00', { rts_planned: 19 }),
  ])
  const tomorrow = day('2026-10-15', '2026-10-14T01:00:00+00:00', [dayHour('2026-10-15T00:00:00+00:00', { rts_planned: 16 })])
  const rows = timelineRows([today, tomorrow])

  it('lists the hours of every day with data in order and marks where a day starts', () => {
    expect(rows).toHaveLength(3)
    expect(rows.map((r) => r.dayStart)).toEqual([true, false, true])
    expect(new Set(rows.map((r) => r.key)).size).toBe(3)
  })

  it('keeps plan, baseline and what was measured side by side for the same hour', () => {
    expect(rows[0]).toMatchObject({ rtsMeasured: 18, rtsPlanned: 18, rtsBaseline: 18, roomMeasured: 19.4, roomPlanned: 19.5, energyMeasured: 1.7, savingMeasured: 0.4 })
  })

  it('says whether the plan pre-heats, cuts or holds against the flat setpoint', () => {
    expect(rows.map((r) => r.mode)).toEqual(['hold', 'preheat', 'cut'])
    expect(modeOf(null, 18)).toBeNull()
  })

  it('marks the current hour', () => {
    expect(rows.map((r) => r.current)).toEqual([false, true, false])
  })

  it('leaves out a day that has nothing to show', () => {
    const empty = day('2026-10-16', '2026-10-14T01:00:00+00:00', [
      dayHour('2026-10-16T00:00:00+00:00', { price_sek_per_kwh: null, rts_planned: null, rts_baseline: null }),
    ])
    expect(hasData(empty)).toBe(false)
    expect(timelineRows([today, empty])).toHaveLength(2)
    expect(timelineRows([null])).toEqual([])
  })
})

describe('quarterRows', () => {
  const view = day('2026-10-14', '2026-10-14T01:00:00+00:00', [
    dayHour('2026-10-14T00:00:00+00:00', { price_quarters: [1, 2, 3, 4], rts_planned: 17 }),
    dayHour('2026-10-14T01:00:00+00:00', { price_quarters: [5, 6, 7, null] }),
  ])
  const rows = quarterRows([view])

  it('has four rows per hour with the quarter price and the hours setpoint held through them', () => {
    expect(rows).toHaveLength(8)
    expect(rows.map((r) => r.price)).toEqual([1, 2, 3, 4, 5, 6, 7, null])
    expect(rows.slice(0, 4).every((r) => r.rtsPlanned === 17)).toBe(true)
  })

  it('has a unique key per quarter and marks now on the first quarter of the current hour', () => {
    expect(new Set(rows.map((r) => r.key)).size).toBe(8)
    expect(rows.map((r) => r.current)).toEqual([false, false, false, false, true, false, false, false])
  })
})

describe('axis helpers', () => {
  const keys = Array.from({ length: 10 }, (_, i) => `2026-10-14 ${String(i).padStart(2, '0')}:00`)

  it('ticks every n-th key', () => {
    expect(axisTicks(keys, 3)).toEqual([keys[0], keys[3], keys[6], keys[9]])
  })

  it('names the day in place of the time at midnight', () => {
    expect(axisLabel('2026-10-14 00:00', { '2026-10-14': 'ons 14' })).toBe('ons 14')
    expect(axisLabel('2026-10-14 00:00', {})).toBe('00:00')
    expect(axisLabel('2026-10-14 06:00', {})).toBe('06:00')
  })

  it('gives the tooltip the span an hour or a quarter covers', () => {
    expect(rangeLabel('2026-10-14 19:00')).toBe('2026-10-14  19:00-20:00')
    expect(rangeLabel('2026-10-14 19:15', 15)).toBe('2026-10-14  19:15-19:30')
  })
})

describe('savingsTotals', () => {
  it('adds the planned saving of every hour and the measured one of the hours that have it', () => {
    const rows = timelineRows([
      day('2026-10-14', '2026-10-14T02:00:00+00:00', [
        dayHour('2026-10-14T00:00:00+00:00', { saving_planned_sek: 1, saving_measured_sek: 0.5 }),
        dayHour('2026-10-14T01:00:00+00:00', { saving_planned_sek: 2, saving_measured_sek: -0.25 }),
        dayHour('2026-10-14T02:00:00+00:00', { saving_planned_sek: 3, saving_measured_sek: null }),
      ]),
    ])
    expect(savingsTotals(rows)).toEqual({ planned: 6, measured: 0.25, hours: 2 })
  })

  it('has no measured total when nothing was measured', () => {
    expect(savingsTotals(timelineRows([day('2026-10-14', '', [dayHour('2026-10-14T00:00:00+00:00')])])).measured).toBeNull()
  })
})
