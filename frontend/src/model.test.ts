import { describe, expect, it } from 'vitest'
import {
  planRows, formatValue, tooltipFormatter, UNITS, sortedRecommendations, planSaving, tabHref,
  timelineRows, quarterRows, hasData, dayStatus, modeOf, axisTicks, axisLabel, rangeLabel, savingsTotals, splitPredicted, firstPredicted, shiftWindows, combinedNeed, roomColour,
  type PlanResponse, type DayHour, type DayView, type HourRow,
} from './model'

const hour = (h: number, rts: number, room: number, price: number) => ({
  start: `2026-10-12T${String(h).padStart(2, '0')}:00:00Z`,
  rts_c: rts, price_sek_per_kwh: price, floor_c: 18,
  flow_target_c: 36, energy_kwh: 2, room_c_end: room, slab_c_end: 25,
  energy_cost_sek: 2 * price, reason: { binding: 'price' },
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

describe('planSaving', () => {
  const withHours = (planHours: ReturnType<typeof hour>[], flatHours: ReturnType<typeof hour>[]): PlanResponse => ({
    ...plan, plan: { ...plan.plan, hours: planHours, baseline_hours: flatHours },
  })
  const at = (h: number, price: number, kwh: number, room: number) => ({ ...hour(h, 17, room, price), energy_kwh: kwh, energy_cost_sek: kwh * price })

  it('splits the saving into using less and using it at a better time', () => {
    // flat: 2 kWh in each hour; plan: 1 kWh in the dear hour, 3 in the cheap one; same end room
    const flat = [at(0, 1, 2, 19), at(1, 3, 2, 19)]
    const planned = [at(0, 1, 3, 19), at(1, 3, 1, 19)]
    const s = planSaving(withHours(planned, flat))
    expect(s.kwhSaved).toBe(0)
    expect(s.volumeSek).toBeCloseTo(0)
    expect(s.timingSek).toBeCloseTo(2) // 1 kWh moved from a 3 kr hour to a 1 kr hour
    expect(s.netSek).toBeCloseTo(2)
  })

  it('shows how much colder the room ends beside the saving', () => {
    const flat = [at(0, 2, 2, 19.5), at(1, 2, 2, 19.5)]
    const planned = [at(0, 2, 1, 19.2), at(1, 2, 1, 19.0)]
    const s = planSaving(withHours(planned, flat))
    expect(s.volumeSek).toBeCloseTo(4)
    expect(s.endDiffC).toBeCloseTo(-0.5)
  })

  it('is zero for a plan equal to the flat setpoint', () => {
    const same = [at(0, 1, 2, 19), at(1, 3, 2, 19)]
    const s = planSaving(withHours(same, same))
    expect(s.netSek).toBeCloseTo(0)
    expect(s.volumeSek).toBeCloseTo(0)
  })
})

import { describeReason, formatAge, fitQuality } from './model'

describe('describeReason', () => {
  it('explains known binding values and passes unknown ones through', () => {
    expect(describeReason('shift_sell')).toMatch(/dear/i)
    expect(describeReason('shift_buy')).toMatch(/cheap/i)
    expect(describeReason('shift_guard')).toMatch(/limit/i)
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
  start, past: false, price_sek_per_kwh: 1, price_quarters: [1, 1, 1, 1], floor_c: 18,
  salon_measured: null, salon_planned: null, need_ground_kw: null, need_salon_kw: null, delivered_ground_kw: null, flow_measured_c: null, flow_target_c: null, thermostat: null, thermostat_target_c: null, loops_expected: null,
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

  it('ticks the even hours on the hour, whatever the row length', () => {
    expect(axisTicks(keys)).toEqual([keys[0], keys[2], keys[4], keys[6], keys[8]])
    const quarters = ['2026-10-14 01:45', '2026-10-14 02:00', '2026-10-14 02:15', '2026-10-14 03:00', '2026-10-14 04:00']
    expect(axisTicks(quarters)).toEqual(['2026-10-14 02:00', '2026-10-14 04:00'])
  })

  it('labels the hours 00, 01, 02 like BESS Manager', () => {
    expect(axisLabel('2026-10-14 00:00')).toBe('00')
    expect(axisLabel('2026-10-14 06:00')).toBe('06')
    expect(axisLabel('2026-10-14 23:15')).toBe('23')
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

describe('history and prediction', () => {
  const rows = [
    { key: 'a', past: true, plan: 16 },
    { key: 'b', past: true, plan: 18 },
    { key: 'c', past: false, plan: 20 },
    { key: 'd', past: false, plan: 17 },
  ]

  it('keeps the hours that have passed under the key and the rest under its predicted twin', () => {
    const split = splitPredicted(rows, ['plan'])
    expect(split.map((r) => r.plan)).toEqual([16, 18, null, null])
    expect(split.map((r) => r.planPredicted)).toEqual([null, 18, 20, 17])
  })

  it('has no prediction when everything is history and no history when nothing is', () => {
    expect(splitPredicted(rows.slice(0, 2), ['plan']).map((r) => r.planPredicted)).toEqual([null, null])
    const future = splitPredicted(rows.slice(2), ['plan'])
    expect(future.map((r) => r.plan)).toEqual([null, null])
    expect(future.map((r) => r.planPredicted)).toEqual([20, 17])
  })

  it('finds where the prediction starts', () => {
    expect(firstPredicted(rows)).toBe('c')
    expect(firstPredicted(rows.slice(0, 2))).toBeUndefined()
  })
})

describe('dayStatus', () => {
  it('says loading only until the first answer has come', () => {
    expect(dayStatus(false, false, 'Nothing to show')).toBe('Loading...')
  })
  it('says why there is no plan when the add-on has none', () => {
    expect(dayStatus(true, false, 'Nothing to show')).toMatch(/No plan.*Control/)
  })
  it('keeps the view’s own text when the day is there but empty', () => {
    expect(dayStatus(true, true, 'Nothing to show')).toBe('Nothing to show')
  })
})

describe('shift windows', () => {
  const row = (key: string, reason: string | null, thermostat: string | null = null) =>
    ({ key, reason, thermostat }) as unknown as HourRow
  it('joins the hours of a sell, and ends at the next hour', () => {
    const rows = [row('a', 'shift_none'), row('b', 'shift_sell'), row('c', 'shift_sell'), row('d', 'shift_none')]
    expect(shiftWindows(rows)).toEqual([{ kind: 'sell', x1: 'b', x2: 'd' }])
  })
  it('shows a buy', () => {
    expect(shiftWindows([row('a', 'shift_buy'), row('b', null)])).toEqual([{ kind: 'buy', x1: 'a', x2: 'b' }])
  })
  it('counts a sell the guard ended by its closing thermostats', () => {
    expect(shiftWindows([row('a', 'shift_guard', 'closing'), row('b', 'shift_none')])).toEqual([{ kind: 'sell', x1: 'a', x2: 'b' }])
  })
  it('runs to the last hour', () => {
    expect(shiftWindows([row('a', 'shift_sell'), row('b', 'shift_sell')])).toEqual([{ kind: 'sell', x1: 'a', x2: 'b' }])
  })
})

describe('combined need', () => {
  it('adds the zones', () => {
    expect(combinedNeed(1.5, 0.5)).toBe(2)
  })
  it('is unknown when a zone is unknown', () => {
    expect(combinedNeed(null, 0.5)).toBeNull()
  })
})

describe('room colours', () => {
  it('gives a room the same colour whatever the name looks like', () => {
    expect(roomColour('lk_bibliotek')).toBe(roomColour('bibliotek'))
    expect(roomColour('vardagsrum')).not.toBe(roomColour('bibliotek'))
  })
})
