import { describe, expect, it } from 'vitest'
import type { DayOutcome, HourRow } from './model'
import { dailyRows, dailyTotals, shiftRows, shiftTotals } from './savings'

const row = (hour: number, price: number | null, flat: number | null, plan: number | null, measuredSaving: number | null = null): HourRow => ({
  key: `2026-10-12 ${String(hour).padStart(2, '0')}:00`, hour: `${String(hour).padStart(2, '0')}:00`, dayStart: hour === 0, dayLabel: 'mån 12',
  current: false, past: true, price, floor: 18, extreme: false, rtsMeasured: null, rtsPlanned: null, rtsBaseline: null,
  roomMeasured: null, roomPlanned: null, roomBaseline: null, energyMeasured: null, energyPlanned: plan, energyBaseline: flat,
  savingPlanned: null, savingMeasured: measuredSaving, flowPlanned: null, flowBaseline: null, reason: null, mode: null, loops: {}, thermostats: {},
})

describe('shiftRows', () => {
  it('costs the flat and the planned energy at the hour price and runs the saving up', () => {
    const rows = shiftRows([row(0, 0.5, 2, 3), row(1, 2, 2, 1)])
    expect(rows.map((r) => r.movedKwh)).toEqual([1, -1])
    expect(rows[0].saving).toBeCloseTo(-0.5) // heats one kWh more at 0.5
    expect(rows[1].saving).toBeCloseTo(2) // heats one kWh less at 2
    expect(rows.map((r) => r.cumulative)).toEqual([-0.5, 1.5])
  })

  it('leaves out hours without a price, a plan or a baseline', () => {
    expect(shiftRows([row(0, null, 2, 2), row(1, 1, null, 2), row(2, 1, 2, null), row(3, 1, 2, 2)]).map((r) => r.hour)).toEqual(['03:00'])
  })

  it('runs the measured saving up only over hours that have one', () => {
    const rows = shiftRows([row(0, 1, 2, 2), row(1, 1, 2, 2, 0.4), row(2, 1, 2, 2, -0.1)])
    expect(rows[0].cumulativeMeasured).toBeNull()
    expect(rows[1].cumulativeMeasured).toBeCloseTo(0.4)
    expect(rows[2].cumulativeMeasured).toBeCloseTo(0.3)
  })
})

describe('shiftTotals', () => {
  it('adds up the cost, the saving and the energy moved with the price it moved from and to', () => {
    const t = shiftTotals(shiftRows([row(0, 0.5, 2, 3), row(1, 2, 2, 1), row(2, 1, 2, 2)]))
    expect(t.flatKwh).toBe(6)
    expect(t.planKwh).toBe(6)
    expect(t.flatCost).toBeCloseTo(0.5 * 2 + 2 * 2 + 1 * 2)
    expect(t.saving).toBeCloseTo(1.5)
    expect(t.percent).toBeCloseTo((1.5 / 7) * 100)
    expect(t.addedKwh).toBe(1)
    expect(t.cutKwh).toBe(1)
    expect(t.addedPrice).toBeCloseTo(0.5)
    expect(t.cutPrice).toBeCloseTo(2)
  })

  it('has no prices for moves that did not happen and no percentage without a cost', () => {
    const t = shiftTotals(shiftRows([row(0, 0, 2, 2)]))
    expect([t.addedPrice, t.cutPrice, t.percent]).toEqual([null, null, 0])
  })

  it('reports the measured saving only when some hour has one', () => {
    expect(shiftTotals(shiftRows([row(0, 1, 2, 2)])).measuredSaving).toBeNull()
    expect(shiftTotals(shiftRows([row(0, 1, 2, 2, 0.4), row(1, 1, 2, 2, 0.2)])).measuredSaving).toBeCloseTo(0.6)
  })
})

const day = (d: string, baseline: number, actual: number): DayOutcome => ({
  day: d, coverage: 1, compressor_kwh: 40, aux_kwh: 1, actual_cost_sek: actual, baseline_cost_sek: baseline,
  saving_sek: baseline - actual, mean_rts_c: 18, room_min_c: 19,
})

describe('dailyRows and dailyTotals', () => {
  const days = [day('2026-10-10', 100, 90), day('2026-10-11', 80, 84)]
  it('gives each day its share of the flat cost and a running total', () => {
    const rows = dailyRows(days)
    expect(rows.map((r) => r.percent)).toEqual([10, -5])
    expect(rows.map((r) => r.cumulative)).toEqual([10, 6])
  })
  it('sums the days', () => {
    const t = dailyTotals(days)
    expect([t.baseline, t.actual, t.saving, t.compressorKwh, t.auxKwh]).toEqual([180, 174, 6, 80, 2])
    expect(t.percent).toBeCloseTo((6 / 180) * 100)
  })
})
