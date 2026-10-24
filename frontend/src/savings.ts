// The saving as a shift of energy: what the flat setpoint would have used and cost each hour, what the plan uses and
// costs, how much energy moved from dear hours to cheap ones, and the same per day over time.
import type { DayOutcome, HourRow } from './model'

export interface ShiftRow {
  key: string
  hour: string
  dayStart: boolean
  dayLabel: string
  current: boolean
  price: number
  flatKwh: number
  planKwh: number
  /** plan minus flat: above zero the plan heats more in this hour (pre-heat), below zero it heats less (cut) */
  movedKwh: number
  flatCost: number
  planCost: number
  saving: number
  /** running total of the saving over the hours shown */
  cumulative: number
  measuredKwh: number | null
  measuredSaving: number | null
  cumulativeMeasured: number | null
}

/** One row per hour that has a price, a plan and the flat baseline; other hours cannot be compared and are left out. */
export function shiftRows(rows: HourRow[]): ShiftRow[] {
  const out: ShiftRow[] = []
  let cumulative = 0
  let cumulativeMeasured: number | null = null
  for (const r of rows) {
    if (r.price === null || r.energyPlanned === null || r.energyBaseline === null) continue
    const flatCost = r.energyBaseline * r.price
    const planCost = r.energyPlanned * r.price
    cumulative += flatCost - planCost
    if (r.savingMeasured !== null) cumulativeMeasured = (cumulativeMeasured ?? 0) + r.savingMeasured
    out.push({
      key: r.key,
      hour: r.hour,
      dayStart: r.dayStart,
      dayLabel: r.dayLabel,
      current: r.current,
      price: r.price,
      flatKwh: r.energyBaseline,
      planKwh: r.energyPlanned,
      movedKwh: r.energyPlanned - r.energyBaseline,
      flatCost,
      planCost,
      saving: flatCost - planCost,
      cumulative,
      measuredKwh: r.energyMeasured,
      measuredSaving: r.savingMeasured,
      cumulativeMeasured: r.savingMeasured === null ? null : cumulativeMeasured,
    })
  }
  return out
}

export interface ShiftTotals {
  hours: number
  flatKwh: number
  planKwh: number
  flatCost: number
  planCost: number
  saving: number
  /** saving as a share of the flat cost, 0 when the flat cost is 0 */
  percent: number
  /** energy the plan adds in the hours it pre-heats and the energy it cuts in the other hours */
  addedKwh: number
  cutKwh: number
  /** the energy-weighted price of those hours: the plan buys at the added price instead of the cut price */
  addedPrice: number | null
  cutPrice: number | null
  measuredSaving: number | null
}

export function shiftTotals(rows: ShiftRow[]): ShiftTotals {
  const sum = (pick: (r: ShiftRow) => number) => rows.reduce((total, r) => total + pick(r), 0)
  const added = rows.filter((r) => r.movedKwh > 0)
  const cut = rows.filter((r) => r.movedKwh < 0)
  const addedKwh = added.reduce((total, r) => total + r.movedKwh, 0)
  const cutKwh = cut.reduce((total, r) => total - r.movedKwh, 0)
  const measured = rows.filter((r) => r.measuredSaving !== null)
  const flatCost = sum((r) => r.flatCost)
  const saving = sum((r) => r.saving)
  return {
    hours: rows.length,
    flatKwh: sum((r) => r.flatKwh),
    planKwh: sum((r) => r.planKwh),
    flatCost,
    planCost: sum((r) => r.planCost),
    saving,
    percent: flatCost > 0 ? (saving / flatCost) * 100 : 0,
    addedKwh,
    cutKwh,
    addedPrice: addedKwh > 0 ? added.reduce((total, r) => total + r.movedKwh * r.price, 0) / addedKwh : null,
    cutPrice: cutKwh > 0 ? cut.reduce((total, r) => total - r.movedKwh * r.price, 0) / cutKwh : null,
    measuredSaving: measured.length === 0 ? null : measured.reduce((total, r) => total + (r.measuredSaving ?? 0), 0),
  }
}

export interface DailyRow extends DayOutcome {
  percent: number
  cumulative: number
}

/** The measured days with the saving as a share of the flat cost and a running total. */
export function dailyRows(days: DayOutcome[]): DailyRow[] {
  let cumulative = 0
  return days.map((d) => {
    cumulative += d.saving_sek
    return { ...d, percent: d.baseline_cost_sek > 0 ? (d.saving_sek / d.baseline_cost_sek) * 100 : 0, cumulative }
  })
}

export function dailyTotals(days: DayOutcome[]): { baseline: number; actual: number; saving: number; percent: number; compressorKwh: number; auxKwh: number } {
  const sum = (pick: (d: DayOutcome) => number) => days.reduce((total, d) => total + pick(d), 0)
  const baseline = sum((d) => d.baseline_cost_sek)
  const saving = sum((d) => d.saving_sek)
  return {
    baseline,
    actual: sum((d) => d.actual_cost_sek),
    saving,
    percent: baseline > 0 ? (saving / baseline) * 100 : 0,
    compressorKwh: sum((d) => d.compressor_kwh),
    auxKwh: sum((d) => d.aux_kwh),
  }
}
