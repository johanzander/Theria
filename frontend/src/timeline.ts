// The charts and tables show whole days, 00:00 to 24:00 local time: the day asked for and, when
// there is anything to show for it, the next one. Every hour carries the plan, the flat baseline
// and what was measured side by side.

export interface DayHour {
  start: string
  past: boolean
  price_sek_per_kwh: number | null
  /** the four 15-minute prices of the hour, null for a quarter without a price */
  price_quarters: (number | null)[]
  floor_c: number
  extreme: boolean
  rts_measured: number | null
  rts_planned: number | null
  rts_baseline: number | null
  room_measured: number | null
  room_planned: number | null
  room_baseline: number | null
  energy_measured_kwh: number | null
  energy_planned_kwh: number | null
  energy_baseline_kwh: number | null
  saving_planned_sek: number | null
  saving_measured_sek: number | null
  flow_planned_c: number | null
  flow_baseline_c: number | null
  reason: string | null
  loops_open: Record<string, number | null>
  thermostats: Record<string, { room_c: number | null; set_c: number | null }>
}

export interface DayView {
  day: string
  now: string
  loops_total: Record<string, number>
  hours: DayHour[]
}

export type Mode = 'preheat' | 'cut' | 'hold'

export interface HourRow {
  /** unique local hour, 'YYYY-MM-DD HH:00' */
  key: string
  hour: string
  dayStart: boolean
  dayLabel: string
  current: boolean
  past: boolean
  price: number | null
  floor: number
  extreme: boolean
  rtsMeasured: number | null
  rtsPlanned: number | null
  rtsBaseline: number | null
  roomMeasured: number | null
  roomPlanned: number | null
  roomBaseline: number | null
  energyMeasured: number | null
  energyPlanned: number | null
  energyBaseline: number | null
  savingPlanned: number | null
  savingMeasured: number | null
  flowPlanned: number | null
  flowBaseline: number | null
  reason: string | null
  mode: Mode | null
  loops: Record<string, number | null>
  thermostats: DayHour['thermostats']
}

const localKey = (iso: string, extraMinutes = 0) => {
  const d = new Date(new Date(iso).getTime() + extraMinutes * 60_000)
  return d.toLocaleString('sv-SE').slice(0, 16)
}

const dayLabelOf = (iso: string) =>
  new Date(iso).toLocaleDateString('sv-SE', { weekday: 'short', day: 'numeric' })

/** What the plan does with the pump setpoint in an hour, against the flat setpoint it is compared with. */
export function modeOf(planned: number | null, baseline: number | null): Mode | null {
  if (planned === null || baseline === null) return null
  if (planned > baseline + 0.05) return 'preheat'
  if (planned < baseline - 0.05) return 'cut'
  return 'hold'
}

/** Whether a day has anything to draw: a price, a plan or a measurement. */
export function hasData(view: DayView | null): boolean {
  if (view === null) return false
  return view.hours.some(
    (h) => h.price_sek_per_kwh !== null || h.rts_planned !== null || h.rts_measured !== null || h.room_measured !== null,
  )
}

/** One row per hour of the given days, in order; days without data are left out. */
export function timelineRows(views: (DayView | null)[]): HourRow[] {
  const rows: HourRow[] = []
  for (const view of views) {
    if (!hasData(view)) continue
    view!.hours.forEach((h, i) => {
      rows.push({
        key: localKey(h.start),
        hour: localKey(h.start).slice(11, 16),
        dayStart: i === 0,
        dayLabel: dayLabelOf(h.start),
        current: h.start === view!.now,
        past: h.past,
        price: h.price_sek_per_kwh,
        floor: h.floor_c,
        extreme: h.extreme,
        rtsMeasured: h.rts_measured,
        rtsPlanned: h.rts_planned,
        rtsBaseline: h.rts_baseline,
        roomMeasured: h.room_measured,
        roomPlanned: h.room_planned,
        roomBaseline: h.room_baseline,
        energyMeasured: h.energy_measured_kwh,
        energyPlanned: h.energy_planned_kwh,
        energyBaseline: h.energy_baseline_kwh,
        savingPlanned: h.saving_planned_sek,
        savingMeasured: h.saving_measured_sek,
        flowPlanned: h.flow_planned_c,
        flowBaseline: h.flow_baseline_c,
        reason: h.reason,
        mode: modeOf(h.rts_planned, h.rts_baseline),
        loops: h.loops_open,
        thermostats: h.thermostats,
      })
    })
  }
  return rows
}

export interface QuarterRow {
  key: string
  price: number | null
  rtsMeasured: number | null
  rtsPlanned: number | null
  rtsBaseline: number | null
  current: boolean
}

/** The price chart at the resolution of the market (15 minutes); an hour's setpoint holds for its quarters. */
export function quarterRows(views: (DayView | null)[]): QuarterRow[] {
  const out: QuarterRow[] = []
  for (const view of views) {
    if (!hasData(view)) continue
    for (const h of view!.hours) {
      h.price_quarters.forEach((price, q) =>
        out.push({
          key: localKey(h.start, q * 15),
          price,
          rtsMeasured: h.rts_measured,
          rtsPlanned: h.rts_planned,
          rtsBaseline: h.rts_baseline,
          current: h.start === view!.now && q === 0,
        }),
      )
    }
  }
  return out
}

/** Every `every`-th key starting from the first, for the x axis. */
export function axisTicks(keys: string[], every: number): string[] {
  return keys.filter((_, i) => i % every === 0)
}

/** 'HH:00' on the axis; at midnight the day's name stands in for the time. */
export function axisLabel(key: string, dayLabels: Record<string, string>): string {
  const time = key.slice(11, 16)
  return time === '00:00' ? (dayLabels[key.slice(0, 10)] ?? time) : time
}

/** For the tooltip header: the hour or quarter a point stands for. */
export function rangeLabel(key: string, minutes = 60): string {
  const start = new Date(key.replace(' ', 'T'))
  const end = new Date(start.getTime() + minutes * 60_000)
  const clock = (d: Date) => d.toLocaleTimeString('sv-SE', { hour: '2-digit', minute: '2-digit' })
  return `${key.slice(0, 10)}  ${clock(start)}-${clock(end)}`
}

/** What the days shown add up to: the plan's saving against the flat setpoint and, for hours that have passed, the measured one. */
export function savingsTotals(rows: HourRow[]): { planned: number; measured: number | null; hours: number } {
  const planned = rows.reduce((sum, r) => sum + (r.savingPlanned ?? 0), 0)
  const measuredRows = rows.filter((r) => r.savingMeasured !== null)
  const measured = measuredRows.length === 0 ? null : measuredRows.reduce((sum, r) => sum + (r.savingMeasured ?? 0), 0)
  return { planned, measured, hours: measuredRows.length }
}

export const MODE_LABEL: Record<Mode, string> = {
  preheat: 'Pre-heat',
  cut: 'Cut',
  hold: 'Hold',
}
