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
  salon_measured: number | null
  salon_planned: number | null
  need_ground_kw: number | null
  need_salon_kw: number | null
  delivered_ground_kw: number | null
  flow_measured_c: number | null
  flow_target_c: number | null
  thermostat: string | null
  thermostat_target_c: number | null
  loops_expected: number | null
  /** per section id: the rooms' temperatures, the thermostats' setpoints and their heat demand (%), measured */
  sections: Record<string, SectionHour>
  /** the whole house: its loss, what the pump should give for it, and what the pump's heat counter shows */
  need_house_kw: number | null
  house_pump_kw: number | null
  heat_measured_kw: number | null
}

/** The whole house's loss fitted to the pump's heat counter. */
export interface HouseLoss {
  w_per_k: number
  free_kw: number
  days: number
  rmse_kw: number
}

export interface SectionHour {
  rooms: Record<string, number | null>
  setpoints: Record<string, number | null>
  demand: Record<string, number | null>
}

/** A part of the house as the dashboard shows it: a floor, the salon, the guest house. */
export interface Section {
  id: string
  name: string
  rooms: string[]
  thermostats: string[]
}

export interface DayView {
  day: string
  now: string
  loops_total: Record<string, number>
  sections: Section[]
  house: HouseLoss | null
  /** per section id: rooms and thermostats Home Assistant has lost now, with their last report (ISO) */
  offline: Record<string, Record<string, string | null>>
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
  salonMeasured: number | null
  salonPlanned: number | null
  needGround: number | null
  needSalon: number | null
  deliveredGround: number | null
  flowMeasured: number | null
  flowTarget: number | null
  thermostat: string | null
  thermostatTarget: number | null
  loopsExpected: number | null
  sections: Record<string, SectionHour>
  needHouse: number | null
  housePump: number | null
  heatMeasured: number | null
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

const ROOM_COLOURS: Record<string, string> = { bibliotek: '#2f855a', vardagsrum: '#b7791f' }

/** A room keeps its colour on every chart, whichever way its name is written. */
export function roomColour(name: string): string {
  return ROOM_COLOURS[name.replace(/^lk_/, '')] ?? '#0e7490'
}

export interface ShiftWindow {
  kind: 'sell' | 'buy'
  x1: string
  x2: string
}

function windowKind(r: { reason: string | null; thermostat: string | null }): ShiftWindow['kind'] | null {
  if (r.reason === 'shift_sell' || r.thermostat === 'closing') return 'sell'
  if (r.reason === 'shift_buy' || r.thermostat === 'opening') return 'buy'
  return null
}

/** The runs of sell and buy hours, to shade on every chart; each ends where the next hour starts. */
export function shiftWindows(rows: { key: string; reason: string | null; thermostat: string | null }[]): ShiftWindow[] {
  const out: ShiftWindow[] = []
  rows.forEach((r, i) => {
    const kind = windowKind(r)
    if (kind === null) return
    const last = out[out.length - 1]
    const next = rows[i + 1]?.key ?? r.key
    if (last && last.kind === kind && last.x2 === r.key) last.x2 = next
    else out.push({ kind, x1: r.key, x2: next })
  })
  return out
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
        salonMeasured: h.salon_measured,
        salonPlanned: h.salon_planned,
        needGround: h.need_ground_kw,
        needSalon: h.need_salon_kw,
        deliveredGround: h.delivered_ground_kw,
        flowMeasured: h.flow_measured_c,
        flowTarget: h.flow_target_c,
        thermostat: h.thermostat,
        thermostatTarget: h.thermostat_target_c,
        loopsExpected: h.loops_expected,
        sections: h.sections,
        needHouse: h.need_house_kw,
        housePump: h.house_pump_kw,
        heatMeasured: h.heat_measured_kw,
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
  past: boolean
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
          past: h.past,
        }),
      )
    }
  }
  return out
}

/** The keys that fall on an even hour (00:00, 02:00, ...), for the x axis. */
export function axisTicks(keys: string[]): string[] {
  return keys.filter((k) => k.slice(14, 16) === '00' && Number(k.slice(11, 13)) % 2 === 0)
}

/** The hour as 'HH' on the axis (00, 01, 02), like BESS Manager; the day lines mark the midnights. */
export function axisLabel(key: string): string {
  return key.slice(11, 13)
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

/**
 * History and prediction in separate series: for each key the value stays under `key` for hours that
 * have passed and goes to `${key}Predicted` for the rest, so that a chart can draw the two differently.
 * The last hour of the history is in both, so the lines meet.
 */
export function splitPredicted<T extends { past: boolean }>(rows: T[], keys: (keyof T & string)[]): (T & Record<string, unknown>)[] {
  const firstFuture = rows.findIndex((r) => !r.past)
  return rows.map((row, i) => {
    const out: Record<string, unknown> = { ...row }
    for (const key of keys) {
      out[key] = firstFuture === -1 || i < firstFuture ? row[key] : null
      out[`${key}Predicted`] = firstFuture !== -1 && i >= firstFuture - 1 ? row[key] : null
    }
    return out as T & Record<string, unknown>
  })
}

/** The first row that is not history, for shading the prediction. */
export function firstPredicted(rows: { key: string; past: boolean }[]): string | undefined {
  return rows.find((r) => !r.past)?.key
}

// Clear of the plan's own colours (dark, blue, red) and of the targets' grey and of the two LK rooms' colours.
const SECTION_PALETTE = ['#0e7490', '#be185d', '#7c3aed', '#a16207', '#65a30d', '#0f766e', '#9d174d']

/** A room's colour within its section; a thermostat with the room's name gets the same colour. The LK rooms keep theirs. */
export function sectionColour(section: Section, name: string): string {
  const known = section.id === 'ground' ? ROOM_COLOURS[name.toLowerCase()] : undefined
  if (known !== undefined) return known
  const names = [...section.rooms, ...section.thermostats.filter((t) => !section.rooms.includes(t))]
    .filter((n) => section.id !== 'ground' || ROOM_COLOURS[n.toLowerCase()] === undefined)
  return SECTION_PALETTE[Math.max(0, names.indexOf(name)) % SECTION_PALETTE.length]
}

/** A section's measured values per hour, flattened for a chart: room_<name>, set_<name>, demand_<name>. */
export function sectionRows<T extends { sections: Record<string, SectionHour> }>(rows: T[], section: Section): (T & Record<string, number | null>)[] {
  return rows.map((r) => {
    const m = r.sections[section.id]
    return {
      ...r,
      ...Object.fromEntries(section.rooms.map((n) => [`room_${n}`, m?.rooms[n] ?? null])),
      ...Object.fromEntries(section.thermostats.map((n) => [`set_${n}`, m?.setpoints[n] ?? null])),
      ...Object.fromEntries(section.thermostats.map((n) => [`demand_${n}`, m?.demand[n] ?? null])),
    } as T & Record<string, number | null>
  })
}

/** An LK room's target: its thermostat's measured setpoint, and for coming hours the plan's LK target (both thermostats get it). */
export function lkTarget(row: { past: boolean; thermostats: DayHour['thermostats']; thermostatTarget: number | null }, room: string): number | null {
  return row.thermostats[`lk_${room.toLowerCase()}`]?.set_c ?? (row.past ? null : row.thermostatTarget)
}

/** A room's name, with the note that it is offline and since its last report: "Väntsidan (offline, last 19:03)". */
export function offlineName(name: string, lastReport: string | null | undefined): string {
  if (lastReport === undefined) return name
  if (lastReport === null) return `${name} (offline)`
  const at = new Date(lastReport)
  const today = new Date().toLocaleDateString('sv-SE') === at.toLocaleDateString('sv-SE')
  const clock = at.toLocaleTimeString('sv-SE', { hour: '2-digit', minute: '2-digit' })
  return `${name} (offline, last ${today ? clock : `${at.toLocaleDateString('sv-SE')} ${clock}`})`
}

/**
 * A room's expected temperature from now on: where it is now plus the change the plan expects for its
 * zone, so that the prediction starts where the room is and has no jump. Null before now, and where
 * the room or the plan is unknown.
 */
export function continueFromNow<T extends { current: boolean }>(rows: T[], actual: (r: T) => number | null, zone: (r: T) => number | null): (number | null)[] {
  const now = rows.findIndex((r) => r.current)
  const start = now === -1 ? null : actual(rows[now])
  const base = now === -1 ? null : zone(rows[now])
  return rows.map((r, i) => {
    const planned = zone(r)
    if (i < now || start === null || base === null || planned === null) return null
    return start + (planned - base)
  })
}
