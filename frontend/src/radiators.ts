// The salon and guest house radiator plan (shadow mode): types and the small pure helpers of its page.

export const WEEKDAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'] as const
export type Weekday = (typeof WEEKDAYS)[number]

export interface OpeningHours { open: string; close: string }

export interface GuestRoom { entity: string; t_room_c: number; band_c: number | null }

export interface RadiatorPrefs {
  salon_schedule: Record<Weekday, OpeningHours | null>
  salon_open_target_c: number
  salon_open_floor_c: number
  salon_closed_floor_c: number
  salon_preheat_c: number
  salon_margin_c: number
  salon_max_setpoint_c: number
  guest_band_c: number
  guest_rooms: GuestRoom[]
  mode: 'off' | 'shadow'
}

export interface PrefsResponse { prefs: RadiatorPrefs; defaults: RadiatorPrefs; is_default: boolean }

export interface SalonHourView {
  start: string; open: boolean; heat: boolean; mean_end_c: number; setpoint_c: number; price_sek_per_kwh: number; reason: string
}

export interface GuestHourView { start: string; setpoint_c: number; reason: string }

export interface RadiatorSnapshot {
  computed_at: string
  age_seconds: number
  salon: {
    reading: { mean_c: number | null; used: string[]; skipped: Record<string, string>; thermostat_mean_c: number | null; unknown: boolean }
    hours: SalonHourView[]
    saving_sek: number | null
    unknown: boolean
  }
  guest: Record<string, GuestHourView[]>
  would_write: Record<string, number>
  read: Record<string, number | null>
  difference: Record<string, number | null>
  availability: Record<string, boolean>
  availability_share: Record<string, number>
  guest_saving: [number, string]
  outdoor_c: number
  outdoor_stale: boolean
  left_alone: string[]
  last_error: string | null
}

/** A plan older than this many seconds (two cycles of 15 minutes) no longer counts as current. */
export const STALE_AFTER_SECONDS = 1800

/** What to warn about: a failed cycle, an old plan, an outdoor value that is not a reading. */
export function warnings(snap: RadiatorSnapshot): string[] {
  const out: string[] = []
  if (snap.last_error) out.push(`The last cycle failed (${snap.last_error}); this is the previous plan.`)
  else if (snap.age_seconds > STALE_AFTER_SECONDS) out.push(`This plan is ${Math.round(snap.age_seconds / 60)} minutes old.`)
  if (snap.outdoor_stale) out.push('The outdoor sensor could not be read; the last value is used.')
  if (snap.left_alone.length) out.push(`Set to off by hand and left alone: ${snap.left_alone.join(', ')}.`)
  return out
}

export interface ScheduleRow { day: Weekday; open: string | null; close: string | null }

export const scheduleToRows = (prefs: RadiatorPrefs): ScheduleRow[] =>
  WEEKDAYS.map((day) => {
    const hours = prefs.salon_schedule[day]
    return { day, open: hours?.open ?? null, close: hours?.close ?? null }
  })

/** The rows back as a schedule, or the first problem found ('HH:MM' strings compare in order). */
export function rowsToSchedule(rows: ScheduleRow[]): { schedule?: Record<Weekday, OpeningHours | null>; error?: string } {
  const schedule = {} as Record<Weekday, OpeningHours | null>
  for (const row of rows) {
    if (row.open === null && row.close === null) {
      schedule[row.day] = null
    } else if (row.open === null || row.close === null) {
      return { error: `${row.day}: give both an opening and a closing time, or neither` }
    } else if (!(row.open < row.close)) {
      return { error: `${row.day}: opening must be before closing` }
    } else {
      schedule[row.day] = { open: row.open, close: row.close }
    }
  }
  return { schedule }
}

/** What the plan would change: '+1.5', '-0.5', '=' when the same, '-' when the thermostat cannot be read. */
export function formatDifference(current: number | null, wouldBe: number): string {
  if (current === null) return '-'
  const diff = wouldBe - current
  if (Math.abs(diff) < 0.05) return '='
  return `${diff > 0 ? '+' : ''}${diff.toFixed(1)}`
}

/** The mean share of cycles in which the thermostats could be read. */
export function overallAvailability(share: Record<string, number>): number {
  const values = Object.values(share)
  return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length
}
