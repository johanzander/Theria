/** The time range and the drawing arithmetic of the Värmepump tab. */

export const HOUR = 3_600_000
const MINUTE = 60_000
export const MAX_RANGE = 366 * 24 * HOUR // what the add-on answers

export type Column = (number | null)[]

export interface Timeline {
  start_ms: number
  stop_ms: number
  step_ms: number
  times: number[]
  series: Record<string, Column>
  /** `times` has one more entry than each column: the last one closes the last bar. */
  energy: { times: number[]; series: Record<string, Column> }
  prices: { times: number[]; values: Column }
}

/** What Home Assistant reads now; a sensor that is unavailable or shows garbage has null. */
export interface PumpNow {
  values: Record<string, number | null>
  hot_water: boolean
}

export type Range = { kind: 'last'; hours: number } | { kind: 'fixed'; from: number; to: number }
export interface Span { from: number; to: number }

export function resolveRange(range: Range, now: number): Span {
  return range.kind === 'last' ? { from: now - range.hours * HOUR, to: now } : { from: range.from, to: range.to }
}

/** Half a range back (-1) or forward (1), never past now. */
export function pan({ from, to }: Span, direction: -1 | 1, now: number): Span {
  const length = to - from
  const end = Math.min(now, to + (direction * length) / 2)
  return { from: end - length, to: end }
}

/** Twice as long around the same middle, never past now and never longer than the add-on answers. */
export function zoomOut({ from, to }: Span, now: number): Span {
  const half = (to - from) / 2
  const end = Math.min(now, to + half)
  return { from: Math.max(end - MAX_RANGE, from - half), to: end }
}

/** The unbroken stretches of a series as [time, value] points. A stepped one holds until the next reading. */
export function runs(times: number[], values: Column, stepped: boolean): [number, number][][] {
  const out: [number, number][][] = []
  let run: [number, number][] | null = null
  times.forEach((t, i) => {
    const value = values[i]
    if (value === null) {
      run = null
      return
    }
    if (run === null) {
      run = []
      out.push(run)
    } else if (stepped) {
      run.push([t, run[run.length - 1][1]])
    }
    run.push([t, value])
  })
  return out
}

export function yRange(columns: Column[]): [number, number] {
  const known = columns.flat().filter((v): v is number => v !== null)
  if (known.length === 0) return [0, 1]
  const low = Math.min(...known)
  const high = Math.max(...known)
  const margin = (high - low) * 0.08 || 1
  return [low - margin, high + margin]
}

export function niceTicks(low: number, high: number, count: number): number[] {
  const raw = (high - low) / count
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const leading = raw / magnitude
  const step = (leading < 1.5 ? 1 : leading < 3 ? 2 : leading < 7 ? 5 : 10) * magnitude
  const out: number[] = []
  for (let v = Math.ceil(low / step) * step; v <= high + step * 1e-9; v += step) out.push(Number(v.toFixed(6)))
  return out
}

const TICK_MINUTES = [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440, 2880, 10080, 20160, 43200, 86400]
const MIN_TICK_PX = 70

/** Round local times, far enough apart for their labels; local arithmetic, so a clock change stays on the hour. */
export function timeTicks(from: number, to: number, widthPx: number): number[] {
  const pxPerMinute = widthPx / ((to - from) / MINUTE)
  const minutes = TICK_MINUTES.find((m) => m * pxPerMinute >= MIN_TICK_PX) ?? TICK_MINUTES[TICK_MINUTES.length - 1]
  const moment = new Date(from)
  moment.setHours(0, 0, 0, 0)
  const out: number[] = []
  while (moment.getTime() <= to) {
    if (moment.getTime() >= from) out.push(moment.getTime())
    moment.setMinutes(moment.getMinutes() + minutes)
  }
  return out
}

export function timeLabel(t: number): string {
  const d = new Date(t)
  const clock = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  return clock === '00:00' ? `${d.getDate()}/${d.getMonth() + 1}` : clock
}

/** What the series held at a moment: the value of the last bucket that had started. */
export function valueAt(times: number[], values: Column, t: number): number | null {
  let found: number | null = null
  for (let i = 0; i < times.length && times[i] <= t; i++) found = values[i]
  return found
}

/** A time for an input of type datetime-local, in the browser's zone. */
export function toLocalInput(t: number): string {
  return new Date(t - new Date(t).getTimezoneOffset() * MINUTE).toISOString().slice(0, 16)
}
