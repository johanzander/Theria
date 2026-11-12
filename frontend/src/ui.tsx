import { useEffect, useState, type ReactNode } from 'react'

const REFRESH_MS = 60_000

export function useLoaded<T>(
  load: () => Promise<T | null>,
  deps: unknown[] = [],
): { data: T | null; error: string | null; loaded: boolean; reload: () => void } {
  // loaded: an answer has come; data stays null when the add-on had none (404)
  const [state, setState] = useState<{ data: T | null; error: string | null; loaded: boolean }>({
    data: null,
    error: null,
    loaded: false,
  })
  const [tick, setTick] = useState(0)
  useEffect(() => {
    let current = true // an answer to an earlier request must not replace the newer one
    const run = () =>
      load().then(
        (data) => {
          if (current) setState({ data, error: null, loaded: true })
        },
        (e: Error) => {
          if (current) setState((s) => ({ ...s, error: e.message, loaded: true }))
        },
      )
    run()
    const id = setInterval(run, REFRESH_MS)
    return () => {
      current = false
      clearInterval(id)
    }
  }, [tick, ...deps])
  return { ...state, reload: () => setTick((n) => n + 1) }
}

export function Card({ label, value, note, tone }: { label: string; value: string; note?: string; tone?: string }) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className={`value ${tone ?? ''}`}>{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  )
}

/** A titled block with a line saying what it shows. */
export function Panel({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <div className="panel">
      <h2>{title}</h2>
      <p className="note">{description}</p>
      {children}
    </div>
  )
}

/** The introduction at the top of a page. */
export function PageIntro({ children }: { children: ReactNode }) {
  return <p className="intro">{children}</p>
}

export const shiftDay = (day: string, by: number) => {
  const d = new Date(`${day}T12:00:00`)
  d.setDate(d.getDate() + by)
  return d.toLocaleDateString('sv-SE')
}

export const todayIso = () => new Date().toLocaleDateString('sv-SE')

/** The day a page shows: the picked one, or today when none was (null). */
export const shownDay = (picked: string | null, today: string) => picked ?? today

/** Picking today keeps no day, so the page moves on to the next day at midnight. */
export const keptPick = (day: string, today: string) => (day === today ? null : day)

/** Today's date, which changes at midnight while the page stays open. */
function useToday(): string {
  const [today, setToday] = useState(todayIso())
  useEffect(() => {
    const id = setInterval(() => setToday(todayIso()), REFRESH_MS)
    return () => clearInterval(id)
  }, [])
  return today
}

/** The day a page shows and how to pick another; showing today follows it past midnight. */
export function useDay(): [string, (day: string) => void] {
  const today = useToday()
  const [picked, setPicked] = useState<string | null>(null)
  return [shownDay(picked, today), (day) => setPicked(keptPick(day, today))]
}

/** Step through the days: the chosen day and the next one (when it has data) are shown. */
export function DayNav({ day, onChange }: { day: string; onChange: (day: string) => void }) {
  const today = todayIso()
  const title = new Date(`${day}T12:00:00`).toLocaleDateString('sv-SE', { weekday: 'long', day: 'numeric', month: 'long' })
  return (
    <div className="daynav">
      <button onClick={() => onChange(shiftDay(day, -1))} aria-label="Previous day">◀</button>
      <span className="daytitle">{title}</span>
      <button onClick={() => onChange(shiftDay(day, 1))} disabled={day >= today} aria-label="Next day">▶</button>
      {day !== today && <button onClick={() => onChange(today)}>Today</button>}
    </div>
  )
}
