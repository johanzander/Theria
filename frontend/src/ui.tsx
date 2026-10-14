import { useEffect, useState, type ReactNode } from 'react'

const REFRESH_MS = 60_000

export function useLoaded<T>(
  load: () => Promise<T | null>,
  deps: unknown[] = [],
): { data: T | null; error: string | null; reload: () => void } {
  const [state, setState] = useState<{ data: T | null; error: string | null }>({ data: null, error: null })
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const run = () =>
      load().then(
        (data) => setState({ data, error: null }),
        (e: Error) => setState((s) => ({ ...s, error: e.message })),
      )
    run()
    const id = setInterval(run, REFRESH_MS)
    return () => clearInterval(id)
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
