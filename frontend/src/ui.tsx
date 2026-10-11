import { useEffect, useState } from 'react'

const REFRESH_MS = 60_000

export function useLoaded<T>(
  load: () => Promise<T | null>,
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
  }, [tick])
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
