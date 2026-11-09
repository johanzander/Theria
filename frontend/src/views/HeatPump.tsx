import { useEffect, useRef, useState } from 'react'
import { fetchTimeline } from '../api'
import {
  HOUR, MAX_RANGE, niceTicks, pan, resolveRange, runs, timeLabel, timeTicks, toLocalInput, valueAt, yRange, zoomOut,
  type Column, type Range, type Span, type Timeline,
} from '../heatpump'
import { useLoaded } from '../ui'

const G = {
  green: '#73bf69', yellow: '#fade2a', blue: '#5794f2', orange: '#ff9830', purple: '#b877d9', pale: '#c0d8ff',
  red: '#f2495c', lightOrange: '#ffcb7d', teal: '#8ab8ff',
}

interface Line { key: string; name: string; colour: string; stepped?: boolean; area?: number; band?: boolean }
interface Bar { key: string; name: string; colour: string; group: 0 | 1 }
interface LinePanel { kind: 'lines'; id: string; title: string; height: number; unit: string; min?: number; max?: number; lines: Line[]; prices?: boolean }
interface BarPanel { kind: 'bars'; id: string; title: string; height: number; unit: string; bars: Bar[] }
type ChartPanel = LinePanel | BarPanel

const PANELS: ChartPanel[] = [
  { kind: 'lines', id: 'pump', title: 'Värmepump', height: 170, unit: ' %', min: 0, max: 100, lines: [
    { key: 'valve', name: 'Varmvatten', colour: G.pale, band: true },
    { key: 'comp', name: 'Kompressorhastighet', colour: G.green, stepped: true, area: 0.18 },
    { key: 'aux', name: 'Tillskott, elpatron', colour: G.yellow, stepped: true, area: 0.25 },
  ] },
  { kind: 'lines', id: 'heat', title: 'Värme', height: 200, unit: ' °C', lines: [
    { key: 'outdoor', name: 'Utomhustemperatur', colour: G.purple },
    { key: 'flowset', name: 'Börvärde', colour: G.yellow, stepped: true },
    { key: 'rts', name: 'Rumsbörvärde', colour: G.blue, stepped: true },
    { key: 'flow', name: 'Framledningstemperatur', colour: G.orange, area: 0.1 },
  ] },
  { kind: 'lines', id: 'dhw', title: 'Varmvatten', height: 170, unit: ' °C', lines: [
    { key: 'dhwset', name: 'Börvärde varmvatten', colour: G.green, stepped: true },
    { key: 'dhwtop', name: 'Varmvatten topp', colour: G.yellow },
    { key: 'dhwmid', name: 'Varmvatten mitt', colour: G.blue },
  ] },
  { kind: 'lines', id: 'circ', title: 'Cirkulationspump', height: 130, unit: ' %', min: 0, max: 100, lines: [
    { key: 'circ', name: 'Cirkulationspump', colour: G.teal, stepped: true },
  ] },
  { kind: 'bars', id: 'energy', title: 'Energi: avgiven och förbrukad', height: 190, unit: ' kWh', bars: [
    { group: 0, key: 'supp_heat', name: 'Avgiven, värme', colour: G.green },
    { group: 0, key: 'supp_hw', name: 'Avgiven, varmvatten', colour: G.pale },
    { group: 1, key: 'cons_heat', name: 'Förbrukad, värme', colour: G.orange },
    { group: 1, key: 'cons_hw', name: 'Förbrukad, varmvatten', colour: G.lightOrange },
    { group: 1, key: 'cons_aux', name: 'Förbrukad, elpatron', colour: G.red },
  ] },
  { kind: 'lines', id: 'price', title: 'Elpris', height: 150, unit: ' kr', prices: true, lines: [
    { key: 'price', name: 'Nord Pool SE4, spot', colour: G.teal, stepped: true, area: 0.12 },
  ] },
  { kind: 'lines', id: 'circuit', title: 'Köldkrets', height: 170, unit: ' °C', lines: [
    { key: 'hotgas', name: 'Hetgas kompressor', colour: G.red },
    { key: 'cond', name: 'Kondensor ut', colour: G.orange },
    { key: 'evap', name: 'Förångare in', colour: G.blue },
  ] },
  { kind: 'lines', id: 'carrier', title: 'Värmebärare: fram, retur och delta', height: 170, unit: ' °C', lines: [
    { key: 'hc_fwd', name: 'Värmebärare fram', colour: G.orange },
    { key: 'hc_ret', name: 'Värmebärare retur', colour: G.blue },
    { key: 'hc_delta', name: 'Delta (fram - retur)', colour: G.green },
  ] },
  { kind: 'lines', id: 'power', title: 'Effekt', height: 140, unit: ' kW', min: 0, lines: [
    { key: 'power', name: 'Effekt, pumpens egen mätning', colour: G.orange, area: 0.15 },
  ] },
]

const PAD = { left: 8, right: 58, top: 6, bottom: 22 }
const MIN_ZOOM = 5 * 60_000
const PRESETS = [6, 12, 24, 48, 168, 720]
const PRESET_LABEL: Record<number, string> = {
  6: 'Senaste 6 timmar', 12: 'Senaste 12 timmar', 24: 'Senaste 24 timmar', 48: 'Senaste 2 dygn', 168: 'Senaste 7 dygn', 720: 'Senaste 30 dygn',
}

interface Pointer {
  hover: { panel: ChartPanel; t: number; x: number; y: number } | null
  drag: { from: number; to: number } | null
}

const column = (data: Timeline, panel: LinePanel, line: Line): { times: number[]; values: Column } =>
  panel.prices ? { times: data.prices.times, values: data.prices.values } : { times: data.times, values: data.series[line.key] }

function barTotals(data: Timeline, panel: BarPanel, hidden: Set<string>): Column {
  const shown = panel.bars.filter((b) => !hidden.has(b.key))
  return data.energy.times.slice(1).flatMap((_, i) =>
    ([0, 1] as const).map((group) => shown.filter((b) => b.group === group).reduce((sum, b) => sum + (data.energy.series[b.key][i] ?? 0), 0)),
  )
}

function axisRange(data: Timeline, panel: ChartPanel, hidden: Set<string>): [number, number] {
  if (panel.kind === 'bars') return [0, Math.max(1, ...barTotals(data, panel, hidden).map((v) => v ?? 0)) * 1.1]
  const [low, high] = yRange(panel.lines.filter((l) => !hidden.has(l.key)).map((l) => column(data, panel, l).values))
  return [panel.min ?? low, panel.max ?? high]
}

function drawPanel(canvas: HTMLCanvasElement, panel: ChartPanel, data: Timeline, hidden: Set<string>, pointer: Pointer) {
  const ratio = window.devicePixelRatio || 1
  const cssWidth = canvas.clientWidth
  canvas.width = cssWidth * ratio
  canvas.height = panel.height * ratio
  canvas.style.height = `${panel.height}px`
  const c = canvas.getContext('2d') as CanvasRenderingContext2D
  c.scale(ratio, ratio)
  const width = cssWidth - PAD.left - PAD.right
  const floor = panel.height - PAD.bottom
  const [low, high] = axisRange(data, panel, hidden)
  const X = (t: number) => PAD.left + ((t - data.start_ms) / (data.stop_ms - data.start_ms)) * width
  const Y = (v: number) => PAD.top + ((high - v) / (high - low)) * (floor - PAD.top)

  c.font = '12px system-ui, sans-serif'
  c.lineWidth = 1
  c.fillStyle = '#ccccdc'
  c.strokeStyle = '#262a2e'
  for (const v of niceTicks(low, high, 4)) {
    const y = Math.round(Y(v)) + 0.5
    c.beginPath()
    c.moveTo(PAD.left, y)
    c.lineTo(PAD.left + width, y)
    c.stroke()
    c.fillText(v + panel.unit, PAD.left + width + 6, y + 4)
  }
  c.textAlign = 'center'
  for (const t of timeTicks(data.start_ms, data.stop_ms, width)) {
    const label = timeLabel(t)
    const x = Math.round(X(t)) + 0.5
    c.strokeStyle = label.includes('/') ? '#454b50' : '#262a2e'
    c.beginPath()
    c.moveTo(x, PAD.top)
    c.lineTo(x, floor)
    c.stroke()
    c.fillText(label, x, panel.height - 6)
  }
  c.textAlign = 'left'

  c.save()
  c.beginPath()
  c.rect(PAD.left, 0, width, floor)
  c.clip()
  if (panel.kind === 'bars') drawBars(c, panel, data, hidden, X, Y)
  else drawLines(c, panel, data, hidden, X, Y, low, floor)
  if (pointer.drag) {
    const a = X(pointer.drag.from)
    const b = X(pointer.drag.to)
    c.fillStyle = 'rgba(120,160,255,.15)'
    c.fillRect(Math.min(a, b), PAD.top, Math.abs(b - a), floor - PAD.top)
  }
  if (pointer.hover) {
    const x = X(pointer.hover.t)
    c.strokeStyle = '#9aa0a8'
    c.setLineDash([3, 3])
    c.beginPath()
    c.moveTo(x, PAD.top)
    c.lineTo(x, floor)
    c.stroke()
    c.setLineDash([])
  }
  c.restore()
}

type Scale = (value: number) => number

function drawBars(c: CanvasRenderingContext2D, panel: BarPanel, data: Timeline, hidden: Set<string>, X: Scale, Y: Scale) {
  const shown = panel.bars.filter((b) => !hidden.has(b.key))
  data.energy.times.slice(0, -1).forEach((t, i) => {
    const slot = X(data.energy.times[i + 1]) - X(t)
    const barWidth = Math.max(1, slot * 0.4)
    for (const group of [0, 1] as const) {
      let base = 0
      const x = X(t) + slot * 0.08 + group * (barWidth + slot * 0.02)
      for (const bar of shown.filter((b) => b.group === group)) {
        const value = data.energy.series[bar.key][i]
        if (!value) continue
        c.fillStyle = bar.colour
        c.fillRect(x, Y(base + value), barWidth, Y(base) - Y(base + value))
        base += value
      }
    }
  })
}

function drawLines(c: CanvasRenderingContext2D, panel: LinePanel, data: Timeline, hidden: Set<string>, X: Scale, Y: Scale, low: number, floor: number) {
  for (const line of panel.lines.filter((l) => !hidden.has(l.key))) {
    const { times, values } = column(data, panel, line)
    if (line.band) {
      c.fillStyle = `${line.colour}40`
      times.forEach((t, i) => {
        if (!values[i]) return
        const x = X(t)
        c.fillRect(x, PAD.top, X(times[i + 1] ?? t + data.step_ms) - x + 0.5, floor - PAD.top)
      })
      continue
    }
    for (const run of runs(times, values, line.stepped ?? false)) {
      if (line.area) {
        c.fillStyle = line.colour + Math.round(line.area * 255).toString(16).padStart(2, '0')
        c.beginPath()
        c.moveTo(X(run[0][0]), Y(low))
        run.forEach(([t, v]) => c.lineTo(X(t), Y(v)))
        c.lineTo(X(run[run.length - 1][0]), Y(low))
        c.fill()
      }
      c.strokeStyle = line.colour
      c.lineWidth = 1.25
      c.beginPath()
      run.forEach(([t, v], i) => (i ? c.lineTo(X(t), Y(v)) : c.moveTo(X(t), Y(v))))
      c.stroke()
    }
  }
}

const shownValue = (value: number | null, unit: string) =>
  value === null ? '-' : `${value.toFixed(Math.abs(value) < 10 && unit !== ' %' ? 2 : 1)}${unit}`

function tooltipRows(panel: ChartPanel, data: Timeline, hidden: Set<string>, t: number): { name: string; colour: string; text: string }[] {
  if (panel.kind === 'bars') {
    return panel.bars.filter((b) => !hidden.has(b.key)).map((b) => ({
      name: b.name, colour: b.colour, text: shownValue(valueAt(data.energy.times, data.energy.series[b.key], t), panel.unit),
    }))
  }
  return panel.lines.filter((l) => !hidden.has(l.key)).map((l) => {
    const { times, values } = column(data, panel, l)
    const value = valueAt(times, values, t)
    return { name: l.name, colour: l.colour, text: l.band ? (value ? 'på' : 'av') : shownValue(value, panel.unit) }
  })
}

function Charts({ data, onZoom }: { data: Timeline; onZoom: (span: Span) => void }) {
  const canvases = useRef(new Map<string, HTMLCanvasElement>())
  const tip = useRef<HTMLDivElement>(null)
  const pointer = useRef<Pointer>({ hover: null, drag: null })
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  const redraw = () => {
    for (const panel of PANELS) {
      const canvas = canvases.current.get(panel.id)
      if (canvas) drawPanel(canvas, panel, data, hidden, pointer.current)
    }
    showTip()
  }

  const showTip = () => {
    const el = tip.current
    if (!el) return
    const { hover, drag } = pointer.current
    if (!hover || drag) {
      el.style.display = 'none'
      return
    }
    const when = new Date(hover.t).toLocaleString('sv-SE', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
    el.replaceChildren(Object.assign(document.createElement('b'), { textContent: when }))
    for (const row of tooltipRows(hover.panel, data, hidden, hover.t)) {
      const line = document.createElement('div')
      const name = document.createElement('em')
      const swatch = document.createElement('i')
      swatch.style.background = row.colour
      name.append(swatch, row.name)
      line.append(name, Object.assign(document.createElement('span'), { textContent: row.text }))
      el.append(line)
    }
    el.style.display = 'block'
    el.style.left = `${Math.min(window.innerWidth - el.offsetWidth - 10, hover.x + 14)}px`
    el.style.top = `${Math.min(window.innerHeight - el.offsetHeight - 10, hover.y + 14)}px`
  }

  useEffect(() => {
    const finishDrag = () => {
      const drag = pointer.current.drag
      if (!drag) return
      pointer.current.drag = null
      const from = Math.min(drag.from, drag.to)
      const to = Math.max(drag.from, drag.to)
      if (to - from >= MIN_ZOOM) onZoom({ from, to })
      else redraw()
    }
    redraw()
    window.addEventListener('resize', redraw)
    window.addEventListener('mouseup', finishDrag)
    return () => {
      window.removeEventListener('resize', redraw)
      window.removeEventListener('mouseup', finishDrag)
    }
  }, [data, hidden])

  const timeAt = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const share = (e.clientX - box.left - PAD.left) / (box.width - PAD.left - PAD.right)
    return data.start_ms + Math.min(1, Math.max(0, share)) * (data.stop_ms - data.start_ms)
  }

  const toggle = (key: string) =>
    setHidden((old) => {
      const next = new Set(old)
      if (!next.delete(key)) next.add(key)
      return next
    })

  const daily = data.energy.times.length > 1 && data.energy.times[1] - data.energy.times[0] > 2 * HOUR

  return (
    <>
      {PANELS.map((panel) => (
        <div className="hp-panel" key={panel.id}>
          <h2>{panel.kind === 'bars' ? `${panel.title}, per ${daily ? 'dygn' : 'timme'}` : panel.title}</h2>
          <canvas
            ref={(el) => {
              if (el) canvases.current.set(panel.id, el)
            }}
            onMouseDown={(e) => {
              const t = timeAt(e)
              pointer.current.drag = { from: t, to: t }
            }}
            onMouseMove={(e) => {
              const t = timeAt(e)
              pointer.current.hover = { panel, t, x: e.clientX, y: e.clientY }
              if (pointer.current.drag) pointer.current.drag.to = t
              redraw()
            }}
            onMouseLeave={() => {
              pointer.current.hover = null
              redraw()
            }}
          />
          <div className="hp-legend">
            {(panel.kind === 'bars' ? panel.bars : panel.lines).map((s) => (
              <span key={s.key} className={hidden.has(s.key) ? 'off' : undefined} onClick={() => toggle(s.key)}>
                <i style={{ background: s.colour }} />
                {s.name}
              </span>
            ))}
          </div>
        </div>
      ))}
      <div className="hp-tip" ref={tip} />
    </>
  )
}

export function HeatPump() {
  const [range, setRange] = useState<Range>({ kind: 'last', hours: 24 })
  const { data, error } = useLoaded(() => {
    const { from, to } = resolveRange(range, Date.now())
    return fetchTimeline(from, to)
  }, [range])
  const now = Date.now()
  const span = resolveRange(range, now)

  const setFixed = ({ from, to }: Span) => setRange({ kind: 'fixed', from, to })
  const setEnd = (which: keyof Span, text: string) => {
    const t = new Date(text).getTime()
    const next = { ...span, [which]: t }
    if (Number.isFinite(t) && next.from < next.to && next.to - next.from <= MAX_RANGE && next.from < now) setFixed(next)
  }

  return (
    <div className="hp">
      <div className="hp-top">
        <h2>Värmepump</h2>
        <button aria-label="Bakåt" onClick={() => setFixed(pan(span, -1, now))}>«</button>
        <select
          aria-label="Tidsintervall"
          value={range.kind === 'last' ? range.hours : ''}
          onChange={(e) => {
            if (e.target.value) setRange({ kind: 'last', hours: Number(e.target.value) })
          }}
        >
          <option value="">Eget intervall</option>
          {PRESETS.map((h) => <option key={h} value={h}>{PRESET_LABEL[h]}</option>)}
        </select>
        <input aria-label="Från" type="datetime-local" value={toLocalInput(span.from)} onChange={(e) => setEnd('from', e.target.value)} />
        <input aria-label="Till" type="datetime-local" value={toLocalInput(span.to)} onChange={(e) => setEnd('to', e.target.value)} />
        <button aria-label="Framåt" onClick={() => setFixed(pan(span, 1, now))}>»</button>
        <button onClick={() => setFixed(zoomOut(span, now))}>Zooma ut</button>
      </div>
      <p className="hp-hint">Dra i en graf för att zooma, alla grafer följer med. Klicka på en förklaring för att dölja den serien.</p>
      {error && <p className="bad">{error}</p>}
      {data ? <Charts data={data} onZoom={setFixed} /> : !error && <p>Läser historiken...</p>}
    </div>
  )
}
