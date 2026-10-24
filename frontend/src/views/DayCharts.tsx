import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  axisLabel, axisTicks, MODE_LABEL, quarterRows, rangeLabel, tooltipFormatter, UNITS,
  type DayView, type HourRow, type Unit,
} from '../model'
import { Panel } from '../ui'

const C = {
  measured: '#102a43', plan: '#0b6bcb', flat: '#9aa5b1', price: '#cbd2d9', floor: '#b42318', limit: '#d9822b',
  target: '#067647', savingPlan: '#74a9e8', savingMeasured: '#067647',
}
const ROOM_COLOURS = ['#2f855a', '#b7791f', '#7b3fa0', '#c2410c']

/** The x axis shared by the charts: every third hour, and the day named at midnight. */
export function axisFor(rows: { key: string }[], perHour: number) {
  const dayLabels: Record<string, string> = {}
  for (const r of rows as HourRow[]) if (r.dayStart) dayLabels[r.key.slice(0, 10)] = r.dayLabel
  const keys = rows.map((r) => r.key)
  return {
    ticks: axisTicks(keys, 3 * perHour),
    tickFormatter: (key: string) => axisLabel(key, dayLabels),
    dayStarts: keys.filter((k) => k.slice(11, 16) === '00:00'),
  }
}

/** The day boundaries and the current hour; a plain function because Recharts only finds its own elements. */
export function guides(now: string | undefined, dayStarts: string[], yAxisId?: string) {
  return [
    ...dayStarts.map((k) => <ReferenceLine key={`day-${k}`} x={k} yAxisId={yAxisId} stroke="#b8c1cc" />),
    ...(now ? [<ReferenceLine key="now" x={now} yAxisId={yAxisId} stroke="#7b8794" strokeDasharray="2 2" label="now" />] : []),
  ]
}

export function Chart({ rows, height, children, unitFor, minutes = 60, perHour = 1, axisRows }: {
  rows: object[]; height: number; children: React.ReactNode; unitFor: (name: string) => Unit; minutes?: number; perHour?: number
  axisRows: { key: string }[]
}) {
  const axis = axisFor(axisRows, perHour)
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="key" ticks={axis.ticks} tickFormatter={axis.tickFormatter} interval={0} />
        <Tooltip labelFormatter={(k) => rangeLabel(String(k), minutes)} formatter={tooltipFormatter(unitFor)} />
        <Legend />
        {children}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function DayCharts({ rows, views, loopsTotal, limits }: {
  rows: HourRow[]; views: (DayView | null)[]; loopsTotal: Record<string, number>
  limits: { ceiling: number | null; target: number | null }
}) {
  const quarters = quarterRows(views)
  const now = rows.find((r) => r.current)?.key
  const quarterNow = quarters.find((q) => q.current)?.key
  const dayStarts = axisFor(rows, 1).dayStarts
  const quarterDayStarts = axisFor(quarters, 4).dayStarts
  const thermostats = Object.keys(rows.find((r) => Object.keys(r.thermostats).length > 0)?.thermostats ?? {})
  const roomRows = rows.map((r) => ({
    ...r,
    ...Object.fromEntries(thermostats.map((n) => [`room_${n}`, r.thermostats[n]?.room_c ?? null])),
  }))
  const rooms = Object.keys(loopsTotal)
  const loopRows = rows.map((r) => ({ ...r, ...Object.fromEntries(rooms.map((room) => [room, r.loops[room] ?? null])) }))
  const temps = rows.flatMap((r) => [r.roomMeasured, r.roomPlanned, r.roomBaseline, r.floor]).filter((v): v is number => v !== null)
  const low = Math.floor(Math.min(...temps, 18) * 2) / 2 - 0.5
  const high = Math.ceil(Math.max(...temps, limits.ceiling ?? 0) * 2) / 2 + 0.5
  const priceOrTemp = (name: string) => (name.includes('Price') ? UNITS.price : UNITS.temp)
  return (
    <>
      <Panel
        title="Price and pump setpoint"
        description="Grey bars: the electricity price per quarter of an hour. Lines: the pump's room setpoint, which is a heat-curve offset and not a room temperature. Dark: what the pump actually had. Blue: what the plan chose (for hours that have passed, the plan that was applied). Dashed grey: the flat setpoint the plan is compared with. A good plan sits above the dashed line when the price is low and below it when the price is high."
      >
        <Chart rows={quarters} axisRows={quarters} height={280} unitFor={priceOrTemp} minutes={15} perHour={4}>
          <YAxis yAxisId="price" orientation="right" unit=" kr" />
          <YAxis yAxisId="rts" domain={[16, 20]} unit="°" />
          <Bar yAxisId="price" dataKey="price" name="Price per quarter" fill={C.price} />
          <Line yAxisId="rts" type="stepAfter" dataKey="rtsMeasured" name="Measured setpoint" stroke={C.measured} strokeWidth={2} dot={false} />
          <Line yAxisId="rts" type="stepAfter" dataKey="rtsPlanned" name="Plan setpoint" stroke={C.plan} strokeWidth={2} dot={false} />
          <Line yAxisId="rts" type="stepAfter" dataKey="rtsBaseline" name="Flat setpoint" stroke={C.flat} strokeDasharray="4 3" dot={false} />
          {guides(quarterNow, quarterDayStarts, 'rts')}
        </Chart>
      </Panel>

      <Panel
        title="Room temperature"
        description="Dark: the measured mean of the ground-floor rooms. Blue: what the plan expected the room to be, from the plan that was applied. Dashed grey: what the model expects with the flat setpoint. Thin coloured lines: each room. Dashed red: the comfort floor. Orange: the room limit, where the LK valves close. Green: the comfort target. Plan and measurement close together means the model is right."
      >
        <Chart rows={roomRows} axisRows={rows} height={260} unitFor={() => UNITS.temp}>
          <YAxis domain={[low, high]} unit="°" />
          <Line type="stepAfter" dataKey="floor" name="Floor" stroke={C.floor} strokeDasharray="4 3" dot={false} />
          {limits.ceiling !== null && <ReferenceLine y={limits.ceiling} stroke={C.limit} strokeDasharray="6 3" label="limit" />}
          {limits.target !== null && <ReferenceLine y={limits.target} stroke={C.target} strokeDasharray="2 4" label="target" />}
          {thermostats.map((n, i) => (
            <Line key={n} type="monotone" dataKey={`room_${n}`} name={n.replace(/^lk_/, '')} stroke={ROOM_COLOURS[i % ROOM_COLOURS.length]} strokeWidth={1} dot={false} />
          ))}
          <Line type="monotone" dataKey="roomMeasured" name="Measured (mean)" stroke={C.measured} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="roomPlanned" name="Expected with plan" stroke={C.plan} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="roomBaseline" name="Expected, flat setpoint" stroke={C.flat} strokeDasharray="4 3" dot={false} />
          {guides(now, dayStarts)}
        </Chart>
      </Panel>

      <Panel
        title="Heating energy per hour"
        description="The electricity the heat pump uses for heating, per hour. Dark bars: measured (compressor plus electric heater). Blue line: what the plan expected. Dashed grey: what the flat setpoint would use according to the model. The plan moves energy from dear hours to cheap ones; the area under the lines is roughly equal."
      >
        <Chart rows={rows} axisRows={rows} height={240} unitFor={() => UNITS.kwh}>
          <YAxis unit=" kWh" />
          <Bar dataKey="energyMeasured" name="Measured" fill={C.measured} />
          <Line type="stepAfter" dataKey="energyPlanned" name="Plan" stroke={C.plan} strokeWidth={2} dot={false} />
          <Line type="stepAfter" dataKey="energyBaseline" name="Flat setpoint" stroke={C.flat} strokeDasharray="4 3" dot={false} />
          {guides(now, dayStarts)}
        </Chart>
      </Panel>

      <Panel
        title="Saving per hour against the flat setpoint"
        description="Per hour: what the hour's electricity would have cost with the flat setpoint minus what it cost, at that hour's price. Above zero is saved, below zero cost more (heat bought early is paid back later). Light blue: from the plan's own model. Green: measured electricity against the model's flat estimate, for hours that have passed."
      >
        <Chart rows={rows} axisRows={rows} height={220} unitFor={() => UNITS.sek}>
          <YAxis unit=" kr" />
          <ReferenceLine y={0} stroke="#52606d" />
          <Bar dataKey="savingPlanned" name="Plan (model)" fill={C.savingPlan} />
          <Bar dataKey="savingMeasured" name="Measured" fill={C.savingMeasured} />
          {guides(now, dayStarts)}
        </Chart>
      </Panel>

      {rooms.length > 0 && (
        <Panel
          title="Open floor loops"
          description="How many of each room's floor loops had their valve open, per hour. A loop closes when its room reaches the LK thermostat setpoint; zero means the floor is not heating that room."
        >
          <Chart rows={loopRows} axisRows={rows} height={180} unitFor={() => UNITS.count}>
            <YAxis domain={[0, Math.max(...Object.values(loopsTotal))]} allowDecimals={false} />
            {rooms.map((room, i) => (
              <Line key={room} type="stepAfter" dataKey={room} name={`${room} (of ${loopsTotal[room]})`} stroke={ROOM_COLOURS[i % ROOM_COLOURS.length]} strokeWidth={2} dot={false} />
            ))}
            {guides(now, dayStarts)}
          </Chart>
        </Panel>
      )}

      <Panel
        title="Schedule"
        description="What the plan does with the pump setpoint each hour compared with the flat setpoint: orange pre-heats (cheap hours), blue cuts (dear hours), grey holds. The hour you are in has a dark outline."
      >
        <div className="schedule">
          {rows.map((r) => (
            <div key={r.key} className={`slot ${r.mode ?? 'none'}${r.current ? ' current' : ''}`}
              title={`${r.key}  ${r.mode ? MODE_LABEL[r.mode] : 'no plan'}${r.rtsPlanned !== null ? `  ${r.rtsPlanned.toFixed(1)} °C` : ''}`} />
          ))}
        </div>
        <div className="schedule-labels">
          {rows.map((r) => <span key={r.key}>{r.hour.slice(0, 2) === '00' || Number(r.hour.slice(0, 2)) % 6 === 0 ? r.hour.slice(0, 2) : ''}</span>)}
        </div>
        <div className="legend">
          <span><i style={{ background: '#e8903a' }} />Pre-heat</span>
          <span><i style={{ background: '#0b6bcb' }} />Cut</span>
          <span><i style={{ background: '#d5dae0' }} />Hold</span>
        </div>
      </Panel>
    </>
  )
}
