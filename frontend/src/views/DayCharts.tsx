import {
  Bar, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceArea, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  axisLabel, axisTicks, firstPredicted, MODE_LABEL, pairedTooltipRows, quarterRows, rangeLabel, roomColour, lkTarget, offlineName, sectionColour, sectionRows, shiftWindows, splitPredicted,
  tooltipFormatter, UNITS,
  type DayView, type HouseLoss, type HourRow, type Section, type ShiftWindow, type TooltipItem, type Unit,
} from '../model'
import { Panel } from '../ui'

const C = {
  measured: '#102a43', plan: '#0b6bcb', flat: '#9aa5b1', price: '#cbd2d9', floor: '#b42318',
  target: '#067647', setTarget: '#52606d', salon: '#7b3fa0', delivered: '#067647', savingPlan: '#74a9e8', savingMeasured: '#067647',
}

/** The x axis shared by the charts: every even hour as 00, 02, 04, with a line at each midnight. */
export function axisFor(rows: { key: string }[]) {
  const keys = rows.map((r) => r.key)
  return {
    ticks: axisTicks(keys),
    tickFormatter: axisLabel,
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

/** The future is shaded: to its left is what happened, in it what the plan expects. */
export function predictedArea(rows: { key: string; past: boolean }[], yAxisId?: string) {
  const from = firstPredicted(rows)
  if (from === undefined) return []
  return [<ReferenceArea key="predicted" x1={from} x2={rows[rows.length - 1].key} yAxisId={yAxisId} fill="#0b6bcb" fillOpacity={0.06}
    label={{ value: 'predicted', position: 'insideTopRight', fill: '#52606d', fontSize: 12 }} />]
}

/** Sell and buy windows, shaded the same on every chart so that one window can be followed from room to price. */
function windowAreas(windows: ShiftWindow[], yAxisId?: string) {
  return windows.map((w) => (
    <ReferenceArea key={`${w.kind}-${w.x1}`} x1={w.x1} x2={w.x2} yAxisId={yAxisId} fill={w.kind === 'sell' ? '#b42318' : '#d9822b'} fillOpacity={0.1} />
  ))
}

const WINDOW_NOTE = 'Red shading: a sell, the thermostats close the loops and the setpoint is lowered. Orange: a buy just before a sell, the thermostats open the loops.'
const PREDICTED_DASH = '6 3'
const TARGET_DASH = '5 4' // a target (a thermostat's setpoint, the LK target) is stepped and dashed, in the colour of what it is for

/** A thermostat's setpoint, in the colour of its room; it shares the room's legend entry and tooltip row. */
function targetLine(section: Section, name: string) {
  return (
    <Line key={`set-${name}`} type="stepAfter" dataKey={`set_${name}`} name={`${name}, target`} stroke={sectionColour(section, name)}
      strokeWidth={1.75} strokeOpacity={0.85} strokeDasharray={TARGET_DASH} dot={false} legendType="none" />
  )
}

/** A tooltip that shows a target in its actual's row: "Kök  20.1 °C (target 20.0 °C)". */
function PairedTooltip({ active, payload, label, targets, unit, minutes }: {
  active?: boolean; payload?: TooltipItem[]; label?: string | number; targets: Record<string, string>; unit: Unit; minutes: number
}) {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div className="paired-tooltip">
      <p className="paired-tooltip-label">{rangeLabel(String(label), minutes)}</p>
      {pairedTooltipRows(payload, targets, unit).map((r) => (
        <p key={r.name} style={{ color: r.color }}>{r.name} : {r.text}</p>
      ))}
    </div>
  )
}

export function Chart({ rows, height, children, unitFor, minutes = 60, axisRows, targets }: {
  rows: object[]; height: number; children: React.ReactNode; unitFor: (name: string) => Unit; minutes?: number
  axisRows: { key: string }[]
  /** a target's data key to its actual's, to show them on one tooltip row; the chart's unit must then be one */
  targets?: Record<string, string>
}) {
  const axis = axisFor(axisRows)
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="key" ticks={axis.ticks} tickFormatter={axis.tickFormatter} interval={0} />
        {targets
          ? <Tooltip content={<PairedTooltip targets={targets} unit={unitFor('')} minutes={minutes} />} />
          : <Tooltip labelFormatter={(k) => rangeLabel(String(k), minutes)} formatter={tooltipFormatter(unitFor)} />}
        <Legend />
        {children}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

/** The plan's own zones always get a chart, even when the room sections are not configured. */
export function withPlanZones(sections: Section[]): Section[] {
  const out = [...sections]
  if (!out.some((s) => s.id === 'salon')) out.unshift({ id: 'salon', name: 'Salong', rooms: [], thermostats: [] })
  if (!out.some((s) => s.id === 'ground')) out.splice(1, 0, { id: 'ground', name: 'Bottenvåning', rooms: [], thermostats: [] })
  return out
}

function yRange(values: (number | null | undefined)[]): [number, number] {
  const known = values.filter((v): v is number => typeof v === 'number')
  if (known.length === 0) return [18, 22]
  return [Math.floor(Math.min(...known)), Math.ceil(Math.max(...known))]
}

/** One section's rooms, measured, with its thermostats' setpoints; the ground floor and the salon also carry the plan. */
function SectionRoomChart({ section, rows, windows, now, dayStarts, limits, lkNames, offline }: {
  section: Section; rows: HourRow[]; windows: ShiftWindow[]; now: string | undefined; dayStarts: string[]; limits: { target: number | null }
  /** the rooms behind the LK thermostats, lower case */
  lkNames: string[]
  /** rooms Home Assistant has lost now, with their last report */
  offline: Record<string, string | null>
}) {
  const ground = section.id === 'ground'
  const salon = section.id === 'salon'
  // Without configured rooms the ground floor falls back to the LK thermostats' own readings.
  const lkRooms = ground && section.rooms.length === 0
    ? Object.keys(rows.find((r) => Object.keys(r.thermostats).length > 0)?.thermostats ?? {})
    : []
  // The LK rooms have a thermostat each: its own target, in the room's colour, instead of their mean.
  const lkTargeted = ground ? section.rooms.filter((n) => lkNames.includes(n.toLowerCase())) : []
  const targeted = [...section.thermostats, ...lkTargeted]
  const meanTarget = ground && lkTargeted.length === 0
  const data = sectionRows(rows, section).map((r) => ({
      ...r,
      roomExpected: r.past ? null : r.roomPlanned,
      ...Object.fromEntries(lkRooms.map((n) => [`lk_${n}`, r.thermostats[n]?.room_c ?? null])),
      ...Object.fromEntries(lkTargeted.map((n) => [`set_${n}`, lkTarget(r, n)])),
    }) as unknown as HourRow & Record<string, number | null>)
  const planValues = rows.flatMap((r) => ground
    ? [r.roomPlanned, r.floor, r.thermostatTarget]
    : salon ? [r.salonPlanned] : [])
  const [low, high] = yRange([
    ...planValues,
    ...data.flatMap((r) => [...section.rooms.map((n) => r[`room_${n}`]), ...targeted.map((n) => r[`set_${n}`] as number | null)]),
    ...data.flatMap((r) => lkRooms.map((n) => r[`lk_${n}`] as number | null)),
  ])
  return (
    <>
      <h3 className="chart-heading">{section.name}</h3>
      <Chart rows={data} axisRows={rows} height={240} unitFor={() => UNITS.temp} targets={{
        ...Object.fromEntries(targeted.map((n) => [`set_${n}`, `room_${n}`])),
      }}>
        <YAxis domain={[low, high]} tickCount={high - low + 1} allowDecimals={false} unit="°" />
        {predictedArea(rows)}
        {windowAreas(windows)}
        {/* actual: solid curves; each room's target right after it, dashed in its colour */}
        {section.rooms.flatMap((n) => [
          <Line key={n} type="monotone" dataKey={`room_${n}`} name={offlineName(n, offline[n])} stroke={sectionColour(section, n)} strokeWidth={1.25}
            dot={false} />,
          ...(targeted.includes(n) ? [targetLine(section, n)] : []),
        ])}
        {section.thermostats.filter((n) => !section.rooms.includes(n)).map((n) => targetLine(section, n))}
        {lkRooms.map((n) => (
          <Line key={n} type="monotone" dataKey={`lk_${n}`} name={n.replace(/^lk_/, '')} stroke={roomColour(n)} strokeWidth={1} dot={false} />
        ))}
        {/* expected by the plan, coming hours only: dashed blue */}
        {ground && <Line type="monotone" dataKey="roomExpected" name="LK rooms, expected" stroke={C.plan} strokeWidth={2} strokeDasharray={PREDICTED_DASH} dot={false} />}
        {salon && <Line type="monotone" dataKey="salonPlanned" name="Salon, expected" stroke={C.plan} strokeWidth={2} strokeDasharray={PREDICTED_DASH} dot={false} />}
        {/* targets are stepped and dashed; the comfort floor is a limit, dashed red */}
        {meanTarget && <Line type="stepAfter" dataKey="thermostatTarget" name="LK target" stroke={C.measured} strokeWidth={2} strokeDasharray={TARGET_DASH}
          dot={false} connectNulls />}
        {ground && limits.target !== null && <ReferenceLine y={limits.target} stroke={C.setTarget} strokeDasharray={TARGET_DASH} label="comfort target" />}
        {ground && <Line type="stepAfter" dataKey="floor" name="Floor" stroke={C.floor} strokeDasharray="4 3" dot={false} />}
        {offlineMarks(section, data, offline)}
        {guides(now, dayStarts)}
      </Chart>
    </>
  )
}

/** Where an offline room's line ends: a ring at its last value, labelled. */
function offlineMarks(section: Section, data: Record<string, unknown>[], offline: Record<string, string | null>) {
  return section.rooms.filter((n) => n in offline).flatMap((n) => {
    const last = [...data].reverse().find((r) => typeof r[`room_${n}`] === 'number')
    if (last === undefined) return []
    return [<ReferenceDot key={`offline-${n}`} x={String(last.key)} y={last[`room_${n}`] as number} r={5} fill="#fff"
      stroke={sectionColour(section, n)} strokeWidth={2} label={{ value: `${n} offline`, position: 'right', fill: '#52606d', fontSize: 12 }} />]
  })
}

/** Each radiator thermostat's heat demand in a section, in percent. */
function SectionDemandChart({ section, rows, windows, now, dayStarts, offline }: {
  section: Section; rows: HourRow[]; windows: ShiftWindow[]; now: string | undefined; dayStarts: string[]
  offline: Record<string, string | null>
}) {
  return (
    <>
      <h3 className="chart-heading">{section.name}, heat demand</h3>
      <Chart rows={sectionRows(rows, section)} axisRows={rows} height={150} unitFor={() => UNITS.percent}>
        <YAxis domain={[0, 100]} unit=" %" />
        {predictedArea(rows)}
        {windowAreas(windows)}
        {section.thermostats.map((n) => (
          <Line key={n} type="stepAfter" dataKey={`demand_${n}`} name={offlineName(n, offline[n])} stroke={sectionColour(section, n)} strokeWidth={1.5}
            dot={false} />
        ))}
        {guides(now, dayStarts)}
      </Chart>
    </>
  )
}

/** What the heating need chart says about the whole-house line, with the fit's own numbers. */
export function houseNote(house: HouseLoss | null): string {
  if (house === null) return 'The whole house is not shown: its fit to the heat counter is missing or was rejected (see Models).'
  return `Dark: the whole house's loss, ${house.w_per_k.toFixed(0)} W/K times the LK rooms' temperature above the outdoor, fitted to the pump's own heat counter (hot water left out) on ${house.days} whole days below 12 °C. The same fit puts the free heat at ${house.free_kw.toFixed(1)} kW (household electricity, people, sun, the stove), so dashed grey is what the pump should give, the loss less that; grey bars are what its heat counter shows each hour. A day's mean is about ${house.rmse_kw.toFixed(1)} kW off the fit. Between the zones and the whole house are the rooms without a heat demand analysis: the upper floor, the kitchen and large hall, and the guest house if the pump heats it.`
}

export function DayCharts({ rows, views, loopsTotal, sections, house, offline, limits }: {
  rows: HourRow[]; views: (DayView | null)[]; loopsTotal: Record<string, number>; sections: Section[]; house: HouseLoss | null
  offline: DayView['offline']
  limits: { target: number | null }
}) {
  const quarters = splitPredicted(quarterRows(views), ['rtsPlanned'])
  const now = rows.find((r) => r.current)?.key
  const quarterNow = quarters.find((q) => q.current)?.key
  const dayStarts = axisFor(rows).dayStarts
  const quarterDayStarts = axisFor(quarters).dayStarts
  const windows = shiftWindows(rows)
  const energyRows = splitPredicted(rows, ['energyPlanned'])
  const flowRows = splitPredicted(rows, ['flowPlanned'])
  const rooms = Object.keys(loopsTotal)
  const loopRows = rows.map((r) => ({
    ...r,
    expectedOpen: r.loopsExpected === null ? null : 100 * r.loopsExpected,
    ...Object.fromEntries(rooms.map((room) => {
      const open = r.loops[room]
      return [room, open === null || open === undefined ? null : (100 * open) / loopsTotal[room]]
    })),
  }))
  const shown = withPlanZones(sections)
  const withThermostats = shown.filter((s) => s.thermostats.length > 0)
  const priceOrTemp = (name: string) => (name.includes('Price') ? UNITS.price : UNITS.temp)
  return (
    <>
      <Panel
        title="Room temperature"
        description={`The house part by part, as on the Home Assistant climate view; read the page top down: rooms, what they need, what the pump sends, and why. Each room is a solid line, its target a dashed line in the same colour (Bibliotek and Vardagsrum their own LK thermostat's, and for coming hours the target the plan gives both); hover shows them together. Dashed blue is the temperature the plan expects in the coming hours: for the ground floor the mean of the two LK rooms, the LK loops close when a room is above its target; for the salon where it is now plus what the plan changes. Dashed red is the comfort floor the plan holds, the grey dashed line the comfort target. The upper floor and the guest house are only measured: the plan does not move them. ${WINDOW_NOTE}`}
      >
        {shown.map((section) => (
          <SectionRoomChart key={section.id} section={section} rows={rows} windows={windows} now={now} dayStarts={dayStarts} limits={limits}
            lkNames={rooms} offline={offline[section.id] ?? {}} />
        ))}
      </Panel>

      <Panel
        title="Heating need"
        description={`The heat lost at the room and outdoor temperature, in kW, whatever meets it: the pump, heat stored in the floor, the stove or the sun. Blue and purple are the two zones the plan moves, calculated from the heat demand analysis: the rooms behind the LK thermostats (161.5 W/K) and the salon (84.2 W/K, less 250 W of measured other heat); sun and people are not counted for them. Green, dotted: what the ground-floor loops give, from the LK figures (167 W/K of flow above the room, at most 40 °C, through the loops that are open); green under blue means the rooms are living on stored heat, the stove or the sun, green over blue that the floor is storing heat. ${houseNote(house)} ${WINDOW_NOTE}`}
      >
        <Chart rows={rows} axisRows={rows} height={260} unitFor={() => UNITS.kw}>
          <YAxis unit=" kW" width={72} />
          {predictedArea(rows)}
          {windowAreas(windows)}
          <Bar dataKey="heatMeasured" name="Whole house, pump heat measured" fill={C.price} />
          <Line type="stepAfter" dataKey="needHouse" name="Whole house, loss (fitted)" stroke={C.measured} strokeWidth={2.5} dot={false} />
          <Line type="stepAfter" dataKey="housePump" name="Whole house, pump heat expected" stroke={C.flat} strokeWidth={2} strokeDasharray="4 3" dot={false} />
          <Line type="stepAfter" dataKey="needGround" name="LK rooms (analysis)" stroke={C.plan} strokeWidth={2} dot={false} />
          <Line type="stepAfter" dataKey="deliveredGround" name="LK rooms, delivered by the loops" stroke={C.delivered} strokeWidth={2} strokeDasharray="3 2" dot={false} />
          <Line type="stepAfter" dataKey="needSalon" name="Salon (analysis)" stroke={C.salon} strokeWidth={2} dot={false} />
          {guides(now, dayStarts)}
        </Chart>
      </Panel>

      <Panel
        title="Flow temperature and thermostats"
        description={`Top: the flow to the heating. Dark: the pump's own flow target; blue, dashed: the target the plan expects, which continues it. Grey: the measured flow, without hot water and the quarter of an hour after it; it runs above the target while the pump catches up after hot water. The floor gets at most 40 °C. Below it, per zone: on the ground floor the share of each room's floor loops that were open, and dashed what the LK thermostats are expected to do; in the salon (Netatmo) and the guest house (Tado) each radiator thermostat's heat demand, 0 to 100 %, measured. ${WINDOW_NOTE}`}
      >
        <Chart rows={flowRows} axisRows={rows} height={200} unitFor={() => UNITS.temp}>
          <YAxis domain={['auto', 'auto']} unit="°" />
          {predictedArea(rows)}
          {windowAreas(windows)}
          <Line type="linear" dataKey="flowMeasured" name="Flow, measured" stroke={C.flat} strokeWidth={1.5} dot={false} connectNulls />
          <Line type="stepAfter" dataKey="flowTarget" name="Pump's flow target" stroke={C.measured} strokeWidth={2} dot={false} />
          <Line type="stepAfter" dataKey="flowPlanned" name="Flow, expected (applied)" stroke={C.plan} strokeWidth={2} dot={false} />
          <Line type="stepAfter" dataKey="flowPlannedPredicted" name="Flow, expected" stroke={C.plan} strokeWidth={2} strokeDasharray={PREDICTED_DASH} dot={false} />
          {guides(now, dayStarts)}
        </Chart>
        <h3 className="chart-heading">Bottenvåning, LK floor loops</h3>
        <Chart rows={loopRows} axisRows={rows} height={170} unitFor={() => UNITS.percent}>
          <YAxis domain={[0, 100]} unit=" %" />
          {predictedArea(rows)}
          {windowAreas(windows)}
          {rooms.map((room) => (
            <Line key={room} type="stepAfter" dataKey={room} name={`${room}, loops open`} stroke={roomColour(room)} strokeWidth={2} dot={false} />
          ))}
          <Line type="stepAfter" dataKey="expectedOpen" name="Loops open, expected" stroke={C.plan} strokeWidth={2} strokeDasharray={PREDICTED_DASH} dot={false} />
          {guides(now, dayStarts)}
        </Chart>
        {withThermostats.map((section) => (
          <SectionDemandChart key={section.id} section={section} rows={rows} windows={windows} now={now} dayStarts={dayStarts}
            offline={offline[section.id] ?? {}} />
        ))}
      </Panel>

      <Panel
        title="Pump setpoint and price"
        description={`Grey bars: the electricity price per quarter of an hour. Lines: the pump's room setpoint, a heat-curve offset and not a room temperature. Dark: what the pump actually had. Blue: what the plan chose, dashed where it is a prediction that changes with every replan. Dashed grey: the flat setpoint it is compared with. ${WINDOW_NOTE}`}
      >
        <Chart rows={quarters} axisRows={quarters} height={280} unitFor={priceOrTemp} minutes={15}>
          <YAxis yAxisId="price" orientation="right" unit=" kr" />
          <YAxis yAxisId="rts" domain={[16, 20]} unit="°" />
          {predictedArea(quarters, 'rts')}
          {windowAreas(windows, 'rts')}
          <Bar yAxisId="price" dataKey="price" name="Price per quarter" fill={C.price} />
          <Line yAxisId="rts" type="stepAfter" dataKey="rtsMeasured" name="Measured setpoint" stroke={C.measured} strokeWidth={2} dot={false} />
          <Line yAxisId="rts" type="stepAfter" dataKey="rtsPlanned" name="Plan setpoint (applied)" stroke={C.plan} strokeWidth={2} dot={false} />
          <Line yAxisId="rts" type="stepAfter" dataKey="rtsPlannedPredicted" name="Plan setpoint (predicted)" stroke={C.plan} strokeWidth={2} strokeDasharray={PREDICTED_DASH} dot={false} />
          <Line yAxisId="rts" type="stepAfter" dataKey="rtsBaseline" name="Flat setpoint" stroke={C.flat} strokeDasharray="4 3" dot={false} />
          {guides(quarterNow, quarterDayStarts, 'rts')}
        </Chart>
      </Panel>

      <Panel
        title="Heating energy per hour"
        description="The electricity the heat pump uses for heating, per hour. Dark bars: measured (compressor plus electric heater). Blue line: what the plan expected, solid for hours that have passed and dashed in the shaded area, which is the future. Dashed grey: what the flat setpoint would use according to the model. The plan moves energy from dear hours to cheap ones; the area under the lines is roughly equal."
      >
        <Chart rows={energyRows} axisRows={rows} height={240} unitFor={() => UNITS.kwh}>
          <YAxis unit=" kWh" />
          {predictedArea(rows)}
          <Bar dataKey="energyMeasured" name="Measured" fill={C.measured} />
          <Line type="stepAfter" dataKey="energyPlanned" name="Plan (applied)" stroke={C.plan} strokeWidth={2} dot={false} />
          <Line type="stepAfter" dataKey="energyPlannedPredicted" name="Plan (predicted)" stroke={C.plan} strokeWidth={2} strokeDasharray={PREDICTED_DASH} dot={false} />
          <Line type="stepAfter" dataKey="energyBaseline" name="Flat setpoint" stroke={C.flat} strokeDasharray="4 3" dot={false} />
          {guides(now, dayStarts)}
        </Chart>
      </Panel>

      <Panel
        title="Saving per hour against the flat setpoint"
        description="Per hour: what the hour's electricity would have cost with the flat setpoint minus what it cost, at that hour's price. Above zero is saved, below zero cost more (heat bought early is paid back later). Light blue: from the plan's own model. Green: measured electricity against the model's flat estimate, for hours that have passed. Pale bars in the shaded area are predictions."
      >
        <Chart rows={rows} axisRows={rows} height={220} unitFor={() => UNITS.sek}>
          <YAxis unit=" kr" />
          {predictedArea(rows)}
          <ReferenceLine y={0} stroke="#52606d" />
          <Bar dataKey="savingPlanned" name="Plan (model)" fill={C.savingPlan}>
            {rows.map((r) => <Cell key={r.key} fillOpacity={r.past ? 1 : 0.45} />)}
          </Bar>
          <Bar dataKey="savingMeasured" name="Measured" fill={C.savingMeasured} />
          {guides(now, dayStarts)}
        </Chart>
      </Panel>

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
