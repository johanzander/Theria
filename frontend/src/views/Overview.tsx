import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { useState } from 'react'
import { fetchDay, fetchExecution, fetchPlan, fetchSavings } from '../api'
import { dayRows, quarterRows, outlookText, tooltipFormatter, UNITS, savingsSummary, type DayView, type Execution, type PlanResponse, type SavingsResponse } from '../model'
import { Card, useLoaded } from '../ui'

const tomorrowIso = () => {
  const d = new Date(Date.now() + 24 * 3600 * 1000)
  return d.toLocaleDateString('sv-SE')
}

function Status({ plan, execution }: { plan: PlanResponse | null; execution: Execution | null }) {
  const saving = plan ? savingsSummary(plan) : null
  return (
    <div className="cards">
      <Card label="Mode" value={execution?.mode ?? plan?.mode ?? '-'} note={plan ? `plan ${Math.round(plan.age_seconds / 60)} min old` : 'no plan yet'} />
      <Card
        label="Setpoint now (plan)"
        value={plan ? `${plan.plan.hours[0].rts_c.toFixed(1)} °C` : '-'}
        note={plan ? `reason: ${plan.plan.hours[0].reason.binding ?? ''}` : undefined}
      />
      <Card
        label="Plan vs flat setpoint"
        value={saving ? `${saving.savingSek.toFixed(1)} SEK` : '-'}
        note={saving ? `${saving.percent.toFixed(1)} % of the model cost` : undefined}
        tone={saving && saving.savingSek >= 0 ? 'good' : undefined}
      />
      {plan?.comfort_outlook && (
        <Card
          label="Room when comfort starts"
          value={outlookText(plan.comfort_outlook).value}
          note={outlookText(plan.comfort_outlook).note}
          tone={plan.comfort_outlook.deficit_c > 0.3 ? 'bad' : undefined}
        />
      )}
      <Card
        label="Last write"
        value={execution?.last_execution?.action ?? '-'}
        note={execution?.last_trouble ?? execution?.last_execution?.reason}
        tone={execution?.last_trouble ? 'bad' : undefined}
      />
    </div>
  )
}

function DayCharts({ view }: { view: DayView }) {
  const rows = dayRows(view)
  const quarters = quarterRows(view)
  const now = rows.find((r) => r.current)?.label
  const temps = rows
    .flatMap((r) => [r.measuredRoom, r.plannedRoom, r.baselineRoom, r.floor, ...Object.values(r.thermostats).flatMap((x) => [x.room_c, x.set_c])])
    .filter((v): v is number => v !== null)
  const thermostats = Object.keys(rows.find((r) => Object.keys(r.thermostats).length > 0)?.thermostats ?? {})
  const roomRows = rows.map((r) => ({
    ...r,
    ...Object.fromEntries(thermostats.flatMap((n) => [[`${n}_room`, r.thermostats[n]?.room_c ?? null], [`${n}_set`, r.thermostats[n]?.set_c ?? null]])),
  }))
  const roomColours = ['#0b6bcb', '#b42318', '#2f855a', '#b7791f']
  const roomLow = Math.floor(Math.min(...temps) * 2) / 2 - 0.5
  const roomHigh = Math.ceil(Math.max(...temps) * 2) / 2 + 0.5
  // with several y axes the line must name the one it is drawn against
  const rooms = Object.keys(view.loops_total)
  const loopRows = rows.map((r) => ({ ...r, ...Object.fromEntries(rooms.map((room) => [room, r.loops[room] ?? null])) }))
  const loopColours = ['#0b6bcb', '#b42318', '#2f855a', '#b7791f']
  const marker = (yAxisId?: string) =>
    now ? <ReferenceLine x={now} yAxisId={yAxisId} stroke="#7b8794" strokeDasharray="2 2" label="now" /> : null
  return (
    <>
      {rows.every((r) => r.price === null) && <p className="note">No prices for this day yet (tomorrow's are published in the afternoon).</p>}
      <div className="panel">
        <h2>Price and room setpoint</h2>
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={quarters}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" interval={7} />
            <YAxis yAxisId="price" orientation="right" unit=" kr" />
            <YAxis yAxisId="rts" domain={[16, 20]} unit="°" />
            <Tooltip formatter={tooltipFormatter((n) => (n.includes('Price') ? UNITS.price : UNITS.temp))} />
            <Legend />
            <Bar yAxisId="price" dataKey="price" name="Price per quarter (SEK/kWh)" fill="#cbd2d9" />
            <Line yAxisId="rts" type="stepAfter" dataKey="measuredRts" name="Measured setpoint" stroke="#102a43" strokeWidth={2} dot={false} />
            <Line yAxisId="rts" type="stepAfter" dataKey="plannedRts" name="Plan setpoint" stroke="#0b6bcb" strokeWidth={2} dot={false} />
            <Line yAxisId="rts" type="stepAfter" dataKey="baselineRts" name="Flat setpoint" stroke="#9aa5b1" strokeDasharray="4 3" dot={false} />
            {marker('rts')}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="panel">
        <h2>Room temperature</h2>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={roomRows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" interval={2} />
            <YAxis domain={[roomLow, roomHigh]} unit="°" />
            <Tooltip formatter={tooltipFormatter(() => UNITS.temp)} />
            <Legend />
            <Line type="stepAfter" dataKey="floor" name="Floor" stroke="#b42318" strokeDasharray="4 3" dot={false} />
            {thermostats.map((n, i) => (
              <Line key={`${n}_room`} type="monotone" dataKey={`${n}_room`} name={`${n} now`} stroke={roomColours[i % roomColours.length]} strokeWidth={1.5} dot={false} />
            ))}
            {thermostats.map((n, i) => (
              <Line key={`${n}_set`} type="stepAfter" dataKey={`${n}_set`} name={`${n} setpoint`} stroke={roomColours[i % roomColours.length]} strokeDasharray="2 3" dot={false} />
            ))}
            <Line type="monotone" dataKey="measuredRoom" name="Measured (mean)" stroke="#102a43" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="plannedRoom" name="Expected with plan" stroke="#0b6bcb" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="baselineRoom" name="Expected, flat setpoint" stroke="#9aa5b1" strokeDasharray="4 3" dot={false} />
            {marker()}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {rooms.length > 0 && (
        <div className="panel">
          <h2>Open floor loops</h2>
          <ResponsiveContainer width="100%" height={200}>
            <ComposedChart data={loopRows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" interval={2} />
              <YAxis domain={[0, Math.max(...Object.values(view.loops_total))]} allowDecimals={false} />
              <Tooltip formatter={tooltipFormatter(() => UNITS.count)} />
              <Legend />
              {rooms.map((room, i) => (
                <Line key={room} type="stepAfter" dataKey={room} name={`${room} (of ${view.loops_total[room]})`} stroke={loopColours[i % loopColours.length]} strokeWidth={2} dot={false} />
              ))}
              {marker()}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  )
}

function DayPicker({ day, onChange }: { day: 'today' | 'tomorrow'; onChange: (d: 'today' | 'tomorrow') => void }) {
  return (
    <div className="buttons">
      <button onClick={() => onChange('today')} disabled={day === 'today'}>Today</button>
      <button onClick={() => onChange('tomorrow')} disabled={day === 'tomorrow'}>Tomorrow</button>
    </div>
  )
}

function SavingsChart({ savings }: { savings: SavingsResponse }) {
  return (
    <div className="panel">
      <h2>
        Measured saving per day ({savings.total_saving_sek.toFixed(0)} SEK of{' '}
        {savings.total_baseline_cost_sek.toFixed(0)} SEK)
      </h2>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={savings.days}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis unit=" kr" />
          <Tooltip formatter={tooltipFormatter((n) => (n.includes('kWh') ? UNITS.kwh : UNITS.sek))} />
          <Legend />
          <Bar dataKey="saving_sek" name="Saving (SEK)" fill="#067647" />
          <Line type="monotone" dataKey="aux_kwh" name="Electric heater (kWh)" stroke="#b42318" dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export function Overview() {
  const plan = useLoaded(fetchPlan)
  const execution = useLoaded(fetchExecution)
  const savings = useLoaded(() => fetchSavings(14))
  const [day, setDay] = useState<'today' | 'tomorrow'>('today')
  const dayView = useLoaded(() => fetchDay(day === 'today' ? undefined : tomorrowIso()))
  const pick = (next: 'today' | 'tomorrow') => {
    setDay(next)
    dayView.reload()
  }
  const error = plan.error ?? execution.error ?? savings.error ?? dayView.error
  return (
    <>
      {error && <p className="bad">{error}</p>}
      {execution.data?.plan_error && <p className="bad">The last plan could not be computed: {execution.data.plan_error}</p>}
      <Status plan={plan.data} execution={execution.data} />
      {plan.data ? <DayPicker day={day} onChange={pick} /> : null}
      {dayView.data ? <DayCharts view={dayView.data} /> : <p>No plan computed yet.</p>}
      {savings.data && savings.data.days.length > 0 && <SavingsChart savings={savings.data} />}
    </>
  )
}
