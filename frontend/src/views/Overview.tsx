import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { fetchExecution, fetchPlan, fetchSavings } from '../api'
import { planRows, savingsSummary, type Execution, type PlanResponse, type SavingsResponse } from '../model'
import { Card, useLoaded } from '../ui'

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
      <Card
        label="Stage"
        value={execution ? `± ${execution.stage.max_deviation_c} °C` : '-'}
        note={execution?.stage.revert_until ? `at base until ${execution.stage.revert_until}` : execution?.last_stage_note ?? undefined}
      />
      <Card
        label="Last write"
        value={execution?.last_execution?.action ?? '-'}
        note={execution?.last_trouble ?? execution?.last_execution?.reason}
        tone={execution?.last_trouble ? 'bad' : undefined}
      />
    </div>
  )
}

function PlanChart({ plan }: { plan: PlanResponse }) {
  const rows = planRows(plan)
  const temps = rows.flatMap((r) => [r.room, r.baselineRoom, r.floor])
  const roomLow = Math.floor(Math.min(...temps) * 2) / 2 - 0.5
  const roomHigh = Math.ceil(Math.max(...temps) * 2) / 2 + 0.5
  return (
    <>
      <div className="panel">
        <h2>Price and room setpoint</h2>
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" interval={2} />
            <YAxis yAxisId="price" orientation="right" unit=" kr" />
            <YAxis yAxisId="rts" domain={[16, 20]} unit="°" />
            <Tooltip />
            <Legend />
            <Bar yAxisId="price" dataKey="price" name="Price (SEK/kWh)" fill="#cbd2d9" />
            <Line yAxisId="rts" type="stepAfter" dataKey="rts" name="Plan setpoint" stroke="#0b6bcb" strokeWidth={2} dot={false} />
            <Line yAxisId="rts" type="stepAfter" dataKey="baselineRts" name="Flat setpoint" stroke="#9aa5b1" strokeDasharray="4 3" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="panel">
        <h2>Expected room temperature</h2>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" interval={2} />
            <YAxis domain={[roomLow, roomHigh]} unit="°" />
            <Tooltip />
            <Legend />
            <ReferenceLine y={rows[0].floor} stroke="#b42318" strokeDasharray="4 3" label="floor" />
            <Line type="monotone" dataKey="room" name="With plan" stroke="#0b6bcb" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="baselineRoom" name="Flat setpoint" stroke="#9aa5b1" strokeDasharray="4 3" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </>
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
          <Tooltip />
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
  const error = plan.error ?? execution.error ?? savings.error
  return (
    <>
      {error && <p className="bad">{error}</p>}
      <Status plan={plan.data} execution={execution.data} />
      {plan.data ? <PlanChart plan={plan.data} /> : <p>No plan computed yet.</p>}
      {savings.data && savings.data.days.length > 0 && <SavingsChart savings={savings.data} />}
    </>
  )
}
