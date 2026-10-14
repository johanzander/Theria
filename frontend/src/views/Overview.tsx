import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useState } from 'react'
import { fetchDay, fetchExecution, fetchPlan, fetchSavings, fetchSettings } from '../api'
import {
  outlookText, savingsSummary, savingsTotals, timelineRows, tooltipFormatter, UNITS,
  type Execution, type PlanResponse, type SavingsResponse, type SettingEntry,
} from '../model'
import { Card, DayNav, PageIntro, Panel, shiftDay, todayIso, useLoaded } from '../ui'
import { DayCharts } from './DayCharts'

function Status({ plan, execution, totals, roomNow }: {
  plan: PlanResponse | null
  execution: Execution | null
  totals: ReturnType<typeof savingsTotals>
  roomNow: number | null
}) {
  const saving = plan ? savingsSummary(plan) : null
  return (
    <div className="cards">
      <Card label="Mode" value={execution?.mode ?? plan?.mode ?? '-'} note={plan ? `plan ${Math.round(plan.age_seconds / 60)} min old` : 'no plan yet'} />
      <Card
        label="Setpoint now (plan)"
        value={plan ? `${plan.plan.hours[0].rts_c.toFixed(1)} °C` : '-'}
        note={plan ? `reason: ${plan.plan.hours[0].reason.binding ?? ''}` : undefined}
      />
      <Card label="Room now" value={roomNow === null ? '-' : `${roomNow.toFixed(1)} °C`} note="measured mean of the ground-floor rooms" />
      <Card
        label="Saving, days shown"
        value={`${totals.planned.toFixed(1)} kr`}
        note={totals.measured === null ? 'from the plan' : `plan; ${totals.measured.toFixed(1)} kr measured over ${totals.hours} h`}
        tone={totals.planned >= 0 ? 'good' : 'bad'}
      />
      <Card
        label="Plan vs flat, coming hours"
        value={saving ? `${saving.savingSek.toFixed(1)} kr` : '-'}
        note={saving ? `${saving.percent.toFixed(1)} % of the model cost, comfort and payback included` : undefined}
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

function SavingsChart({ savings }: { savings: SavingsResponse }) {
  return (
    <Panel
      title={`Measured saving per day (${savings.total_saving_sek.toFixed(0)} kr of ${savings.total_baseline_cost_sek.toFixed(0)} kr)`}
      description="Per day over the last two weeks: the measured heating cost against what the pump would have cost with the flat setpoint (green bars, kr). The red line is the electric heater's energy, which should stay near zero."
    >
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={savings.days}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis unit=" kr" />
          <Tooltip formatter={tooltipFormatter((n) => (n.includes('kWh') ? UNITS.kwh : UNITS.sek))} />
          <Legend />
          <Bar dataKey="saving_sek" name="Saving (kr)" fill="#067647" />
          <Line type="monotone" dataKey="aux_kwh" name="Electric heater (kWh)" stroke="#b42318" dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </Panel>
  )
}

function limitsOf(settings: SettingEntry[] | null) {
  const value = (key: string) => {
    const v = settings?.find((s) => s.key === key)?.value
    return typeof v === 'number' ? v : null
  }
  return { ceiling: value('ceiling_c'), target: value('target_c') }
}

export function Overview() {
  const [day, setDay] = useState(todayIso())
  const plan = useLoaded(fetchPlan)
  const execution = useLoaded(fetchExecution)
  const savings = useLoaded(() => fetchSavings(14))
  const settings = useLoaded(fetchSettings)
  const first = useLoaded(() => fetchDay(day), [day])
  const second = useLoaded(() => fetchDay(shiftDay(day, 1)), [day])
  const views = [first.data, second.data]
  const rows = timelineRows(views)
  const roomNow = [...rows].reverse().find((r) => r.roomMeasured !== null)?.roomMeasured ?? null
  const error = plan.error ?? execution.error ?? savings.error ?? first.error ?? second.error
  return (
    <>
      <PageIntro>
        The plan moves heating from expensive to cheap hours by raising or lowering the heat pump's setpoint, using the floor's thermal mass
        as a store, and never lets the room under its comfort floor. This page shows whole days: the day you choose and the next one when
        it has prices or a plan. Every chart puts the plan, the flat setpoint it is compared with, and what really happened in the same picture.
      </PageIntro>
      {error && <p className="bad">{error}</p>}
      {execution.data?.plan_error && <p className="bad">The last plan could not be computed: {execution.data.plan_error}</p>}
      <Status plan={plan.data} execution={execution.data} totals={savingsTotals(rows)} roomNow={roomNow} />
      <DayNav day={day} onChange={setDay} />
      {rows.length > 0 ? (
        <DayCharts rows={rows} views={views} loopsTotal={first.data?.loops_total ?? {}} limits={limitsOf(settings.data)} />
      ) : (
        <p>{first.data ? 'Nothing to show for this day yet.' : 'Loading...'}</p>
      )}
      {savings.data && savings.data.days.length > 0 && <SavingsChart savings={savings.data} />}
    </>
  )
}
