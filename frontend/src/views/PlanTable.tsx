import { CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useState } from 'react'
import { fetchDay, fetchPlan, fetchSettings } from '../api'
import { axisLabel, axisTicks, describeReason, rangeLabel, timelineRows, tooltipFormatter, UNITS, MODE_LABEL, type HourRow } from '../model'
import { DayNav, PageIntro, Panel, shiftDay, todayIso, useLoaded } from '../ui'

const fix = (v: number | null, digits: number, suffix = '') => (v === null ? '-' : `${v.toFixed(digits)}${suffix}`)

function FlowChart({ rows, cap }: { rows: HourRow[]; cap: number | null }) {
  const dayLabels: Record<string, string> = {}
  for (const r of rows) if (r.dayStart) dayLabels[r.key.slice(0, 10)] = r.dayLabel
  return (
    <ResponsiveContainer width="100%" height={240}>
      <ComposedChart data={rows}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="key" ticks={axisTicks(rows.map((r) => r.key), 3)} tickFormatter={(k: string) => axisLabel(k, dayLabels)} interval={0} />
        <YAxis unit="°" domain={[(min: number) => Math.floor(min) - 2, (max: number) => Math.ceil(max) + 2]} />
        <Tooltip labelFormatter={(k) => rangeLabel(String(k))} formatter={tooltipFormatter(() => UNITS.temp)} />
        <Legend />
        {cap !== null && <ReferenceLine y={cap} stroke="#d9822b" strokeDasharray="6 3" label="floor limit" />}
        <Line type="stepAfter" dataKey="flowPlanned" name="Plan" stroke="#0b6bcb" strokeWidth={2} dot={false} />
        <Line type="stepAfter" dataKey="flowBaseline" name="Flat setpoint" stroke="#9aa5b1" strokeDasharray="4 3" dot={false} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function PlanTable() {
  const [day, setDay] = useState(todayIso())
  const plan = useLoaded(fetchPlan)
  const settings = useLoaded(fetchSettings)
  const first = useLoaded(() => fetchDay(day), [day])
  const second = useLoaded(() => fetchDay(shiftDay(day, 1)), [day])
  const rows = timelineRows([first.data, second.data])
  const cap = settings.data?.find((s) => s.key === 'floor_mix_max_c')?.value
  const error = plan.error ?? first.error ?? second.error
  return (
    <>
      <PageIntro>
        The plan hour by hour: what it chose, what the flat setpoint would have done, what it expects, and why. For hours that have
        passed the plan is the one that was applied at the time. The plan is recomputed every 15 minutes from the measured room and the latest prices.
      </PageIntro>
      {error && <p className="bad">{error}</p>}
      <DayNav day={day} onChange={setDay} />
      {rows.length === 0 ? (
        <p>{first.data ? 'Nothing to show for this day yet.' : 'Loading...'}</p>
      ) : (
        <>
          <Panel
            title="Flow temperature"
            description="The flow temperature the pump asks for: the heat curve shifted by the setpoint. Blue: the plan. Dashed grey: the flat setpoint. Orange: what the floor mixing valves let through to the floor loops; above it, a lower setpoint costs the floor nothing and the radiators are the only ones who notice."
          >
            <FlowChart rows={rows} cap={typeof cap === 'number' ? cap : null} />
          </Panel>
          <Panel
            title="Hour by hour"
            description={`${plan.data ? `Computed ${Math.round(plan.data.age_seconds / 60)} min ago. ` : ''}Energy is the model's estimate of the heating electricity for the hour; saving is the flat setpoint's energy minus the plan's, at the hour's price.${
              plan.data && plan.data.plan.floor_violation_hours > 0 ? ` ${plan.data.plan.floor_violation_hours} h are expected below the comfort floor.` : ''
            }`}
          >
            <table>
              <thead>
                <tr>
                  <th>Hour</th><th className="num">Price</th><th>Plan</th><th className="num">Setpoint</th><th className="num">Flat</th>
                  <th className="num">Flow</th><th className="num">Room (plan)</th><th className="num">Room (flat)</th>
                  <th className="num">Energy</th><th className="num">Flat energy</th><th className="num">Saving</th><th>Why</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.key} className={r.extreme ? 'extreme' : undefined} style={r.current ? { outline: '2px solid #1f2933' } : undefined}>
                    <td>{r.key.slice(5)}</td>
                    <td className="num">{fix(r.price, 2)}{r.extreme ? ' !' : ''}</td>
                    <td>{r.mode ? MODE_LABEL[r.mode] : '-'}</td>
                    <td className="num"><b>{fix(r.rtsPlanned, 1)}</b></td>
                    <td className="num">{fix(r.rtsBaseline, 1)}</td>
                    <td className="num">{fix(r.flowPlanned, 1, ' °C')}</td>
                    <td className="num">{fix(r.roomPlanned, 1, ' °C')}</td>
                    <td className="num">{fix(r.roomBaseline, 1, ' °C')}</td>
                    <td className="num">{fix(r.energyPlanned, 2, ' kWh')}</td>
                    <td className="num">{fix(r.energyBaseline, 2, ' kWh')}</td>
                    <td className="num">{fix(r.savingPlanned, 2, ' kr')}</td>
                    <td>{r.reason ? describeReason(r.reason) : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </>
      )}
    </>
  )
}
