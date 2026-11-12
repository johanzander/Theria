import { Line, ReferenceLine, YAxis } from 'recharts'
import { fetchDay, fetchPlan, fetchSettings } from '../api'
import { dayStatus, describeReason, splitPredicted, timelineRows, UNITS, MODE_LABEL, type HourRow } from '../model'
import { DayNav, PageIntro, Panel, shiftDay, useDay, useLoaded } from '../ui'
import { axisFor, Chart, guides, predictedArea } from './DayCharts'

const fix = (v: number | null, digits: number, suffix = '') => (v === null ? '-' : `${v.toFixed(digits)}${suffix}`)

/** As on the dashboard: a line at midnight and at now, the coming hours shaded, the plan solid where applied and dashed where expected. */
function FlowChart({ rows, cap }: { rows: HourRow[]; cap: number | null }) {
  const now = rows.find((r) => r.current)?.key
  const data = splitPredicted(rows, ['flowPlanned'])
  return (
    <Chart rows={data} axisRows={rows} height={240} unitFor={() => UNITS.temp}>
      <YAxis unit="°" domain={[(min: number) => Math.floor(min) - 2, (max: number) => Math.ceil(max) + 2]} />
      {predictedArea(rows)}
      {cap !== null && <ReferenceLine y={cap} stroke="#d9822b" strokeDasharray="6 3" label={{ value: 'floor limit', position: 'insideTopLeft' }} />}
      <Line type="stepAfter" dataKey="flowPlanned" name="Plan (applied)" stroke="#0b6bcb" strokeWidth={2} dot={false} />
      <Line type="stepAfter" dataKey="flowPlannedPredicted" name="Plan, expected" stroke="#0b6bcb" strokeWidth={2} strokeDasharray="6 3" dot={false} />
      <Line type="stepAfter" dataKey="flowBaseline" name="Flat setpoint" stroke="#9aa5b1" strokeDasharray="4 3" dot={false} />
      {guides(now, axisFor(rows).dayStarts)}
    </Chart>
  )
}

export function PlanTable() {
  const [day, setDay] = useDay()
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
        <p>{dayStatus(first.loaded, first.data !== null, 'Nothing to show for this day yet.')}</p>
      ) : (
        <>
          <Panel
            title="Flow temperature"
            description="The flow temperature the pump asks for: the heat curve shifted by the setpoint, for the day shown and the next; the grey line is midnight, the dotted line now, and the shaded part the coming hours. Blue: the plan, solid where it was applied and dashed where it is expected. Dashed grey: the flat setpoint. Orange: what the floor mixing valves let through to the floor loops; above it, a lower setpoint costs the floor nothing and the radiators are the only ones who notice."
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
                  <tr key={r.key} style={r.current ? { outline: '2px solid #1f2933' } : undefined}>
                    <td>{r.key.slice(5)}</td>
                    <td className="num">{fix(r.price, 2)}</td>
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
