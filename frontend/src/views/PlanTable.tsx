import { Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fetchPlan } from '../api'
import { describeReason, energyRows, type PlanResponse } from '../model'
import { useLoaded } from '../ui'

function PlanCharts({ plan }: { plan: PlanResponse }) {
  const rows = energyRows(plan)
  return (
    <>
      <div className="panel">
        <h2>Flow temperature</h2>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" interval={2} />
            <YAxis unit="°" domain={[(min: number) => Math.floor(min) - 2, (max: number) => Math.ceil(max) + 2]} />
            <Tooltip />
            <Legend />
            <ReferenceLine y={45} stroke="#b42318" strokeDasharray="4 3" label="max 45" />
            <Line type="stepAfter" dataKey="flow" name="Plan" stroke="#0b6bcb" strokeWidth={2} dot={false} />
            <Line type="stepAfter" dataKey="baselineFlow" name="Flat setpoint" stroke="#9aa5b1" strokeDasharray="4 3" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="panel">
        <h2>Energy per hour</h2>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" interval={2} />
            <YAxis unit=" kWh" />
            <Tooltip />
            <Legend />
            <Bar dataKey="baselineEnergy" name="Flat setpoint" fill="#cbd2d9" />
            <Bar dataKey="energy" name="Plan" fill="#0b6bcb" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="panel">
        <h2>Electricity cost, running total</h2>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" interval={2} />
            <YAxis unit=" kr" />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="cumulativeBaselineCost" name="Flat setpoint" stroke="#9aa5b1" strokeDasharray="4 3" dot={false} />
            <Line type="monotone" dataKey="cumulativeCost" name="Plan" stroke="#067647" strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </>
  )
}

export function PlanTable() {
  const { data: plan, error } = useLoaded(fetchPlan)
  if (error) return <p className="bad">{error}</p>
  if (!plan) return <p>No plan computed yet.</p>
  const { hours, baseline_hours } = plan.plan
  return (
    <>
    <PlanCharts plan={plan} />
    <div className="panel">
      <h2>Hour by hour</h2>
      <p className="note">
        Computed {Math.round(plan.age_seconds / 60)} min ago. Cost is the model's estimate of electricity for the hour.
        {plan.plan.floor_violation_hours > 0 && (
          <span className="bad"> {plan.plan.floor_violation_hours} h below the comfort floor.</span>
        )}
      </p>
      <table>
        <thead>
          <tr>
            <th>Hour</th><th>Price</th><th>Setpoint</th><th>Flat</th><th>Flow target</th>
            <th>Energy</th><th>Cost</th><th>Room end</th><th>Why</th>
          </tr>
        </thead>
        <tbody>
          {hours.map((h, i) => (
            <tr key={h.start} className={h.extreme ? 'extreme' : undefined}>
              <td>{h.start.slice(5, 16).replace('T', ' ')}</td>
              <td>{h.price_sek_per_kwh.toFixed(2)}{h.extreme ? ' !' : ''}</td>
              <td><b>{h.rts_c.toFixed(1)}</b></td>
              <td>{baseline_hours[i].rts_c.toFixed(1)}</td>
              <td>{h.flow_target_c.toFixed(1)} °C</td>
              <td>{h.energy_kwh.toFixed(2)} kWh</td>
              <td>{h.energy_cost_sek.toFixed(2)}</td>
              <td>{h.room_c_end.toFixed(1)} °C</td>
              <td>{describeReason(h.reason.binding ?? '')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    </>
  )
}
