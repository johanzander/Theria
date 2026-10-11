import { fetchPlan } from '../api'
import { describeReason } from '../model'
import { useLoaded } from '../ui'

export function PlanTable() {
  const { data: plan, error } = useLoaded(fetchPlan)
  if (error) return <p className="bad">{error}</p>
  if (!plan) return <p>No plan computed yet.</p>
  const { hours, baseline_hours } = plan.plan
  return (
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
  )
}
