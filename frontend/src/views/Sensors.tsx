import { fetchComfort } from '../api'
import { formatAge } from '../model'
import { Card, useLoaded } from '../ui'

export function Sensors() {
  const { data, error } = useLoaded(fetchComfort)
  if (error) return <p className="bad">{error}</p>
  if (!data) return <p>Loading...</p>
  return (
    <>
      <div className="cards">
        <Card label="Comfort sensors" value={data.status} tone={data.status === 'ok' ? 'good' : 'bad'} note={`checked ${data.checked_at.slice(11, 16)} UTC`} />
        <Card
          label="Room temperature used"
          value={data.room_temperature_c === null ? '-' : `${data.room_temperature_c.toFixed(2)} °C`}
          note={data.room_temperature_note || undefined}
        />
      </div>
      <div className="panel">
        <h2>Sensors</h2>
        <table>
          <thead>
            <tr><th>Sensor</th><th>Status</th><th>Value</th><th>Last change</th><th>Battery</th><th>Signal</th><th>Notes</th></tr>
          </thead>
          <tbody>
            {data.sensors.map((s) => (
              <tr key={s.temperature_entity}>
                <td>{s.name}{s.reference ? ' (reference)' : ''}</td>
                <td className={s.status === 'ok' ? 'good' : 'bad'}>{s.status}</td>
                <td>{s.value_c === null ? '-' : `${s.value_c.toFixed(1)} °C`}</td>
                <td>{s.age_minutes === null ? '-' : formatAge(s.age_minutes)}</td>
                <td>{s.battery_percent === null ? '-' : `${s.battery_percent} %`}</td>
                <td>{s.signal_dbm === null ? '-' : `${s.signal_dbm} dBm`}</td>
                <td>{s.issues.map((i) => i.message).join('; ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
