import { fetchModelHealth } from '../api'
import { Card, useLoaded } from '../ui'
import { fitQuality } from '../model'

export function Models() {
  const { data, error } = useLoaded(fetchModelHealth)
  if (error) return <p className="bad">{error}</p>
  if (!data) return <p>Loading...</p>
  const { pump, room, fitted_at, last_fit_error } = data.models
  const response = room.fit?.response
  return (
    <>
      <div className="cards">
        <Card label="Pump model" value={pump.source} note={fitted_at ? `fitted ${fitted_at.slice(0, 16).replace('T', ' ')}` : undefined} />
        <Card label="Room model" value={room.source} note={room.fit_error ?? undefined} tone={room.fit_error ? 'bad' : undefined} />
        <Card label="Last fit error" value={last_fit_error ?? 'none'} tone={last_fit_error ? 'bad' : 'good'} />
      </div>
      {pump.fit && (
        <div className="panel">
          <h2>Heat pump</h2>
          <p>
            Flow temperature = {pump.params.flow_intercept.toFixed(1)} {pump.params.flow_slope.toFixed(2)} × outdoor, plus{' '}
            {pump.params.flow_per_rts} °C per °C of room setpoint (error {pump.fit.flow_rmse_c.toFixed(2)} °C over {pump.fit.flow_rows} hours).
          </p>
          <p>
            Electricity: {pump.params.energy_per_rts.toFixed(2)} kWh/h more per °C of setpoint (error {pump.fit.energy_rmse_kwh.toFixed(2)} kWh over {pump.fit.energy_rows} hours).
          </p>
        </div>
      )}
      {response && (
        <div className="panel">
          <h2>Room response</h2>
          <p>
            One degree more flow temperature raises the average room temperature about {response.gain_c_per_c.toFixed(2)} °C, settling with a time
            constant of {response.time_constant_days.toFixed(1)} days ({response.windows} windows).
          </p>
          <table>
            <thead><tr><th>Season</th><th>Windows</th><th>Gain</th></tr></thead>
            <tbody>
              {response.seasons.map((s) => (
                <tr key={s.season}><td>{s.season}</td><td>{s.windows}</td><td>{s.gain.toFixed(2)}</td></tr>
              ))}
            </tbody>
          </table>
          <h2 style={{ marginTop: 12 }}>Does it predict?</h2>
          <table>
            <thead><tr><th>Days ahead</th><th>Model error</th><th>Assuming no change</th><th></th></tr></thead>
            <tbody>
              {Object.keys(response.cv_rmse_c).map((d) => (
                <tr key={d}>
                  <td>{d}</td>
                  <td>{response.cv_rmse_c[d].toFixed(2)} °C</td>
                  <td>{response.cv_persistence_rmse_c[d].toFixed(2)} °C</td>
                  <td>{fitQuality(response.cv_rmse_c[d], response.cv_persistence_rmse_c[d])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">The room model gives a sensitivity, not a forecast: it does not beat "no change" at predicting the room.</p>
        </div>
      )}
      <div className="panel">
        <h2>Live prediction error (shadow plans vs measured room)</h2>
        {data.scores.length === 0 ? (
          <p className="note">No scores yet: they appear once plans from earlier hours can be compared with measurements.</p>
        ) : (
          <table>
            <thead><tr><th>Hours ahead</th><th>Samples</th><th>Mean error</th><th>Bias</th></tr></thead>
            <tbody>
              {data.scores.map((s) => (
                <tr key={s.lead_h}><td>{s.lead_h}</td><td>{s.n}</td><td>{s.mae_c.toFixed(2)} °C</td><td>{s.bias_c.toFixed(2)} °C</td></tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}
