import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fetchModelCatalog, fetchModelHealth, fetchParameters } from '../api'
import { Card, PageIntro, Panel, useLoaded } from '../ui'
import { fitQuality, formatNumber, formatValue, groupBy, parameterSeries, warmupText, type ModelEntry, type ParameterHistory } from '../model'

const TRACKED: { key: string; title: string; unit: string }[] = [
  { key: 'room_gain_c_per_c', title: 'Room response (°C per °C flow)', unit: '' },
  { key: 'warmup_scale', title: 'Warm-up speed: measured / model', unit: '' },
  { key: 'cooling_scale', title: 'Cooling speed: measured / model', unit: '' },
  { key: 'night_slope_c_per_c', title: 'Night cooling: less per °C warmer', unit: '' },
  { key: 'night_intercept_c', title: 'Night change at 0 °C outdoors', unit: ' °C' },
  { key: 'night_level_c_per_c', title: 'Night: extra per °C warmer room', unit: '' },
  { key: 'aux_per_flow_c', title: 'Electric heater (kWh per °C above 44)', unit: '' },
  { key: 'energy_per_rts', title: 'Energy per °C setpoint (kWh/h)', unit: '' },
  { key: 'disturbance_c_per_h', title: 'Room drift the model missed (disturbance, °C/h)', unit: '' },
  { key: 'sun_swing_c_per_wm2', title: 'Sun: °C the room is lifted per W/m² of sun', unit: '' },
  { key: 'morning_margin_c', title: 'Night margin (learned from mornings)', unit: ' °C' },
]

function Learning({ history }: { history: ParameterHistory }) {
  const events = history.history.filter((r) => r.kind !== 'refit').slice(-15).reverse()
  return (
    <div className="panel">
      <h2>What it has learned over time</h2>
      <p className="note">
        One point per refit (nightly) and per change of the night margin. {history.history.length} records so far.
      </p>
      <div className="mini-charts">
        {TRACKED.map(({ key, title, unit }) => {
          const series = parameterSeries(history.history, key)
          return (
            <div key={key} className="mini">
              <div className="label">{title}</div>
              <div className="value">{history.current[key] === undefined ? '-' : history.current[key].toFixed(3)}</div>
              {series.length > 1 && (
                <ResponsiveContainer width="100%" height={70}>
                  <LineChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="label" hide />
                    <YAxis domain={['auto', 'auto']} hide />
                    <Tooltip formatter={(value: unknown) => formatValue(value, { suffix: unit, digits: 3 })} />
                    <Line type="monotone" dataKey="value" stroke="#0b6bcb" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          )
        })}
      </div>
      {events.length > 0 && (
        <table>
          <thead><tr><th>When (UTC)</th><th>What</th><th>Why</th></tr></thead>
          <tbody>
            {events.map((e, i) => (
              <tr key={i}><td>{e.at.slice(0, 16).replace('T', ' ')}</td><td>{e.kind}</td><td>{e.note}</td></tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}


function ParameterList({ entries }: { entries: ModelEntry[] }) {
  return (
    <Panel
      title="Every parameter and weight"
      description="All the numbers the plan runs on, grouped. Source: fitted = measured from history; prior = a starting value until a fit is accepted; learned = adjusted by the running system; setting = chosen by you (change it on the Settings page); constant = fixed in the code; derived = computed from the others."
    >
      <table>
        <thead>
          <tr><th>Parameter</th><th className="num">Value</th><th>Unit</th><th>Source</th><th>What it is</th></tr>
        </thead>
        <tbody>
          {groupBy(entries).flatMap(([group, items]) => [
            <tr key={group} className="group"><td colSpan={5}>{group}</td></tr>,
            ...items.map((e) => (
              <tr key={e.key}>
                <td>{e.label}</td>
                <td className="num"><b>{formatNumber(e.value)}</b></td>
                <td>{e.unit}</td>
                <td><span className={`badge ${e.source}`}>{e.source}</span></td>
                <td className="desc">{e.description}</td>
              </tr>
            )),
          ])}
        </tbody>
      </table>
    </Panel>
  )
}

export function Models() {
  const { data, error } = useLoaded(fetchModelHealth)
  const learned = useLoaded(fetchParameters)
  const catalog = useLoaded(fetchModelCatalog)
  if (error) return <p className="bad">{error}</p>
  if (!data) return <p>Loading...</p>
  const { pump, room, fitted_at, last_fit_error } = data.models
  const response = room.fit?.response
  return (
    <>
      <PageIntro>
        The model is a small physical one: how the pump turns a setpoint into a flow temperature and electricity, how the floor and the sun
        move the room, and what the plan trades between money and comfort. It is fitted to your history every night; the numbers can only
        move a little per day. Below: the health of the fits, then every parameter and weight, then how they have changed.
      </PageIntro>
      <div className="cards">
        <Card label="Pump model" value={pump.source} note={fitted_at ? `fitted ${fitted_at.slice(0, 16).replace('T', ' ')}` : undefined} />
        <Card label="Room model" value={room.source} note={room.fit_error ?? undefined} tone={room.fit_error ? 'bad' : undefined} />
        <Card label="Last fit error" value={last_fit_error ?? 'none'} tone={last_fit_error ? 'bad' : 'good'} />
      </div>
      {pump.fit && (
        <div className="panel">
          <h2>Heat pump</h2>
          <p className="note">How the pump turns the setpoint into a flow temperature and into electricity, with how far the fit is off on average.</p>
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
          <p className="note">How the room follows the flow temperature over days, fitted per heating season; the seasons should agree.</p>
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
          <p className="note">The model's error over several days against simply assuming the room stays as it is.</p>
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
      {data.models.warmup.fit ? (
        <div className="panel">
          <h2>Warm-up and cooling speed</h2>
          <p className="note">How fast the room really warmed and cooled after setpoint changes, against the model.</p>
          <p>
            {warmupText(data.models.warmup.fit)} Raising the setpoint 1 °C warmed the room {data.models.warmup.fit.response_c_per_c.toFixed(2)} °C over 24 h
            (the model implies {data.models.warmup.fit.predicted_c_per_c.toFixed(2)}); lowering it cooled the room {data.models.warmup.fit.cooling_response_c_per_c.toFixed(2)} °C per degree,
            {' '}{data.models.warmup.fit.cooling_scale.toFixed(1)} ± {data.models.warmup.fit.cooling_se.toFixed(1)} x the model. {data.models.warmup.fit.windows} days, {data.models.warmup.fit.step_windows} with a setpoint change.
          </p>
          <p className="note">
            Warming is hard to see in the history because the LK thermostats stop heat above their setpoint.
            Cooling looks faster than the model, but the old manual changes were often made when the room was already warm, so it may be overstated.
          </p>
        </div>
      ) : (
        <div className="panel">
          <h2>Warm-up and cooling speed</h2>
          <p className="note">Not measured yet: {data.models.warmup.fit_error ?? 'waiting for the first refit'}</p>
        </div>
      )}
      {data.models.night.fit ? (
        <div className="panel">
          <h2>Night cooling</h2>
          <p className="note">How much the room cools over a night, from your nights; the plan uses it to know how warm to keep the evening.</p>
          <p>
            Over a night ({data.models.night.fit.night_hours} h) the room changes {data.models.night.fit.intercept_c.toFixed(2)} °C at 0 °C outdoors,
            and {data.models.night.fit.slope_c_per_c.toFixed(3)} °C less for every degree warmer outdoors; a room that starts the night 1 °C warmer than{' '}
            {data.models.night.fit.level_ref_c.toFixed(1)} °C changes {data.models.night.fit.level_c_per_c.toFixed(2)} °C more ({data.models.night.fit.nights} nights, error{' '}
            {data.models.night.fit.rmse_c.toFixed(2)} °C).
          </p>
          <table>
            <thead><tr><th>Night</th><th>Outdoors</th><th>Measured</th><th>Predicted</th></tr></thead>
            <tbody>
              {[...data.models.night.fit.recent].reverse().map((n) => (
                <tr key={n.day}>
                  <td>{n.day}</td>
                  <td>{n.t_out_c.toFixed(1)} °C</td>
                  <td>{n.measured_change_c.toFixed(2)} °C</td>
                  <td>{n.predicted_change_c.toFixed(2)} °C</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="panel">
          <h2>Night cooling</h2>
          <p className="note">Not learned yet: {data.models.night.fit_error ?? 'waiting for the first refit'}</p>
        </div>
      )}
      {catalog.data && <ParameterList entries={catalog.data} />}
      {learned.data && <Learning history={learned.data} />}
      <div className="panel">
        <h2>Live prediction error (shadow plans vs measured room)</h2>
        <p className="note">How far the room was from what plans made 1, 6 and 24 hours earlier expected.</p>
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
