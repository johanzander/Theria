import { useState } from 'react'
import { fetchExecution, fetchLog, fetchPlan, fetchPreflight, refitModels, setMode } from '../api'
import { roomRows, statusTone } from '../model'
import { Card, useLoaded } from '../ui'

export function Control() {
  const execution = useLoaded(fetchExecution)
  const plan = useLoaded(fetchPlan)
  const preflight = useLoaded(fetchPreflight)
  const log = useLoaded(fetchLog)
  const [message, setMessage] = useState<string | null>(null)
  const [armed, setArmed] = useState(false)
  const [refitting, setRefitting] = useState(false)
  const [refitMessage, setRefitMessage] = useState<string | null>(null)
  const mode = execution.data?.mode ?? plan.data?.mode ?? '-'
  const canActivate = preflight.data?.ok === true

  const refresh = () => {
    execution.reload()
    plan.reload()
    preflight.reload()
    log.reload()
  }

  const change = async (next: string, confirm: boolean) => {
    const error = await setMode(next, confirm)
    setArmed(false)
    setMessage(error ?? `Mode is now ${next}.`)
    refresh()
  }

  const refit = async () => {
    setRefitting(true)
    const error = await refitModels()
    setRefitting(false)
    setRefitMessage(error ?? 'Models refitted. See the Models tab.')
  }

  return (
    <>
      <div className="cards">
        <Card label="Mode" value={mode} tone={mode === 'active' ? 'bad' : undefined} />
        <Card
          label="Pump setpoint"
          value={execution.data?.last_execution?.action ?? '-'}
          note={execution.data?.last_execution?.reason}
        />
        <Card
          label="LK thermostats"
          value={execution.data?.last_lk ? `${execution.data.last_lk.action} ${execution.data.last_lk.target_c ?? ''}`.trim() : '-'}
          note={execution.data?.last_lk?.reason}
        />
      </div>

      {execution.data && (
        <div className="panel">
          <h2>Rooms</h2>
          <p className="note">
            The regulator watches each room every 15 minutes. The LK thermostat is held at the room limit; a room is closed a little
            earlier, by the rise it has learned after the valves close. Plan freedom (authority): {Math.round((execution.data.authority ?? 0.5) * 100)} % of the setpoint range.
            {execution.data.radiators
              ? ` Radiators: ${execution.data.radiators.calling ? 'calling for heat, no cuts' : 'satisfied'} (${execution.data.radiators.readable} readable).`
              : ''}
          </p>
          {execution.data.regulator && execution.data.regulator.length > 0 ? (
            <table>
              <thead><tr><th>Room</th><th>Now</th><th>Trend</th><th>Valves</th><th>State</th><th>Closes at</th><th>Learned rise</th></tr></thead>
              <tbody>
                {roomRows(execution.data.regulator).map((r) => (
                  <tr key={r.room}>
                    <td>{r.room}</td><td>{r.temperature}</td><td>{r.rate}</td><td>{r.valves}</td>
                    <td className={r.tone}>{r.state}</td><td>{r.closeAt}</td><td>{r.rise}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="note">No rooms are configured for the regulator.</p>
          )}
        </div>
      )}

      <div className="panel">
        <h2>Before activating</h2>
        {preflight.data ? (
          <table>
            <tbody>
              {preflight.data.checks.map((c) => (
                <tr key={c.name}>
                  <td className={statusTone(c.status)}>
                    <b>{c.status.toUpperCase()}</b>
                  </td>
                  <td>{c.name}</td>
                  <td>{c.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>Loading...</p>
        )}
        <p className="note">Active mode can only be switched on when no check fails. Warnings are allowed.</p>
      </div>

      <div className="panel">
        <h2>Mode</h2>
        <p className="note">
          Off: nothing is computed. Shadow: plans are computed and shown, nothing is written. Active: the plan's setpoint is written to the
          heat pump and the LK thermostats are raised while pre-heating. Going back to shadow or off puts the base
          setpoints back.
        </p>
        <div className="buttons">
          <button onClick={() => change('off', false)} disabled={mode === 'off'}>Off</button>
          <button onClick={() => change('shadow', false)} disabled={mode === 'shadow'}>Shadow</button>
          {!armed ? (
            <button className="danger" onClick={() => setArmed(true)} disabled={mode === 'active' || !canActivate}>
              Active...
            </button>
          ) : (
            <>
              <button className="danger" onClick={() => change('active', true)}>Confirm: write to the heat pump and LK</button>
              <button onClick={() => setArmed(false)}>Cancel</button>
            </>
          )}
        </div>
        {message && <p className="note">{message}</p>}
      </div>

      <div className="panel">
        <h2>Models</h2>
        <p className="note">The models are refitted to history every night. Refit now to update them after a change (takes several seconds).</p>
        <div className="buttons">
          <button onClick={refit} disabled={refitting}>{refitting ? 'Refitting...' : 'Refit models now'}</button>
        </div>
        {refitMessage && <p className="note">{refitMessage}</p>}
      </div>

      {execution.data && (
        <div className="panel">
          <h2>Trouble</h2>
          <p className="note">A hand change of the setpoint pauses the plan for an hour; stopping and starting active mode clears the pause.</p>
          {execution.data.last_trouble && <p className="bad">Last trouble: {execution.data.last_trouble}</p>}
        </div>
      )}

      <div className="panel">
        <h2>What it did</h2>
        {log.data && log.data.length > 0 ? (
          <table>
            <thead><tr><th>Time (UTC)</th><th>What</th><th>Action</th><th>Detail</th></tr></thead>
            <tbody>
              {[...log.data].reverse().map((e, i) => (
                <tr key={i}>
                  <td>{e.at.slice(5, 16).replace('T', ' ')}</td>
                  <td>{e.kind}</td>
                  <td>{e.action}</td>
                  <td>{e.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="note">Nothing yet. The log lists setpoint writes, trouble and mode changes since the add-on started.</p>
        )}
      </div>
    </>
  )
}
