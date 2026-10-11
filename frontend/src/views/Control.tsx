import { useState } from 'react'
import { fetchExecution, fetchLog, fetchPlan, fetchPreflight, setMode } from '../api'
import { statusTone } from '../model'
import { Card, useLoaded } from '../ui'

export function Control() {
  const execution = useLoaded(fetchExecution)
  const plan = useLoaded(fetchPlan)
  const preflight = useLoaded(fetchPreflight)
  const log = useLoaded(fetchLog)
  const [message, setMessage] = useState<string | null>(null)
  const [armed, setArmed] = useState(false)
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

  return (
    <>
      <div className="cards">
        <Card label="Mode" value={mode} tone={mode === 'active' ? 'bad' : undefined} />
        <Card
          label="Stage"
          value={execution.data ? `± ${execution.data.stage.max_deviation_c} °C` : '-'}
          note={execution.data ? `raise at most ${execution.data.stage.max_rise_per_hour_c} °C per hour` : undefined}
        />
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
          heat pump (within the stage limits) and the LK thermostats are raised while pre-heating. Going back to shadow or off puts the base
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

      {execution.data && (
        <div className="panel">
          <h2>Stage and safety</h2>
          <p>Stage {execution.data.stage.index + 1}, started {execution.data.stage.started_on}.</p>
          {execution.data.stage.revert_until && <p className="bad">At the base setpoint until {execution.data.stage.revert_until}.</p>}
          {execution.data.last_stage_note && <p className="note">Daily judgement: {execution.data.last_stage_note}</p>}
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
          <p className="note">Nothing yet. The log lists setpoint writes, trouble, stage and mode changes since the add-on started.</p>
        )}
      </div>
    </>
  )
}
