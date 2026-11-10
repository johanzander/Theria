import { useState } from 'react'
import { debugExportUrl, fetchExecution, fetchLog, fetchPlan, fetchPreflight, refitModels, setMode } from '../api'
import { statusTone } from '../model'
import { Card, PageIntro, useLoaded } from '../ui'

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
      <PageIntro>
        Where Theria is allowed to act. Shadow: the plan is computed and shown but nothing is written. Active: the plan sets the heat pump's
        setpoint and each LK thermostat (closing in a sell, opening just before one, the normal target otherwise). A hand change on the pump
        pauses it for an hour; on a thermostat it pauses that room and leaves it alone until the next window.
      </PageIntro>
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

      <div className="panel">
        <h2>Before activating</h2>
        <p className="note">Checks that must pass before active mode can be switched on.</p>
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
        <p className="note">Change the mode. Active needs confirmation because it writes to the heat pump and the thermostats.</p>
        <p className="note">
          Off: nothing is computed. Shadow: plans are computed and shown, nothing is written. Active: the plan's setpoint is written to the
          heat pump and the LK thermostats are raised while pre-heating. Going back to shadow or off puts the base
          setpoints back.
        </p>
        <div className="buttons">
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

      <div className="panel">
        <h2>Debug data</h2>
        <p className="note">
          Everything needed to understand what Theria did and to play a plan again: key findings, settings, the entities it reads and
          writes, every plan run with its inputs, every control cycle with what the pump and the thermostats showed, the action log and
          the log. Attach it to a problem report.
        </p>
        <div className="buttons">
          <a href={debugExportUrl} download>Download debug data</a>
        </div>
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
        <p className="note">The latest writes, pauses and mode changes, newest first.</p>
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
