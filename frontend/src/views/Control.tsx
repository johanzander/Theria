import { useState } from 'react'
import { fetchExecution, fetchPlan, setMode } from '../api'
import { Card, useLoaded } from '../ui'

export function Control() {
  const execution = useLoaded(fetchExecution)
  const plan = useLoaded(fetchPlan)
  const [message, setMessage] = useState<string | null>(null)
  const [armed, setArmed] = useState(false)
  const mode = execution.data?.mode ?? plan.data?.mode ?? '-'

  const change = async (next: string, confirm: boolean) => {
    const error = await setMode(next, confirm)
    setArmed(false)
    setMessage(error ?? `Mode is now ${next}. Reload to see it in the cards.`)
  }

  return (
    <>
      <div className="cards">
        <Card label="Mode" value={mode} />
        <Card label="Stage" value={execution.data ? `± ${execution.data.stage.max_deviation_c} °C` : '-'} note={execution.data ? `raise at most ${execution.data.stage.max_rise_per_hour_c} °C per hour` : undefined} />
        <Card label="Last write" value={execution.data?.last_execution?.action ?? '-'} note={execution.data?.last_execution?.reason} />
        <Card
          label="LK thermostats"
          value={execution.data?.last_lk ? `${execution.data.last_lk.action} ${execution.data.last_lk.target_c ?? ''}`.trim() : '-'}
          note={execution.data?.last_lk?.reason}
        />
      </div>
      <div className="panel">
        <h2>Mode</h2>
        <p className="note">
          Off: nothing is computed. Shadow: plans are computed and shown, nothing is written. Active: the plan's setpoint is written
          to the heat pump (within the stage limits); going back to shadow or off puts the base setpoint back.
        </p>
        <div className="buttons">
          <button onClick={() => change('off', false)} disabled={mode === 'off'}>Off</button>
          <button onClick={() => change('shadow', false)} disabled={mode === 'shadow'}>Shadow</button>
          {!armed ? (
            <button className="danger" onClick={() => setArmed(true)} disabled={mode === 'active'}>Active...</button>
          ) : (
            <>
              <button className="danger" onClick={() => change('active', true)}>Confirm: write to the heat pump</button>
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
    </>
  )
}
