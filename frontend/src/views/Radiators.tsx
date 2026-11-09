import { useEffect, useState } from 'react'
import { fetchRadiatorPrefs, fetchRadiators, resetRadiatorPrefs, saveRadiatorPrefs } from '../api'
import { formatNumber } from '../model'
import {
  formatDifference, overallAvailability, rowsToSchedule, scheduleToRows, warnings,
  type GuestRoom, type PrefsResponse, type RadiatorPrefs, type RadiatorSnapshot, type ScheduleRow,
} from '../radiators'
import { Card, PageIntro, Panel, useLoaded } from '../ui'

const NUMBERS: { key: keyof RadiatorPrefs; label: string; description: string; step: number }[] = [
  { key: 'salon_open_target_c', label: 'Salon target (open)', description: 'The mean of all salon points should be about this while the shop is open.', step: 0.5 },
  { key: 'salon_open_floor_c', label: 'Salon floor (open)', description: 'Below this while the shop is open, the heat pump plan stops lowering the setpoint.', step: 0.5 },
  { key: 'salon_margin_c', label: 'Salon margin', description: 'The plan keeps the mean this far above the target in its prediction.', step: 0.1 },
  { key: 'salon_preheat_c', label: 'Salon pre-heat', description: 'The highest mean the salon is heated to before dear hours.', step: 0.5 },
  { key: 'salon_closed_floor_c', label: 'Salon closed floor', description: 'The salon may cool this far when the shop is closed.', step: 0.5 },
  { key: 'salon_max_setpoint_c', label: 'Salon highest thermostat setting', description: 'The thermostats are never set above this.', step: 0.5 },
  { key: 'guest_band_c', label: 'Guest house band', description: 'How far the guest house setpoints may move around each room temperature, with the price.', step: 0.5 },
]

const clock = (iso: string) => new Date(iso).toLocaleTimeString('sv-SE', { hour: '2-digit', minute: '2-digit' })
const temp = (v: number | null) => (v === null ? '-' : `${v.toFixed(1)} °C`)

function Status({ snap }: { snap: RadiatorSnapshot }) {
  const reading = snap.salon.reading
  const guestSaving = snap.guest_saving
  return (
    <div className="cards">
      <Card label="Mode" value="Shadow" note="Nothing is written to the thermostats; the log shows what it would write." />
      <Card
        label="Salon now"
        value={reading.mean_c === null ? 'unknown' : `${reading.mean_c.toFixed(1)} °C`}
        tone={reading.unknown ? 'bad' : undefined}
        note={`${reading.used.length} valid points${Object.keys(reading.skipped).length ? `, left out: ${Object.keys(reading.skipped).join(', ')}` : ''}`}
      />
      <Card label="Thermostats readable" value={`${Math.round(overallAvailability(snap.availability_share) * 100)} %`} note="Share of cycles since start." />
      <Card
        label="Salon saving (estimate)"
        value={snap.salon.saving_sek === null ? '-' : `${snap.salon.saving_sek.toFixed(1)} SEK`}
        note="Against holding the mean at the baseline all hours: includes the night setback, not only moving heat between hours."
      />
      <Card label="Guest house saving" value={`${guestSaving[0].toFixed(1)} SEK`} note={guestSaving[1]} />
    </div>
  )
}

function Thermostats({ snap }: { snap: RadiatorSnapshot }) {
  return (
    <Panel title="Thermostats now" description="What each thermostat is set to, what the plan would set it to this hour, and whether Home Assistant could read it.">
      <table>
        <thead><tr><th>Thermostat</th><th className="num">Set now</th><th className="num">Would set</th><th className="num">Change</th><th>Readable</th></tr></thead>
        <tbody>
          {Object.entries(snap.availability).map(([entity, ok]) => (
            <tr key={entity}>
              <td>{entity}</td>
              <td className="num">{temp(snap.read[entity])}</td>
              <td className="num">{snap.would_write[entity] === undefined ? '-' : temp(snap.would_write[entity])}</td>
              <td className="num">{snap.would_write[entity] === undefined ? '-' : formatDifference(snap.read[entity], snap.would_write[entity])}</td>
              <td className={ok ? 'good' : 'bad'}>{ok ? 'yes' : 'no'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  )
}

function SalonDay({ snap }: { snap: RadiatorSnapshot }) {
  if (snap.salon.hours.length === 0) {
    return <Panel title="Salon plan" description="The hours ahead."><p className="bad">The salon is unknown (fewer than two valid temperature points), so it is left as it is.</p></Panel>
  }
  return (
    <Panel title="Salon plan" description="Hour by hour: the price, the mean temperature the plan aims for at the end of the hour, and the thermostat setting that gives it.">
      <table>
        <thead><tr><th>Hour</th><th>Shop</th><th className="num">Price</th><th className="num">Mean at end</th><th className="num">Thermostats</th><th>Why</th></tr></thead>
        <tbody>
          {snap.salon.hours.map((h) => (
            <tr key={h.start}>
              <td>{clock(h.start)}</td>
              <td>{h.open ? 'open' : 'closed'}</td>
              <td className="num">{formatNumber(h.price_sek_per_kwh)} SEK</td>
              <td className="num">{temp(h.mean_end_c)}</td>
              <td className="num">{temp(h.setpoint_c)}</td>
              <td className="desc">{h.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  )
}

function Editor({ data, reload }: { data: PrefsResponse; reload: (next: PrefsResponse) => void }) {
  const [rows, setRows] = useState<ScheduleRow[]>(() => scheduleToRows(data.prefs))
  const [numbers, setNumbers] = useState<Record<string, string>>({})
  const [rooms, setRooms] = useState<GuestRoom[]>(data.prefs.guest_rooms)
  const [mode, setMode] = useState(data.prefs.mode)
  const [message, setMessage] = useState<{ text: string; bad: boolean } | null>(null)
  useEffect(() => {
    setRows(scheduleToRows(data.prefs))
    setRooms(data.prefs.guest_rooms)
    setMode(data.prefs.mode)
    setNumbers(Object.fromEntries(NUMBERS.map((n) => [n.key, String(data.prefs[n.key])])))
  }, [data])

  const save = async () => {
    const { schedule, error } = rowsToSchedule(rows)
    if (error || !schedule) return setMessage({ text: error ?? 'not saved', bad: true })
    const values: Record<string, unknown> = { salon_schedule: schedule, guest_rooms: rooms, mode }
    for (const n of NUMBERS) {
      const v = Number(numbers[n.key])
      if (numbers[n.key]?.trim() === '' || Number.isNaN(v)) return setMessage({ text: `${n.label}: not a number`, bad: true })
      values[n.key] = v
    }
    const result = await saveRadiatorPrefs(values as Partial<RadiatorPrefs>)
    if (result.error || !result.prefs) return setMessage({ text: result.error ?? 'not saved', bad: true })
    reload(result.prefs)
    setMessage({ text: 'Saved; in use at the next cycle.', bad: false })
  }
  const reset = async () => {
    const result = await resetRadiatorPrefs()
    if (result.error || !result.prefs) return setMessage({ text: result.error ?? 'not reset', bad: true })
    reload(result.prefs)
    setMessage({ text: 'Back at the add-on configuration.', bad: false })
  }
  const setRow = (i: number, patch: Partial<ScheduleRow>) => setRows(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const time = (v: string) => (v === '' ? null : v)
  return (
    <Panel
      title="Salon opening hours and temperatures, guest house rooms"
      description="Leave both times empty for a closed day. Changes are checked, saved and used at the next cycle (every 15 minutes)."
    >
      <table>
        <thead><tr><th>Day</th><th>Opens</th><th>Closes</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.day}>
              <td>{r.day}</td>
              <td><input type="time" value={r.open ?? ''} onChange={(e) => setRow(i, { open: time(e.target.value) })} /></td>
              <td><input type="time" value={r.close ?? ''} onChange={(e) => setRow(i, { close: time(e.target.value) })} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <table>
        <thead><tr><th>Setting</th><th className="num">Value</th><th className="num">Default</th><th>What it does</th></tr></thead>
        <tbody>
          {NUMBERS.map((n) => (
            <tr key={n.key}>
              <td>{n.label}</td>
              <td className="num"><input className="setting" type="number" step={n.step} value={numbers[n.key] ?? ''} onChange={(e) => setNumbers({ ...numbers, [n.key]: e.target.value })} /></td>
              <td className="num">{formatNumber(data.defaults[n.key] as number)}</td>
              <td className="desc">{n.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <table>
        <thead><tr><th>Guest house room</th><th className="num">Wanted temperature</th><th className="num">Band (empty: the common band)</th></tr></thead>
        <tbody>
          {rooms.map((r, i) => (
            <tr key={r.entity}>
              <td>{r.entity}</td>
              <td className="num"><input className="setting" type="number" step={0.5} value={r.t_room_c} onChange={(e) => setRooms(rooms.map((x, j) => (j === i ? { ...x, t_room_c: Number(e.target.value) } : x)))} /></td>
              <td className="num"><input className="setting" type="number" step={0.5} value={r.band_c ?? ''} onChange={(e) => setRooms(rooms.map((x, j) => (j === i ? { ...x, band_c: e.target.value === '' ? null : Number(e.target.value) } : x)))} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <label>
        Mode{' '}
        <select value={mode} onChange={(e) => setMode(e.target.value as RadiatorPrefs['mode'])}>
          <option value="shadow">shadow (compute and log)</option>
          <option value="off">off</option>
        </select>
      </label>
      <div className="buttons">
        <button onClick={save}>Save</button>
        {!data.is_default && <button onClick={reset}>Reset to configuration</button>}
      </div>
      {message && <p className={message.bad ? 'bad' : 'good'}>{message.text}</p>}
    </Panel>
  )
}

export function Radiators() {
  const status = useLoaded(fetchRadiators)
  const prefs = useLoaded(fetchRadiatorPrefs)
  const [fresh, setFresh] = useState<PrefsResponse | null>(null)
  const current = fresh ?? prefs.data
  if (status.error) return <p className="bad">{status.error}</p>
  return (
    <>
      <PageIntro>
        The salon (Netatmo) and the guest house (Tado) take part in the price saving: less heat in the dear hours, a little extra before them.
        This is shadow mode: the page shows what Theria would set the thermostats to, and nothing is written yet.
      </PageIntro>
      {status.data ? (
        <>
          {warnings(status.data).map((w) => <p key={w} className="bad">{w}</p>)}
          <Status snap={status.data} />
          <Thermostats snap={status.data} />
          <SalonDay snap={status.data} />
        </>
      ) : (
        <p>No radiator plan yet (it is computed every 15 minutes, once the prices are known).</p>
      )}
      {current ? <Editor data={current} reload={setFresh} /> : <p>Loading...</p>}
    </>
  )
}
