import { useEffect, useState } from 'react'
import { fetchSettings, fetchTuning, resetSetting, saveSettings } from '../api'
import { formatNumber, groupBy, sortedRecommendations, type SettingEntry } from '../model'
import { PageIntro, Panel, useLoaded } from '../ui'

const TONE = { ok: 'good', consider: 'bad', 'no data': '' } as const

/** The number or switch as typed text, and back; a switch is 'true' or 'false'. */
const asText = (v: number | boolean) => String(v)
const parse = (entry: SettingEntry, text: string): number | boolean | null => {
  if (entry.kind === 'bool') return text === 'true'
  const n = Number(text)
  return text.trim() === '' || Number.isNaN(n) ? null : n
}

function Editor({ entry, text, onText }: { entry: SettingEntry; text: string; onText: (text: string) => void }) {
  if (entry.kind === 'bool') {
    return (
      <select value={text} onChange={(e) => onText(e.target.value)}>
        <option value="true">on</option>
        <option value="false">off</option>
      </select>
    )
  }
  const changed = text !== asText(entry.value)
  return (
    <input
      className={`setting${changed ? ' changed' : ''}`}
      type="number"
      step={entry.step ?? undefined}
      min={entry.minimum ?? undefined}
      max={entry.maximum ?? undefined}
      value={text}
      onChange={(e) => onText(e.target.value)}
    />
  )
}

function SettingsTable({ settings, reload }: { settings: SettingEntry[]; reload: (next: SettingEntry[]) => void }) {
  const [texts, setTexts] = useState<Record<string, string>>({})
  const [message, setMessage] = useState<{ text: string; bad: boolean } | null>(null)
  useEffect(() => setTexts(Object.fromEntries(settings.map((s) => [s.key, asText(s.value)]))), [settings])
  const changes = settings.filter((s) => texts[s.key] !== undefined && texts[s.key] !== asText(s.value))
  const save = async () => {
    const values: Record<string, number | boolean> = {}
    for (const s of changes) {
      const v = parse(s, texts[s.key])
      if (v === null) return setMessage({ text: `${s.label}: not a number`, bad: true })
      values[s.key] = v
    }
    const { settings: next, error } = await saveSettings(values)
    if (error || !next) return setMessage({ text: error ?? 'not saved', bad: true })
    reload(next)
    setMessage({ text: `Saved ${changes.length} setting${changes.length === 1 ? '' : 's'}; in use now.`, bad: false })
  }
  const reset = async (s: SettingEntry) => {
    const { settings: next, error } = await resetSetting(s.key)
    if (error || !next) return setMessage({ text: error ?? 'not reset', bad: true })
    reload(next)
    setMessage({ text: `${s.label} is back at ${formatNumber(s.default)}.`, bad: false })
  }
  return (
    <Panel
      title="Settings"
      description="Everything you can tune. A changed value is checked, saved and used at once by the plan, the pump and the thermostats, and it is kept across restarts. Orange means it differs from the add-on configuration (the default shown); reset goes back to it. Entities, coordinates and the price feed stay in the add-on configuration."
    >
      <table>
        <thead>
          <tr><th>Setting</th><th className="num">Value</th><th>Unit</th><th className="num">Default</th><th>What it does</th><th></th></tr>
        </thead>
        <tbody>
          {groupBy(settings).flatMap(([group, items]) => [
            <tr key={group} className="group"><td colSpan={6}>{group}</td></tr>,
            ...items.map((s) => (
              <tr key={s.key}>
                <td>{s.label}</td>
                <td className="num"><Editor entry={s} text={texts[s.key] ?? asText(s.value)} onText={(text) => setTexts({ ...texts, [s.key]: text })} /></td>
                <td>{s.unit}</td>
                <td className="num">{formatNumber(s.default)}</td>
                <td className="desc">{s.description}</td>
                <td>{s.source === 'override' && <button onClick={() => reset(s)}>Reset</button>}</td>
              </tr>
            )),
          ])}
        </tbody>
      </table>
      <div className="buttons">
        <button onClick={save} disabled={changes.length === 0}>Save {changes.length > 0 ? `${changes.length} change${changes.length === 1 ? '' : 's'}` : 'changes'}</button>
        {changes.length > 0 && <button onClick={() => setTexts(Object.fromEntries(settings.map((s) => [s.key, asText(s.value)])))}>Discard</button>}
      </div>
      {message && <p className={message.bad ? 'bad' : 'good'}>{message.text}</p>}
    </Panel>
  )
}

export function Settings() {
  const settings = useLoaded(fetchSettings)
  const advice = useLoaded(fetchTuning)
  const [fresh, setFresh] = useState<SettingEntry[] | null>(null)
  const current = fresh ?? settings.data
  if (settings.error) return <p className="bad">{settings.error}</p>
  return (
    <>
      <PageIntro>
        Settings are the choices that are yours: how warm, how much comfort is worth, where the limits are. The advice below them comes from
        the measured days and is only a suggestion; nothing is changed without you.
      </PageIntro>
      {current ? <SettingsTable settings={current} reload={setFresh} /> : <p>Loading...</p>}
      {advice.data && (
        <Panel
          title="Advice from the measured days"
          description="What the days so far say about some of the settings above. 'Consider' means the data suggests a look; it never changes anything by itself."
        >
          <table>
            <thead>
              <tr><th>Setting</th><th>Now</th><th>What the data says</th><th>Consider</th><th>Why</th></tr>
            </thead>
            <tbody>
              {sortedRecommendations(advice.data).map((r) => (
                <tr key={r.key}>
                  <td className={TONE[r.status]}><b>{r.title}</b></td>
                  <td>{r.current}</td>
                  <td>{r.observed}</td>
                  <td>{r.suggestion ?? '-'}</td>
                  <td className="desc">{r.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
    </>
  )
}
