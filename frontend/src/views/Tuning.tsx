import { fetchTuning } from '../api'
import { sortedRecommendations } from '../model'
import { useLoaded } from '../ui'

const TONE = { ok: 'good', consider: 'bad', 'no data': '' } as const

export function Tuning() {
  const { data, error } = useLoaded(fetchTuning)
  if (error) return <p className="bad">{error}</p>
  if (!data) return <p>Loading...</p>
  return (
    <div className="panel">
      <h2>Settings worth a look</h2>
      <p className="note">
        Advice from the measured days. Nothing here is changed by itself: change a setting in the add-on configuration if you agree.
      </p>
      <table>
        <thead>
          <tr><th>Setting</th><th>Now</th><th>What the data says</th><th>Consider</th><th>Why</th></tr>
        </thead>
        <tbody>
          {sortedRecommendations(data).map((r) => (
            <tr key={r.key}>
              <td className={TONE[r.status]}><b>{r.title}</b></td>
              <td>{r.current}</td>
              <td>{r.observed}</td>
              <td>{r.suggestion ?? '-'}</td>
              <td>{r.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
