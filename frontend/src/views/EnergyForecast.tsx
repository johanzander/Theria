import { useState } from 'react'
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fetchEnergyForecast } from '../api'
import { dailyRows, dayTotals, hourlyRows, METHODS, summaryRows, type EnergyForecast as Data } from '../energyForecast'
import { Card, PageIntro, Panel, useLoaded } from '../ui'

const kwh = (value: number | null) => (value === null ? '-' : `${value.toFixed(1)} kWh`)
const signed = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}`
const tooltip = (value: unknown) => (typeof value === 'number' ? `${value.toFixed(2)} kWh` : '-')

export function EnergyForecast() {
  const [days, setDays] = useState(14)
  const { data, error, loaded } = useLoaded<Data>(() => fetchEnergyForecast(days), [days])
  const [picked, setPicked] = useState<string | null>(null)
  if (error) return <p className="note">Could not load: {error}</p>
  if (!loaded) return <p className="note">Loading...</p>
  if (!data || data.days.length === 0) return <p className="note">No forecast has been made yet.</p>
  const selected = data.days.find((d) => d.day === picked) ?? data.days[data.days.length - 1]
  const totals = dayTotals(selected)
  return (
    <>
      <PageIntro>
        The heat pump's compressor electricity, heating and hot water. Tomorrow's forecast is made after 14:00. Days marked
        "weather as it came" were filled in afterwards and show the model without the error of the weather forecast. The
        electric heater is not included.
      </PageIntro>
      <p>
        {[14, 30, 90].map((d) => (
          <button key={d} onClick={() => setDays(d)} disabled={d === days}>{d} days</button>
        ))}
      </p>
      <Panel title="Day by day" description="Bars are what was measured, the line what the model forecast.">
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={dailyRows(data.days)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis unit=" kWh" width={70} />
            <Tooltip formatter={tooltip} />
            <Legend />
            <Bar dataKey="actual" name="Measured" fill="#9aa5b1" />
            <Line dataKey="forecast" name="Model" stroke="#0b6bcb" dot={false} strokeWidth={2} />
          </ComposedChart>
        </ResponsiveContainer>
      </Panel>
      <Panel title="Error" description="The mean error of the day's total. Bias above zero means the forecast was too high.">
        {data.summary.length === 0 ? (
          <p className="note">No finished day to compare yet.</p>
        ) : (
          <table>
            <thead>
              <tr><th>Method</th><th>Weather</th><th>Days</th><th>Error per day</th><th>Bias</th><th>Share of the use</th></tr>
            </thead>
            <tbody>
              {summaryRows(data.summary).map((r) => (
                <tr key={`${r.method}-${r.weather}`}>
                  <td>{r.label}</td><td>{r.weatherLabel}</td><td>{r.days}</td>
                  <td>{r.day_error_kwh.toFixed(1)} kWh</td><td>{signed(r.bias_kwh)} kWh</td><td>{(100 * r.share).toFixed(0)} %</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
      <Panel title="One day, hour by hour" description="Single hours depend on hot water and compressor starts, so read them as indicative.">
        <p>
          <select value={selected.day} onChange={(e) => setPicked(e.target.value)}>
            {data.days.map((d) => <option key={d.day} value={d.day}>{d.day}</option>)}
          </select>
        </p>
        <div className="cards">
          <Card label="Model" value={kwh(totals.forecast)} note={selected.weather === 'forecast' ? 'Weather forecast' : 'Weather as it came'} />
          <Card label="Measured" value={kwh(totals.actual)} />
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart data={hourlyRows(selected)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis unit=" kWh" width={70} />
            <Tooltip formatter={tooltip} />
            <Legend />
            <Bar dataKey="actual" name="Measured" fill="#9aa5b1" />
            {METHODS.map((m) => (
              <Line key={m.key} dataKey={m.key} name={m.label} stroke={m.color} dot={false} strokeWidth={m.key === 'simple' ? 2.5 : 1.5} strokeDasharray={m.key === 'simple' ? undefined : '6 3'} />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </Panel>
    </>
  )
}
