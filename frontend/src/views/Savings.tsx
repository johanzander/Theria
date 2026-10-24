import { useState } from 'react'
import { Bar, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fetchDay, fetchSavings } from '../api'
import { timelineRows, tooltipFormatter, UNITS } from '../model'
import { dailyRows, dailyTotals, shiftRows, shiftTotals, type DailyRow, type ShiftRow } from '../savings'
import { Card, DayNav, PageIntro, Panel, shiftDay, todayIso, useLoaded } from '../ui'
import { axisFor, Chart, guides } from './DayCharts'

const C = { flat: '#9aa5b1', plan: '#0b6bcb', preheat: '#e8903a', cut: '#0b6bcb', price: '#cbd2d9', saving: '#067647', measured: '#102a43' }
const HISTORY_DAYS = 30

const fix = (v: number | null, digits: number, suffix = '') => (v === null ? '-' : `${v.toFixed(digits)}${suffix}`)
const signed = (v: number, digits: number, suffix = '') => `${v > 0 ? '+' : ''}${v.toFixed(digits)}${suffix}`
const tone = (v: number) => (v > 0.005 ? 'good' : v < -0.005 ? 'bad' : undefined)
const unitOf = (name: string) => (name.includes('Price') ? UNITS.price : name.includes('kWh') ? UNITS.kwh : UNITS.sek)

function Summary({ rows }: { rows: ShiftRow[] }) {
  const t = shiftTotals(rows)
  const price = (v: number | null) => (v === null ? '-' : `${v.toFixed(2)} kr/kWh`)
  return (
    <div className="cards">
      <Card
        label="Saving against flat setpoint"
        value={`${t.saving.toFixed(1)} kr`}
        note={`${t.percent.toFixed(1)} % of the flat cost (${t.flatCost.toFixed(1)} kr), ${t.hours} h, from the plan's model`}
        tone={tone(t.saving)}
      />
      <Card label="Cost with the plan" value={`${t.planCost.toFixed(1)} kr`} note={`flat setpoint: ${t.flatCost.toFixed(1)} kr`} />
      <Card
        label="Energy moved"
        value={`${t.addedKwh.toFixed(1)} kWh`}
        note={`added in cheap hours; ${t.cutKwh.toFixed(1)} kWh cut in dear hours. Total ${t.planKwh.toFixed(1)} kWh against ${t.flatKwh.toFixed(1)} kWh flat`}
      />
      <Card label="Cut at" value={price(t.cutPrice)} note="average price of the energy the plan leaves out" />
      <Card label="Bought instead at" value={price(t.addedPrice)} note="average price of the energy the plan adds" />
      <Card
        label="Measured, hours passed"
        value={t.measuredSaving === null ? '-' : `${t.measuredSaving.toFixed(1)} kr`}
        note="measured electricity against the model's flat estimate"
        tone={t.measuredSaving === null ? undefined : tone(t.measuredSaving)}
      />
    </div>
  )
}

function HourCharts({ rows }: { rows: ShiftRow[] }) {
  const axis = axisFor(rows, 1)
  const now = rows.find((r) => r.current)?.key
  const dayStarts = axis.dayStarts
  const withMeasured = rows.some((r) => r.cumulativeMeasured !== null)
  return (
    <>
      <Panel
        title="Energy moved"
        description="Per hour: the heating energy the plan uses minus the energy the flat setpoint would use. Orange above zero: the plan pre-heats and buys extra energy in this hour. Blue below zero: the plan cuts. The grey line is the price; a good plan is orange where the price is low and blue where it is high."
      >
        <Chart rows={rows} axisRows={rows} height={260} unitFor={unitOf}>
          <YAxis yAxisId="kwh" unit=" kWh" />
          <YAxis yAxisId="price" orientation="right" unit=" kr" />
          <ReferenceLine yAxisId="kwh" y={0} stroke="#52606d" />
          <Bar yAxisId="kwh" dataKey="movedKwh" name="Moved energy (kWh)" legendType="none">
            {rows.map((r) => <Cell key={r.key} fill={r.movedKwh >= 0 ? C.preheat : C.cut} />)}
          </Bar>
          <Line yAxisId="price" type="stepAfter" dataKey="price" name="Price" stroke={C.flat} strokeWidth={2} dot={false} legendType="none" />
          {guides(now, dayStarts, 'kwh')}
        </Chart>
        <div className="legend">
          <span><i style={{ background: C.preheat }} />Added (pre-heat)</span>
          <span><i style={{ background: C.cut }} />Cut</span>
          <span><i style={{ background: C.flat }} />Price</span>
        </div>
      </Panel>

      <Panel
        title="Cost per hour, flat setpoint against plan"
        description="What each hour's heating electricity costs: grey with the flat setpoint, blue with the plan, both at the hour's price. The plan costs less in the dear hours and more in the cheap ones."
      >
        <Chart rows={rows} axisRows={rows} height={240} unitFor={unitOf}>
          <YAxis unit=" kr" />
          <Bar dataKey="flatCost" name="Flat setpoint (kr)" fill={C.flat} />
          <Bar dataKey="planCost" name="Plan (kr)" fill={C.plan} />
          {guides(now, dayStarts)}
        </Chart>
      </Panel>

      <Panel
        title="Saving, running total"
        description="The saving added up hour by hour over the days shown. Green: the plan's own model (flat cost minus plan cost). Dark: measured electricity against the model's flat estimate, for the hours that have passed. The line falls in the hours where heat is bought early and rises in the hours where it is not needed."
      >
        <Chart rows={rows} axisRows={rows} height={220} unitFor={unitOf}>
          <YAxis unit=" kr" />
          <ReferenceLine y={0} stroke="#52606d" />
          <Line type="monotone" dataKey="cumulative" name="Plan, model (kr)" stroke={C.saving} strokeWidth={2} dot={false} />
          {withMeasured && <Line type="monotone" dataKey="cumulativeMeasured" name="Measured (kr)" stroke={C.measured} strokeWidth={2} dot={false} connectNulls />}
          {guides(now, dayStarts)}
        </Chart>
      </Panel>
    </>
  )
}

function HourTable({ rows }: { rows: ShiftRow[] }) {
  const t = shiftTotals(rows)
  return (
    <Panel
      title="Hour by hour"
      description="Per hour, with the flat setpoint as the base case: the energy it would use and cost, what the plan uses and costs, the energy moved and the saving. Totals at the bottom. Measured columns fill in for hours that have passed."
    >
      <table>
        <thead>
          <tr>
            <th>Hour</th><th className="num">Price</th>
            <th className="num">Flat energy</th><th className="num">Plan energy</th><th className="num">Moved</th>
            <th className="num">Flat cost</th><th className="num">Plan cost</th><th className="num">Saving</th><th className="num">Total saving</th>
            <th className="num">Measured energy</th><th className="num">Measured saving</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} style={r.current ? { outline: '2px solid #1f2933' } : undefined}>
              <td>{r.key.slice(5)}</td>
              <td className="num">{r.price.toFixed(2)}</td>
              <td className="num">{r.flatKwh.toFixed(2)}</td>
              <td className="num">{r.planKwh.toFixed(2)}</td>
              <td className={`num ${r.movedKwh > 0 ? 'warn' : ''}`}>{signed(r.movedKwh, 2)}</td>
              <td className="num">{r.flatCost.toFixed(2)}</td>
              <td className="num">{r.planCost.toFixed(2)}</td>
              <td className={`num ${tone(r.saving) ?? ''}`}>{signed(r.saving, 2)}</td>
              <td className={`num ${tone(r.cumulative) ?? ''}`}>{signed(r.cumulative, 2)}</td>
              <td className="num">{fix(r.measuredKwh, 2)}</td>
              <td className="num">{r.measuredSaving === null ? '-' : signed(r.measuredSaving, 2)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <th>Total</th><th />
            <th className="num">{t.flatKwh.toFixed(1)} kWh</th><th className="num">{t.planKwh.toFixed(1)} kWh</th>
            <th className="num">+{t.addedKwh.toFixed(1)} / -{t.cutKwh.toFixed(1)}</th>
            <th className="num">{t.flatCost.toFixed(1)} kr</th><th className="num">{t.planCost.toFixed(1)} kr</th>
            <th className={`num ${tone(t.saving) ?? ''}`}>{signed(t.saving, 1, ' kr')}</th><th />
            <th /><th className="num">{t.measuredSaving === null ? '-' : signed(t.measuredSaving, 1, ' kr')}</th>
          </tr>
        </tfoot>
      </table>
    </Panel>
  )
}

function History({ days }: { days: DailyRow[] }) {
  const t = dailyTotals(days)
  return (
    <>
      <Panel
        title={`Measured, last ${days.length} days: ${t.saving.toFixed(0)} kr (${t.percent.toFixed(1)} %)`}
        description="Per day: what the heating electricity actually cost (blue) against what it would have cost with the flat setpoint (grey), and the running total of the saving (green line, right axis). Days with partial measurement are included as they are measured."
      >
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={days}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" tickFormatter={(d: string) => d.slice(5)} />
            <YAxis yAxisId="cost" unit=" kr" />
            <YAxis yAxisId="total" orientation="right" unit=" kr" />
            <Tooltip formatter={tooltipFormatter(unitOf)} />
            <Legend />
            <Bar yAxisId="cost" dataKey="baseline_cost_sek" name="Flat setpoint (kr)" fill={C.flat} />
            <Bar yAxisId="cost" dataKey="actual_cost_sek" name="Actual (kr)" fill={C.plan} />
            <Line yAxisId="total" type="monotone" dataKey="cumulative" name="Total saving (kr)" stroke={C.saving} strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </Panel>
      <Panel
        title="Day by day"
        description="The measured days: the flat-setpoint cost, the actual cost, the saving, the compressor and electric-heater energy, the mean pump setpoint and the lowest room temperature. Coverage is the share of the day's hours with a measurement."
      >
        <table>
          <thead>
            <tr>
              <th>Day</th><th className="num">Flat cost</th><th className="num">Actual cost</th><th className="num">Saving</th><th className="num">%</th>
              <th className="num">Total saving</th><th className="num">Compressor</th><th className="num">Heater</th>
              <th className="num">Mean setpoint</th><th className="num">Room min</th><th className="num">Coverage</th>
            </tr>
          </thead>
          <tbody>
            {[...days].reverse().map((d) => (
              <tr key={d.day}>
                <td>{d.day}</td>
                <td className="num">{d.baseline_cost_sek.toFixed(1)}</td>
                <td className="num">{d.actual_cost_sek.toFixed(1)}</td>
                <td className={`num ${tone(d.saving_sek) ?? ''}`}>{signed(d.saving_sek, 1)}</td>
                <td className={`num ${tone(d.percent) ?? ''}`}>{signed(d.percent, 1)}</td>
                <td className={`num ${tone(d.cumulative) ?? ''}`}>{signed(d.cumulative, 1)}</td>
                <td className="num">{d.compressor_kwh.toFixed(1)} kWh</td>
                <td className="num">{d.aux_kwh.toFixed(1)} kWh</td>
                <td className="num">{fix(d.mean_rts_c, 1, ' °C')}</td>
                <td className="num">{fix(d.room_min_c, 1, ' °C')}</td>
                <td className="num">{Math.round(d.coverage * 100)} %</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th>Total</th>
              <th className="num">{t.baseline.toFixed(1)} kr</th><th className="num">{t.actual.toFixed(1)} kr</th>
              <th className={`num ${tone(t.saving) ?? ''}`}>{signed(t.saving, 1, ' kr')}</th>
              <th className={`num ${tone(t.percent) ?? ''}`}>{signed(t.percent, 1)}</th><th />
              <th className="num">{t.compressorKwh.toFixed(0)} kWh</th><th className="num">{t.auxKwh.toFixed(1)} kWh</th><th /><th /><th />
            </tr>
          </tfoot>
        </table>
      </Panel>
    </>
  )
}

export function Savings() {
  const [day, setDay] = useState(todayIso())
  const first = useLoaded(() => fetchDay(day), [day])
  const second = useLoaded(() => fetchDay(shiftDay(day, 1)), [day])
  const history = useLoaded(() => fetchSavings(HISTORY_DAYS))
  const rows = shiftRows(timelineRows([first.data, second.data]))
  const days = dailyRows(history.data?.days ?? [])
  const error = first.error ?? second.error ?? history.error
  return (
    <>
      <PageIntro>
        The saving seen as energy and money moved in time: the plan cuts heating in dear hours and buys that energy in cheap ones, with
        the flat setpoint as the base case. The first half shows the days you choose hour by hour, from the plan's model; the second half
        shows what was measured, day by day.
      </PageIntro>
      {error && <p className="bad">{error}</p>}
      <DayNav day={day} onChange={setDay} />
      {rows.length === 0 ? (
        <p>{first.data ? 'Nothing to compare for this day yet.' : 'Loading...'}</p>
      ) : (
        <>
          <Summary rows={rows} />
          <HourCharts rows={rows} />
          <HourTable rows={rows} />
        </>
      )}
      {days.length > 0 && <History days={days} />}
    </>
  )
}
