/** The pump's electricity per day (GET energy-forecast): each forecast as it was made next to what was measured. */

export interface EnergyDay {
  day: string
  weather: 'forecast' | 'measured'
  clock: number[]
  forecasts: Record<string, number[]>
  actual: (number | null)[]
}

export interface EnergySummaryRow {
  method: string
  weather: 'forecast' | 'measured'
  days: number
  day_error_kwh: number
  bias_kwh: number
  share: number
}

export interface EnergyForecast {
  days: EnergyDay[]
  summary: EnergySummaryRow[]
}

export const METHODS: { key: string; label: string; color: string }[] = [
  { key: 'simple', label: 'Model', color: '#0b6bcb' },
  { key: 'mean_48h', label: 'Mean of the last 48 h', color: '#eb6834' },
  { key: 'profile_7d', label: 'Last 7 days by hour', color: '#1baf7a' },
]

const WEATHER_LABEL = { forecast: 'Weather forecast', measured: 'Weather as it came' }

const sum = (values: number[]) => values.reduce((a, b) => a + b, 0)

export function hourlyRows(day: EnergyDay) {
  return day.clock.map((hour, i) => ({
    hour: String(hour).padStart(2, '0'),
    actual: day.actual[i],
    ...Object.fromEntries(METHODS.map((m) => [m.key, day.forecasts[m.key][i]])),
  }))
}

export function dayTotals(day: EnergyDay) {
  const measured = day.actual.every((v) => v !== null)
  return { forecast: sum(day.forecasts.simple), actual: measured ? sum(day.actual as number[]) : null }
}

export function dailyRows(days: EnergyDay[]) {
  return days.map((d) => ({ day: d.day.slice(5), ...dayTotals(d) }))
}

export function summaryRows(rows: EnergySummaryRow[]) {
  const order = METHODS.map((m) => m.key)
  return [...rows]
    .sort((a, b) => a.weather.localeCompare(b.weather) || order.indexOf(a.method) - order.indexOf(b.method))
    .map((r) => ({
      ...r,
      label: METHODS.find((m) => m.key === r.method)?.label ?? r.method,
      weatherLabel: WEATHER_LABEL[r.weather],
    }))
}
