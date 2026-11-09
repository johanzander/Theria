import { describe, expect, it } from 'vitest'
import { dailyRows, dayTotals, hourlyRows, summaryRows, type EnergyDay } from './energyForecast'

const day = (name: string, actual: (number | null)[]): EnergyDay => ({
  day: name,
  weather: 'forecast',
  clock: [0, 1],
  forecasts: { simple: [1, 2], mean_48h: [1.5, 1.5], profile_7d: [1, 1] },
  actual,
})

describe('energy forecast', () => {
  it('gives one row per clock hour', () => {
    expect(hourlyRows(day('2026-11-10', [0.5, null]))).toEqual([
      { hour: '00', actual: 0.5, simple: 1, mean_48h: 1.5, profile_7d: 1 },
      { hour: '01', actual: null, simple: 2, mean_48h: 1.5, profile_7d: 1 },
    ])
  })

  it('has no measured total until every hour is measured', () => {
    expect(dayTotals(day('2026-11-10', [1, null]))).toEqual({ forecast: 3, actual: null })
    expect(dayTotals(day('2026-11-10', [1, 1]))).toEqual({ forecast: 3, actual: 2 })
  })

  it('gives one row per day with the forecast and what was measured', () => {
    expect(dailyRows([day('2026-11-09', [1, 1]), day('2026-11-10', [1, null])])).toEqual([
      { day: '11-09', forecast: 3, actual: 2 },
      { day: '11-10', forecast: 3, actual: null },
    ])
  })

  it('names the methods and the weather, with the model first', () => {
    const rows = summaryRows([
      { method: 'profile_7d', weather: 'measured', days: 3, day_error_kwh: 9, bias_kwh: 1, share: 0.2 },
      { method: 'simple', weather: 'measured', days: 3, day_error_kwh: 4, bias_kwh: 0.5, share: 0.1 },
    ])
    expect(rows.map((r) => r.label)).toEqual(['Model', 'Last 7 days by hour'])
    expect(rows[0].weatherLabel).toBe('Weather as it came')
  })
})
