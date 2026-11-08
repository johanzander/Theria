/** Plan against reality (GET plan/forecast-check): every finished hour next to what plans made 0-48 hours earlier said. */

export interface ForecastCheckRow {
  quantity: string
  unit: string
  lead: string
  n: number
  bias: number // planned minus measured, mean
  mae: number // mean absolute error
}

export interface ForecastCheck {
  since: string
  hours: number
  rows: ForecastCheckRow[]
  missing: Record<string, number>
}

export type Cell = { n: number; bias: number; mae: number } | null

const LEADS = ['0-1 h', '1-3 h', '3-6 h', '6-12 h', '12-24 h', '24-48 h']

export const QUANTITY_LABELS: Record<string, { label: string; unit: string }> = {
  room_c: { label: 'Room at end of hour', unit: '°C' },
  energy_kwh: { label: 'Heat pump energy', unit: 'kWh' },
  rts_c: { label: 'Room setpoint', unit: '°C' },
  flow_c: { label: "Pump's flow target", unit: '°C' },
  t_out_c: { label: 'Outdoor temperature', unit: '°C' },
  sun_wm2: { label: 'Sun', unit: 'W/m²' },
}

export function forecastCheckTable(check: ForecastCheck) {
  const rows = Object.keys(QUANTITY_LABELS).map((quantity) => ({
    quantity,
    ...QUANTITY_LABELS[quantity],
    missing: check.missing[quantity] ?? 0,
    cells: LEADS.map((lead): Cell => {
      const row = check.rows.find((r) => r.quantity === quantity && r.lead === lead)
      return row ? { n: row.n, bias: row.bias, mae: row.mae } : null
    }),
  }))
  return { leads: LEADS, rows }
}
