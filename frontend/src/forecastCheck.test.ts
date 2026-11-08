import { describe, expect, it } from 'vitest'
import { forecastCheckTable, type ForecastCheck } from './forecastCheck'

const check: ForecastCheck = {
  since: '2026-11-01T00:00:00Z',
  hours: 48,
  rows: [
    { quantity: 'room_c', unit: '°C', lead: '6-12 h', n: 30, bias: 0.31, mae: 0.42 },
    { quantity: 'room_c', unit: '°C', lead: '0-1 h', n: 40, bias: -0.05, mae: 0.1 },
    { quantity: 't_out_c', unit: '°C', lead: '12-24 h', n: 20, bias: 1.2, mae: 1.5 },
  ],
  missing: { room_c: 2, energy_kwh: 0, rts_c: 0, flow_c: 0, t_out_c: 0, sun_wm2: 0 },
}

describe('forecastCheckTable', () => {
  it('puts each quantity on a row and each lead in a column, nearest lead first', () => {
    const table = forecastCheckTable(check)
    expect(table.leads).toEqual(['0-1 h', '1-3 h', '3-6 h', '6-12 h', '12-24 h', '24-48 h'])
    const room = table.rows.find((r) => r.quantity === 'room_c')!
    expect(room.cells[0]).toEqual({ n: 40, bias: -0.05, mae: 0.1 })
    expect(room.cells[1]).toBeNull()
    expect(room.cells[3]).toEqual({ n: 30, bias: 0.31, mae: 0.42 })
    expect(room.missing).toBe(2)
  })

  it('keeps a quantity without any comparison yet, so its missing hours show', () => {
    const rows = forecastCheckTable(check).rows.map((r) => r.quantity)
    expect(rows).toEqual(['room_c', 'energy_kwh', 'rts_c', 'flow_c', 't_out_c', 'sun_wm2'])
  })
})
