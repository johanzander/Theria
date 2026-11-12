import { describe, expect, it } from 'vitest'
import { continueFromNow, lkTarget, offlineName, pairedTooltipRows, roomColour, sectionColour, sectionRows, UNITS, type Section, type SectionHour } from './model'
import { houseNote, withPlanZones } from './views/DayCharts'

const guest: Section = { id: 'guest', name: 'Gårdshus', rooms: ['Sovrum', 'Kök'], thermostats: ['Kök'] }

describe('room sections', () => {
  it('flattens a section for its chart and leaves an hour without data empty', () => {
    const hour: SectionHour = { rooms: { Sovrum: 18.7, Kök: 20.0 }, setpoints: { Kök: 20.0 }, demand: { Kök: 35 } }
    const [measured, empty] = sectionRows([{ sections: { guest: hour } }, { sections: {} }], guest)
    expect(measured).toMatchObject({ room_Sovrum: 18.7, room_Kök: 20.0, set_Kök: 20.0, demand_Kök: 35 })
    expect(empty).toMatchObject({ room_Sovrum: null, set_Kök: null, demand_Kök: null })
  })

  it('gives a thermostat the colour of its room', () => {
    expect(sectionColour(guest, 'Kök')).toBe(sectionColour(guest, 'Kök'))
    expect(sectionColour(guest, 'Kök')).not.toBe(sectionColour(guest, 'Sovrum'))
  })

  it('always shows the zones the plan moves', () => {
    expect(withPlanZones([]).map((s) => s.id)).toEqual(['salon', 'ground'])
    expect(withPlanZones([guest]).map((s) => s.id)).toEqual(['salon', 'ground', 'guest'])
    const configured = [{ ...guest, id: 'salon' }, { ...guest, id: 'ground' }, guest]
    expect(withPlanZones(configured)).toEqual(configured)
  })
})

describe('section colours', () => {
  it('keeps the LK rooms their colour on the ground floor only, and gives every room its own', () => {
    const ground: Section = { id: 'ground', name: 'Bottenvåning', rooms: ['Hallen', 'Bibliotek', 'Vardagsrum', 'Kök'], thermostats: [] }
    const gard: Section = { id: 'guest', name: 'Gårdshus', rooms: ['Hall', 'Vardagsrum', 'Kök'], thermostats: [] }
    expect(sectionColour(ground, 'Vardagsrum')).toBe(roomColour('vardagsrum'))
    for (const s of [ground, gard]) expect(new Set(s.rooms.map((n) => sectionColour(s, n))).size).toBe(s.rooms.length)
    expect(sectionColour(gard, 'Vardagsrum')).not.toBe(roomColour('vardagsrum'))
  })
})

describe('whole-house note', () => {
  it('states the fit, or says it is missing', () => {
    expect(houseNote({ w_per_k: 710.5, free_kw: 3.37, days: 162, rmse_kw: 1.13 })).toContain('711 W/K')
    expect(houseNote(null)).toContain('missing or was rejected')
  })
})

describe('paired tooltip', () => {
  it('shows a target in the row of its actual, and alone when the actual is missing', () => {
    const items = [
      { dataKey: 'room_Kök', name: 'Kök', value: 20.14, color: 'a' },
      { dataKey: 'set_Kök', name: 'Kök, target', value: 20, color: 'a' },
      { dataKey: 'set_Hall', name: 'Hall, target', value: 19.5, color: 'b' },
    ]
    const rows = pairedTooltipRows(items, { set_Kök: ['room_Kök'], set_Hall: ['room_Hall'] }, UNITS.temp)
    expect(rows.map((r) => [r.name, r.text])).toEqual([
      ['Kök', '20.1 °C (target 20.0 °C)'],
      ['Hall, target', '19.5 °C'],
    ])
  })
})

describe('LK room target', () => {
  it("is the room's own thermostat, or the plan's target for coming hours", () => {
    const thermostats = { lk_bibliotek: { room_c: 19.6, set_c: 19.5 } }
    expect(lkTarget({ past: true, thermostats, thermostatTarget: 19.0 }, 'Bibliotek')).toBe(19.5)
    expect(lkTarget({ past: true, thermostats, thermostatTarget: 19.0 }, 'Vardagsrum')).toBeNull()
    expect(lkTarget({ past: false, thermostats: {}, thermostatTarget: 19.0 }, 'Vardagsrum')).toBe(19.0)
  })
})

describe('offline room', () => {
  it('says so in its name, with its last report', () => {
    expect(offlineName('Kök', undefined)).toBe('Kök')
    expect(offlineName('Kök', null)).toBe('Kök (offline)')
    const today = new Date()
    today.setHours(19, 3, 0, 0)
    expect(offlineName('Väntsidan', today.toISOString())).toBe('Väntsidan (offline, last 19:03)')
    expect(offlineName('Väntsidan', '2026-01-02T08:00:00Z')).toContain('2026-01-02')
  })
})

describe('prediction from now', () => {
  it("starts at the room's value now and follows the zone's expected change", () => {
    const rows = [
      { current: false, room: 20.0, zone: 19.0 },
      { current: true, room: 21.0, zone: 19.5 },
      { current: false, room: null, zone: 19.0 },
      { current: false, room: null, zone: null },
    ]
    expect(continueFromNow(rows, (r) => r.room, (r) => r.zone)).toEqual([null, 21.0, 20.5, null])
  })

  it('pairs a target with the expected value when nothing is measured', () => {
    const items = [{ dataKey: 'pred_Kök', name: 'Kök, expected', value: 20.5, color: 'a' }, { dataKey: 'set_Kök', name: 'Kök, target', value: 20, color: 'a' }]
    expect(pairedTooltipRows(items, { set_Kök: ['room_Kök', 'pred_Kök'] }, UNITS.temp).map((r) => r.text)).toEqual(['20.5 °C (target 20.0 °C)'])
  })
})
