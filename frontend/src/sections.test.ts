import { describe, expect, it } from 'vitest'
import { roomColour, sectionColour, sectionRows, type Section, type SectionHour } from './model'
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
