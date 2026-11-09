import { describe, expect, it } from 'vitest'
import { formatDifference, overallAvailability, rowsToSchedule, scheduleToRows, warnings, type RadiatorPrefs, type RadiatorSnapshot } from './radiators'

const prefs = (over: Partial<RadiatorPrefs['salon_schedule']> = {}): RadiatorPrefs => ({
  salon_schedule: {
    mon: { open: '09:00', close: '19:00' }, tue: { open: '09:00', close: '19:00' }, wed: { open: '09:00', close: '19:00' },
    thu: { open: '09:00', close: '19:00' }, fri: { open: '09:00', close: '19:00' },
    sat: { open: '10:00', close: '16:00' }, sun: { open: '10:00', close: '16:00' }, ...over,
  },
  salon_open_target_c: 20, salon_open_floor_c: 19, salon_closed_floor_c: 17, salon_preheat_c: 21, salon_margin_c: 0.3, salon_max_setpoint_c: 22,
  guest_band_c: 1.5, guest_rooms: [], mode: 'shadow',
})

describe('schedule rows', () => {
  it('round-trips weekdays and weekend', () => {
    const rows = scheduleToRows(prefs())
    expect(rows.map((r) => r.day)).toEqual(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])
    expect(rows[5]).toMatchObject({ open: '10:00', close: '16:00' })
    expect(rowsToSchedule(rows)).toEqual({ schedule: prefs().salon_schedule })
  })

  it('a closed day is null both ways', () => {
    const rows = scheduleToRows(prefs({ sun: null }))
    expect(rows[6]).toMatchObject({ open: null, close: null })
    expect(rowsToSchedule(rows)).toEqual({ schedule: prefs({ sun: null }).salon_schedule })
  })

  it('open must be before close', () => {
    const rows = scheduleToRows(prefs())
    rows[0] = { day: 'mon', open: '19:00', close: '09:00' }
    expect(rowsToSchedule(rows).error).toMatch(/mon.*before/)
  })

  it('a day with only one time is an error', () => {
    const rows = scheduleToRows(prefs())
    rows[1] = { day: 'tue', open: '09:00', close: null }
    expect(rowsToSchedule(rows).error).toMatch(/tue/)
  })
})

describe('formatDifference', () => {
  it('shows sign and equality', () => {
    expect(formatDifference(19, 20.5)).toBe('+1.5')
    expect(formatDifference(20, 19.5)).toBe('-0.5')
    expect(formatDifference(20, 20)).toBe('=')
    expect(formatDifference(null, 20)).toBe('-')
  })
})

describe('overallAvailability', () => {
  it('is the mean share of readable cycles', () => {
    expect(overallAvailability({ a: 1, b: 0.5 })).toBe(0.75)
    expect(overallAvailability({})).toBe(0)
  })
})

describe('warnings', () => {
  const snap = (over: Partial<RadiatorSnapshot>) => ({ age_seconds: 60, last_error: null, outdoor_stale: false, left_alone: [] as string[], ...over }) as RadiatorSnapshot

  it('is empty for a fresh plan', () => expect(warnings(snap({}))).toEqual([]))

  it('names a failed cycle instead of the age', () => {
    const w = warnings(snap({ last_error: 'SensorError: outdoor temperature unavailable', age_seconds: 99999 }))
    expect(w).toHaveLength(1)
    expect(w[0]).toMatch(/outdoor temperature unavailable/)
  })

  it('warns about an old plan, a stale outdoor value and rooms left alone', () => {
    const w = warnings(snap({ age_seconds: 3600, outdoor_stale: true, left_alone: ['climate.kitchen'] }))
    expect(w.join(' ')).toMatch(/60 minutes old/)
    expect(w.join(' ')).toMatch(/outdoor sensor/)
    expect(w.join(' ')).toMatch(/climate.kitchen/)
  })
})
