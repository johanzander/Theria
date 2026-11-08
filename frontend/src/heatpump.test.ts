import { describe, expect, it } from 'vitest'
import { HOUR, MAX_RANGE, niceTicks, pan, resolveRange, runs, timeTicks, valueAt, yRange, zoomOut } from './heatpump'

const NOW = Date.UTC(2026, 10, 8, 12)

describe('the chosen range', () => {
  it('a rolling range ends now', () => {
    expect(resolveRange({ kind: 'last', hours: 24 }, NOW)).toEqual({ from: NOW - 24 * HOUR, to: NOW })
  })

  it('a fixed range stays where it is', () => {
    expect(resolveRange({ kind: 'fixed', from: NOW - 5 * HOUR, to: NOW - 4 * HOUR }, NOW)).toEqual({ from: NOW - 5 * HOUR, to: NOW - 4 * HOUR })
  })

  it('stepping forward stops at now and keeps the length', () => {
    expect(pan({ from: NOW - 5 * HOUR, to: NOW - HOUR }, 1, NOW)).toEqual({ from: NOW - 4 * HOUR, to: NOW })
  })

  it('stepping back moves half the range', () => {
    expect(pan({ from: NOW - 4 * HOUR, to: NOW }, -1, NOW)).toEqual({ from: NOW - 6 * HOUR, to: NOW - 2 * HOUR })
  })

  it('zooming out doubles the range around its middle without passing now', () => {
    expect(zoomOut({ from: NOW - 4 * HOUR, to: NOW - 2 * HOUR }, NOW)).toEqual({ from: NOW - 5 * HOUR, to: NOW - HOUR })
    expect(zoomOut({ from: NOW - 2 * HOUR, to: NOW }, NOW).to).toBe(NOW)
  })

  it('zooming out stops at the longest range the add-on answers', () => {
    const out = zoomOut({ from: NOW - MAX_RANGE, to: NOW }, NOW)
    expect(out.to - out.from).toBeLessThanOrEqual(MAX_RANGE)
  })
})

describe('drawing', () => {
  it('a gap in a series breaks the line', () => {
    expect(runs([1, 2, 3, 4], [10, null, 30, 40], false)).toEqual([[[1, 10]], [[3, 30], [4, 40]]])
  })

  it('a stepped series holds its value until the next reading', () => {
    expect(runs([1, 2], [10, 20], true)).toEqual([[[1, 10], [2, 10], [2, 20]]])
  })

  it('the y axis covers the values with some room, ignoring gaps', () => {
    const [low, high] = yRange([[null, 10, 20]])
    expect(low).toBeLessThan(10)
    expect(high).toBeGreaterThan(20)
  })

  it('an empty series still gives a drawable axis', () => {
    const [low, high] = yRange([[null, null]])
    expect(high).toBeGreaterThan(low)
  })

  it('ticks fall on round values inside the range', () => {
    expect(niceTicks(0, 100, 4)).toEqual([0, 20, 40, 60, 80, 100])
  })

  it('time ticks are spaced so their labels fit', () => {
    const ticks = timeTicks(NOW - 24 * HOUR, NOW, 1000)
    expect(ticks.length).toBeGreaterThan(3)
    expect(ticks.length).toBeLessThanOrEqual(15)
    expect(ticks[1] - ticks[0]).toBe(ticks[2] - ticks[1])
  })

  it('the value at a moment is the one that bucket holds', () => {
    expect(valueAt([100, 200, 300], [1, 2, 3], 250)).toBe(2)
    expect(valueAt([100, 200, 300], [1, 2, 3], 50)).toBeNull()
  })
})
