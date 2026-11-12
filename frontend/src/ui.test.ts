import { describe, expect, it } from 'vitest'
import { keptPick, shownDay } from './ui'

describe('the day a page shows', () => {
  it('follows today past midnight when no other day was picked', () => {
    const pick = keptPick('2026-11-11', '2026-11-11')
    expect(shownDay(pick, '2026-11-11')).toBe('2026-11-11')
    expect(shownDay(pick, '2026-11-12')).toBe('2026-11-12')
  })

  it('keeps a picked earlier day past midnight', () => {
    const pick = keptPick('2026-11-08', '2026-11-11')
    expect(shownDay(pick, '2026-11-12')).toBe('2026-11-08')
  })

  it('follows today again after the Today button', () => {
    const pick = keptPick('2026-11-11', '2026-11-11')
    expect(shownDay(pick, '2026-11-12')).toBe('2026-11-12')
  })
})
