# Period Selection Analysis & Best Practices

## Current Issues

### Inconsistency Between Methods

**getHoursFromRange()** (date-picker.js:233-239)
```javascript
'today': 48,      // 2 days worth of data to fetch
'yesterday': 48,  // 2 days worth of data to fetch
'week': 168,      // 7 days worth of data to fetch
'month': 744      // 31 days worth of data to fetch
```

**buildChartDataUrl()** (date-picker.js:192-223)
- **Today**: 00:00 today → 23:59 today (1 day) ✅ Consistent
- **Yesterday**: 00:00 yesterday → 23:59 yesterday (1 day) ✅ Consistent
- **Week**: 00:00 Monday → 23:59 today ❌ VARIABLE (1-7 days)
- **Month**: 00:00 1st → 23:59 today ❌ VARIABLE (1-31 days)

**getTimeBounds()** (date-picker.js:263-300) - Same logic as buildChartDataUrl()

### The Problem

If today is **Monday Dec 29**:
- "Week" shows: Mon Dec 29 only (1 day)
- But getHoursFromRange returns 168 hours (7 days of data fetched)

If today is **Sunday Dec 29**:
- "Week" shows: Mon Dec 23 - Sun Dec 29 (7 days)
- And getHoursFromRange returns 168 hours (7 days of data fetched)

**This is inconsistent user experience!**

## Best Practice Recommendation

### Option 1: Rolling Windows (RECOMMENDED)

**Pros:**
- Consistent data range regardless of calendar position
- Predictable behavior
- Common in analytics tools (Grafana, Google Analytics)

**Implementation:**
- **Today**: Today 00:00 → Today 23:59 (1 full day)
- **Yesterday**: Yesterday 00:00 → Yesterday 23:59 (1 full day)
- **Week**: 7 days ago 00:00 → Today 23:59 (7 full days)
- **Month**: 30 days ago 00:00 → Today 23:59 (30 full days)

### Option 2: Calendar-Based (Current, needs fixing)

**Pros:**
- Aligns with calendar weeks/months
- Easier to understand for some users

**Cons:**
- Variable data range (1-7 days for week, 1-31 days for month)
- Inconsistent with getHoursFromRange()

**Implementation (if keeping calendar-based):**
- **Week**: Monday 00:00 → Sunday 23:59 (always 7 days)
- **Month**: 1st 00:00 → Last day 23:59 (always full month)

But this means showing **future dates** which doesn't make sense for historical data.

## Recommended Changes

### 1. Update buildChartDataUrl() to use rolling windows:

```javascript
} else if (this.selectedTimeRange === 'week') {
    // Last 7 complete days
    const weekStart = new Date(now);
    weekStart.setDate(weekStart.getDate() - 6);  // 6 days ago + today = 7 days
    weekStart.setHours(0, 0, 0, 0);
    const todayEnd = new Date(now);
    todayEnd.setHours(23, 59, 59, 999);
    fromMs = weekStart.getTime();
    toMs = todayEnd.getTime();
} else if (this.selectedTimeRange === 'month') {
    // Last 30 complete days
    const monthStart = new Date(now);
    monthStart.setDate(monthStart.getDate() - 29);  // 29 days ago + today = 30 days
    monthStart.setHours(0, 0, 0, 0);
    const todayEnd = new Date(now);
    todayEnd.setHours(23, 59, 59, 999);
    fromMs = monthStart.getTime();
    toMs = todayEnd.getTime();
}
```

### 2. Update getTimeBounds() to match:

```javascript
} else if (this.selectedTimeRange === 'week') {
    // Last 7 complete days
    const weekStart = new Date(now);
    weekStart.setDate(weekStart.getDate() - 6);
    weekStart.setHours(0, 0, 0, 0);
    const todayEnd = new Date(now);
    todayEnd.setHours(23, 59, 59, 999);
    xAxisMin = weekStart.toISOString();
    xAxisMax = todayEnd.toISOString();
} else if (this.selectedTimeRange === 'month') {
    // Last 30 complete days
    const monthStart = new Date(now);
    monthStart.setDate(monthStart.getDate() - 29);
    monthStart.setHours(0, 0, 0, 0);
    const todayEnd = new Date(now);
    todayEnd.setHours(23, 59, 59, 999);
    xAxisMin = monthStart.toISOString();
    xAxisMax = todayEnd.toISOString();
}
```

### 3. Update getHoursFromRange() to match:

```javascript
const rangeMap = {
    'today': 48,      // Fetch 2 days to ensure complete data
    'yesterday': 48,  // Fetch 2 days to ensure complete data
    'week': 168,      // 7 days * 24 hours = 168 ✅ Already correct!
    'month': 720      // 30 days * 24 hours = 720 (was 744 for 31 days)
};
```

## Summary

**Current behavior**: Calendar-based but incomplete (shows "week so far", "month so far")
**Recommended**: Rolling windows (always show last N complete days)

**Benefits**:
1. ✅ Consistent data range every time
2. ✅ All three methods (buildChartDataUrl, getTimeBounds, getHoursFromRange) aligned
3. ✅ All charts on page show identical x-axis ranges
4. ✅ Predictable user experience

**User sees**:
- Week button: Always 7 full days (today + 6 days back)
- Month button: Always 30 full days (today + 29 days back)
