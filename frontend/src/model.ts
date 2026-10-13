export interface HourPlan {
  start: string
  rts_c: number
  price_sek_per_kwh: number
  extreme: boolean
  floor_c: number
  flow_target_c: number
  energy_kwh: number
  room_c_end: number
  slab_c_end: number
  energy_cost_sek: number
  comfort_cost_sek: number
  tail: boolean
  reason: { binding?: string }
}

export interface ComfortOutlook {
  start: string
  room_c: number
  target_c: number
  deficit_c: number
  extra_rate_c_per_h: number
  lead_hours: number | null
}

export function outlookText(o: ComfortOutlook): { value: string; note: string } {
  const time = o.start.slice(11, 16)
  if (o.deficit_c < 0.05) return { value: `${o.room_c.toFixed(1)} °C`, note: `at the target when comfort starts (${time} UTC)` }
  const lead = o.lead_hours === null ? 'cannot be recovered' : `would need about ${Math.round(o.lead_hours)} h of maximum heating`
  return { value: `${o.room_c.toFixed(1)} °C`, note: `${o.deficit_c.toFixed(1)} °C under the target; ${lead}` }
}

export interface PlanResponse {
  computed_at: string
  mode: string
  age_seconds: number
  comfort_outlook: ComfortOutlook | null
  plan: {
    hours: HourPlan[]
    baseline_hours: HourPlan[]
    objective_sek: number
    baseline_objective_sek: number
    floor_violation_hours: number
  }
}

export interface DayOutcome {
  day: string
  coverage: number
  compressor_kwh: number
  aux_kwh: number
  actual_cost_sek: number
  baseline_cost_sek: number
  saving_sek: number
  mean_rts_c: number
  room_min_c: number | null
}

export interface SavingsResponse {
  days: DayOutcome[]
  total_saving_sek: number
  total_actual_cost_sek: number
  total_baseline_cost_sek: number
}

export interface Execution {
  mode: string
  stage: {
    index: number
    started_on: string
    max_deviation_c: number
    max_rise_per_hour_c: number
  }
  last_execution: { action: string; target_c: number | null; reason: string; at: string } | null
  last_lk: { action: string; target_c: number | null; reason: string; at: string } | null
  last_trouble: string | null
  last_stage_note: string | null
}

export interface PlanRow {
  label: string
  price: number
  rts: number
  baselineRts: number
  room: number
  baselineRoom: number
  floor: number
  extreme: boolean
  reason: string
}

export function planRows(response: PlanResponse): PlanRow[] {
  const { hours, baseline_hours } = response.plan
  return hours.map((h, i) => ({
    label: h.start.slice(11, 16),
    price: h.price_sek_per_kwh,
    rts: h.rts_c,
    baselineRts: baseline_hours[i].rts_c,
    room: h.room_c_end,
    baselineRoom: baseline_hours[i].room_c_end,
    floor: h.floor_c,
    extreme: h.extreme,
    reason: h.reason.binding ?? '',
  }))
}

export function savingsSummary(response: PlanResponse): { savingSek: number; percent: number } {
  const { objective_sek, baseline_objective_sek } = response.plan
  const savingSek = baseline_objective_sek - objective_sek
  return { savingSek, percent: baseline_objective_sek > 0 ? (savingSek / baseline_objective_sek) * 100 : 0 }
}

const REASONS: Record<string, string> = {
  baseline: 'Flat base setpoint (no optimisation)',
  floor: 'Held up by the comfort floor',
  ceiling: 'Held down by the comfort ceiling',
  flow_cap: 'Limited by the maximum flow temperature',
  rts_min: 'At the lowest allowed setpoint (price high enough to cut as far as allowed)',
  rts_max: 'At the highest allowed setpoint (price low enough to heat as much as allowed)',
  price_trade_off: 'Price against comfort: neither limit binds',
}

export function describeReason(binding: string): string {
  return REASONS[binding] ?? binding
}

export function formatAge(minutes: number): string {
  if (minutes < 60) return `${Math.round(minutes)} min`
  if (minutes < 60 * 48) return `${Math.round(minutes / 60)} h`
  return `${Math.round(minutes / 1440)} days`
}

export function fitQuality(modelRmse: number, persistenceRmse: number): string {
  return modelRmse < persistenceRmse * 0.95 ? 'better than assuming no change' : 'no better than assuming no change'
}

export interface ComfortSensor {
  name: string
  room: string
  temperature_entity: string
  status: string
  issues: { code: string; severity: string; message: string }[]
  value_c: number | null
  age_minutes: number | null
  battery_percent: number | null
  signal_dbm: number | null
  reference: boolean
  usable: boolean
}

export interface ComfortResponse {
  status: string
  checked_at: string
  sensors: ComfortSensor[]
  room_temperature_c: number | null
  room_temperature_note: string
}

export interface ModelHealth {
  mode: string
  models: {
    pump: {
      source: string
      params: Record<string, number>
      fit: { flow_rows: number; flow_rmse_c: number; energy_rows: number; energy_rmse_kwh: number } | null
    }
    room: {
      source: string
      fit: {
        response: {
          gain_c_per_c: number
          time_constant_days: number
          windows: number
          seasons: { season: number; windows: number; gain: number }[]
          cv_rmse_c: Record<string, number>
          cv_persistence_rmse_c: Record<string, number>
        }
      } | null
      fit_error: string | null
    }
    warmup: {
      source: string
      fit: WarmupFit | null
      fit_error: string | null
    }
    night: {
      source: string
      fit: NightFit | null
      fit_error: string | null
    }
    fitted_at: string | null
    last_fit_error: string | null
  }
  scores: { lead_h: number; n: number; mae_c: number; bias_c: number }[]
}

export interface PreflightCheck {
  name: string
  status: 'ok' | 'warn' | 'fail'
  detail: string
}

export interface Preflight {
  ok: boolean
  checks: PreflightCheck[]
}

export interface LogEntry {
  at: string
  kind: string
  action: string
  detail: string
}

export function statusTone(status: string): string {
  return status === 'ok' ? 'good' : status === 'warn' ? 'warn' : 'bad'
}

export interface WarmupFit {
  scale: number
  se: number
  applied_scale: number
  response_c_per_c: number
  predicted_c_per_c: number
  cooling_response_c_per_c: number
  cooling_scale: number
  cooling_se: number
  windows: number
  step_windows: number
}

export function warmupText(w: WarmupFit): string {
  const used = w.applied_scale === 1 ? 'not used (not different from the model)' : `used: the planner warms at ${w.applied_scale.toFixed(2)} x the model`
  return `Warming: ${w.scale.toFixed(2)} ± ${w.se.toFixed(2)} x the model, ${used}.`
}

export interface NightFit {
  intercept_c: number
  slope_c_per_c: number
  level_c_per_c: number
  level_ref_c: number
  night_hours: number
  nights: number
  rmse_c: number
  recent: { day: string; t_out_c: number; measured_change_c: number; predicted_change_c: number }[]
}

export interface ParameterRecord {
  at: string
  kind: string
  note: string
  params: Record<string, number>
}

export interface ParameterHistory {
  current: Record<string, number>
  history: ParameterRecord[]
}

export interface SeriesPoint {
  label: string
  value: number
}

export function parameterSeries(history: ParameterRecord[], key: string): SeriesPoint[] {
  return history
    .filter((r) => typeof r.params[key] === 'number')
    .map((r) => ({ label: r.at.slice(0, 10), value: r.params[key] }))
}

/**
 * Link to a tab of the page that is shown now. A plain "#id" would resolve against the
 * document's <base> (the static directory the page is served from) and load another document.
 */
export function tabHref(id: string, location: { pathname: string; search: string }): string {
  return `${location.pathname}${location.search}#${id}`
}

export interface DayHour {
  start: string
  past: boolean
  price_sek_per_kwh: number | null
  floor_c: number
  rts_c: number | null
  room_c: number | null
  baseline_rts_c: number | null
  baseline_room_c: number | null
  extreme: boolean
  loops_open: Record<string, number | null>
  thermostats: Record<string, { room_c: number | null; set_c: number | null }>
}

export interface DayView {
  day: string
  now: string
  loops_total: Record<string, number>
  hours: DayHour[]
}

export interface DayRow {
  label: string
  price: number | null
  floor: number
  measuredRts: number | null
  plannedRts: number | null
  baselineRts: number | null
  measuredRoom: number | null
  plannedRoom: number | null
  baselineRoom: number | null
  /** floor loops open per room, one entry per room with actuator sensors */
  loops: Record<string, number | null>
  thermostats: DayHour['thermostats']
  current: boolean
}

const clockLabel = (iso: string) =>
  new Date(iso).toLocaleTimeString('sv-SE', { hour: '2-digit', minute: '2-digit' })

/** One row per hour of the whole day; measured values before now, the plan from now on. */
export function dayRows(view: DayView): DayRow[] {
  return view.hours.map((h) => ({
    label: clockLabel(h.start),
    price: h.price_sek_per_kwh,
    floor: h.floor_c,
    measuredRts: h.past ? h.rts_c : null,
    plannedRts: h.past ? null : h.rts_c,
    baselineRts: h.past ? null : h.baseline_rts_c,
    measuredRoom: h.past ? h.room_c : null,
    plannedRoom: h.past ? null : h.room_c,
    baselineRoom: h.past ? null : h.baseline_room_c,
    loops: h.loops_open,
    thermostats: h.thermostats,
    current: h.start === view.now,
  }))
}

export interface EnergyRow {
  label: string
  flow: number
  baselineFlow: number
  energy: number
  baselineEnergy: number
  cost: number
  baselineCost: number
  cumulativeCost: number
  cumulativeBaselineCost: number
}

/** Per hour of the plan: flow temperature, energy and electricity cost, with the running totals. */
export function energyRows(response: PlanResponse): EnergyRow[] {
  const { hours, baseline_hours } = response.plan
  let total = 0
  let baselineTotal = 0
  return hours.map((h, i) => {
    const b = baseline_hours[i]
    total += h.energy_cost_sek
    baselineTotal += b.energy_cost_sek
    return {
      label: clockLabel(h.start),
      flow: h.flow_target_c,
      baselineFlow: b.flow_target_c,
      energy: h.energy_kwh,
      baselineEnergy: b.energy_kwh,
      cost: h.energy_cost_sek,
      baselineCost: b.energy_cost_sek,
      cumulativeCost: total,
      cumulativeBaselineCost: baselineTotal,
    }
  })
}

export interface Recommendation {
  key: string
  title: string
  current: string
  observed: string
  suggestion: string | null
  reason: string
  status: 'ok' | 'consider' | 'no data'
}

/** Settings with something to consider first, then the rest in the order given. */
export function sortedRecommendations(items: Recommendation[]): Recommendation[] {
  return [...items].sort((a, b) => Number(b.status === 'consider') - Number(a.status === 'consider'))
}
