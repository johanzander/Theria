import type { Recommendation, DayView, Execution, ModelEntry, PlanResponse, SavingsResponse, SettingEntry } from './model'

// Served at <ingress>/static/plan/, so the API is two levels up; the dev server proxies /api.
const BASE = import.meta.env.DEV ? '/api/' : '../../api/'

/** The debug export: a markdown file with everything needed to understand what Theria did. */
export const debugExportUrl = BASE + 'export-debug-data'

async function get<T>(path: string): Promise<T | null> {
  const response = await fetch(BASE + path)
  if (response.status === 404 || response.status === 409) return null
  if (!response.ok) throw new Error(`${path}: ${response.status}`)
  return (await response.json()) as T
}

export const fetchPlan = () => get<PlanResponse>('plan')
export const fetchDay = (day?: string) => get<DayView>(day ? `plan/day?day=${day}` : 'plan/day')
export const fetchExecution = () => get<Execution>('plan/execution')
export const fetchSavings = (days: number) => get<SavingsResponse>(`savings?days=${days}`)

import type { Timeline } from './heatpump'

export const fetchTimeline = (from: number, to: number) =>
  get<Timeline>(`heatpump/timeline?start=${new Date(from).toISOString()}&stop=${new Date(to).toISOString()}`)

import type { ComfortResponse, LogEntry, ModelHealth, ParameterHistory, Preflight } from './model'

export const fetchModelHealth = () => get<ModelHealth>('model/health')
export const fetchComfort = () => get<ComfortResponse>('comfort-sensors')

export async function setMode(mode: string, confirm: boolean): Promise<string | null> {
  const response = await fetch(BASE + 'plan/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, confirm }),
  })
  if (response.ok) return null
  const body = (await response.json()) as { detail?: string }
  return body.detail ?? `HTTP ${response.status}`
}

export const fetchPreflight = () => get<Preflight>('plan/preflight')
export const fetchLog = () => get<LogEntry[]>('plan/log')
export const fetchParameters = () => get<ParameterHistory>('plan/parameters?limit=1000')

import type { ForecastCheck } from './forecastCheck'

export const fetchForecastCheck = (days: number) => get<ForecastCheck>(`plan/forecast-check?days=${days}`)

export async function refitModels(): Promise<string | null> {
  const response = await fetch(BASE + 'model/refit', { method: 'POST' })
  if (response.ok) return null
  const body = (await response.json()) as { detail?: string }
  return body.detail ?? `HTTP ${response.status}`
}

export const fetchTuning = () => get<Recommendation[]>('plan/tuning')

export const fetchModelCatalog = () => get<ModelEntry[]>('plan/model')
export const fetchSettings = () => get<SettingEntry[]>('plan/settings')

async function change(path: string, init: RequestInit): Promise<{ settings: SettingEntry[] | null; error: string | null }> {
  const response = await fetch(BASE + path, init)
  if (response.ok) return { settings: (await response.json()) as SettingEntry[], error: null }
  const body = (await response.json()) as { detail?: string }
  return { settings: null, error: body.detail ?? `HTTP ${response.status}` }
}

export const saveSettings = (values: Record<string, number | boolean>) =>
  change('plan/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) })

export const resetSetting = (key: string) => change(`plan/settings/${key}`, { method: 'DELETE' })

import type { PrefsResponse, RadiatorPrefs, RadiatorSnapshot } from './radiators'

export const fetchRadiators = () => get<RadiatorSnapshot>('radiators/status')
export const fetchRadiatorPrefs = () => get<PrefsResponse>('radiators/prefs')

async function changePrefs(init: RequestInit): Promise<{ prefs: PrefsResponse | null; error: string | null }> {
  const response = await fetch(BASE + 'radiators/prefs', init)
  if (response.ok) return { prefs: (await response.json()) as PrefsResponse, error: null }
  const body = (await response.json()) as { detail?: string }
  return { prefs: null, error: body.detail ?? `HTTP ${response.status}` }
}

export const saveRadiatorPrefs = (values: Partial<RadiatorPrefs>) =>
  changePrefs({ method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) })

export const resetRadiatorPrefs = () => changePrefs({ method: 'DELETE' })
