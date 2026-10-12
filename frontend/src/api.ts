import type { DayView, Execution, PlanResponse, SavingsResponse } from './model'

// Served at <ingress>/static/plan/, so the API is two levels up; the dev server proxies /api.
const BASE = import.meta.env.DEV ? '/api/' : '../../api/'

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

export async function refitModels(): Promise<string | null> {
  const response = await fetch(BASE + 'model/refit', { method: 'POST' })
  if (response.ok) return null
  const body = (await response.json()) as { detail?: string }
  return body.detail ?? `HTTP ${response.status}`
}
