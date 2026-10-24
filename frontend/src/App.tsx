import { useEffect, useState } from 'react'
import { tabHref } from './model'
import { Control } from './views/Control'
import { Models } from './views/Models'
import { Overview } from './views/Overview'
import { PlanTable } from './views/PlanTable'
import { Savings } from './views/Savings'
import { Sensors } from './views/Sensors'
import { Settings } from './views/Settings'

// The page is the start page but is resolved against <root>/static/plan/ (see the base tag the
// backend injects); the old pages are at <root>/legacy, /zone-insights and /system-insights
const OLD_PAGES = [
  { href: '../../legacy', label: 'Old dashboard' },
  { href: '../../zone-insights', label: 'Zone insights' },
  { href: '../../system-insights', label: 'System insights' },
]

const VIEWS = [
  { id: 'overview', label: 'Overview', view: <Overview /> },
  { id: 'plan', label: 'Plan', view: <PlanTable /> },
  { id: 'savings', label: 'Savings', view: <Savings /> },
  { id: 'models', label: 'Models', view: <Models /> },
  { id: 'settings', label: 'Settings', view: <Settings /> },
  { id: 'sensors', label: 'Sensors', view: <Sensors /> },
  { id: 'control', label: 'Control', view: <Control /> },
]

function currentId(): string {
  const id = window.location.hash.slice(1)
  return VIEWS.some((v) => v.id === id) ? id : 'overview'
}

export function App() {
  const [id, setId] = useState(currentId)
  useEffect(() => {
    const onHash = () => setId(currentId())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const active = VIEWS.find((v) => v.id === id) ?? VIEWS[0]
  return (
    <main>
      <h1>Theria - price and plan</h1>
      <nav className="tabs">
        {VIEWS.map((v) => (
          <a key={v.id} href={tabHref(v.id, window.location)} className={v.id === id ? 'current' : undefined}>
            {v.label}
          </a>
        ))}
        <span className="spacer" />
        {OLD_PAGES.map((p) => (
          <a key={p.href} className="old" href={import.meta.env.DEV ? '#' : p.href}>
            {p.label}
          </a>
        ))}
      </nav>
      {active.view}
    </main>
  )
}
