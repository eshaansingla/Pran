import { useState, useEffect } from 'react'
import { Activity, TrendingUp, Info, AlertTriangle } from 'lucide-react'
import type { ActiveTab } from './types'
import Dashboard from './pages/Dashboard'
import Forecasting from './pages/Forecasting'
import ModelInfoPage from './pages/ModelInfo'

interface TabDef {
  id: ActiveTab
  label: string
  Icon: typeof Activity
  badge?: string
}

const TABS: TabDef[] = [
  { id: 'dashboard',   label: 'ICP Classification',   Icon: Activity },
  { id: 'forecasting', label: 'ICP Forecasting',       Icon: TrendingUp, badge: 'v2.0' },
  { id: 'model',       label: 'Model Info',             Icon: Info },
]

export default function App() {
  const [tab, setTab] = useState<ActiveTab>('dashboard')
  const now = new Date().toLocaleString('en-GB', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  }).replace(',', '')

  // Keep timestamp updated
  const [time, setTime] = useState(now)
  useEffect(() => {
    const id = setInterval(() => {
      setTime(new Date().toLocaleString('en-GB', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false,
      }).replace(',', ''))
    }, 1000)
    return () => clearInterval(id)
  }, [])

  // Keyboard shortcut: Ctrl+E = export (handled inside Dashboard)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === '1') { e.preventDefault(); setTab('dashboard') }
      if ((e.ctrlKey || e.metaKey) && e.key === '2') { e.preventDefault(); setTab('forecasting') }
      if ((e.ctrlKey || e.metaKey) && e.key === '3') { e.preventDefault(); setTab('model') }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <div className="min-h-screen bg-clinical-background flex flex-col">
      {/* Top navigation bar */}
      <header className="bg-clinical-primary text-white shadow-md" role="banner">
        <div className="max-w-screen-xl mx-auto px-5 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity size={20} strokeWidth={2} aria-hidden="true" />
            <div>
              <span className="font-semibold text-sm tracking-tight">ICP Monitor</span>
              <span className="ml-2 text-xs text-blue-200 font-normal">
                Clinical Decision Support
              </span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span
              className="text-xs font-mono text-blue-200 tabular-nums"
              aria-label="Current timestamp"
              aria-live="polite"
            >
              {time}
            </span>
            <span className="text-xs bg-amber-500 text-white px-2 py-0.5 rounded font-medium">
              RESEARCH PROTOTYPE
            </span>
          </div>
        </div>
      </header>

      {/* Tab bar */}
      <nav
        className="bg-white border-b border-clinical-border shadow-sm"
        role="navigation"
        aria-label="Main navigation"
      >
        <div className="max-w-screen-xl mx-auto px-5">
          <div className="flex" role="tablist">
            {TABS.map(({ id, label, Icon, badge }) => (
              <button
                key={id}
                role="tab"
                aria-selected={tab === id}
                aria-controls={`panel-${id}`}
                onClick={() => setTab(id)}
                className={`
                  relative flex items-center gap-2 px-5 py-3.5 text-sm font-medium border-b-2
                  focus:outline-none focus-visible:ring-2 focus-visible:ring-clinical-primary focus-visible:ring-inset
                  transition-colors duration-100
                  ${tab === id
                    ? 'border-clinical-primary text-clinical-primary'
                    : 'border-transparent text-clinical-text-secondary hover:text-clinical-text-primary hover:border-gray-300'}
                `}
              >
                <Icon size={15} aria-hidden="true" />
                {label}
                {badge && (
                  <span className="ml-1 text-2xs bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded font-medium">
                    {badge}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5" role="main">
        <div
          id="panel-dashboard"
          role="tabpanel"
          aria-labelledby="tab-dashboard"
          hidden={tab !== 'dashboard'}
        >
          {tab === 'dashboard' && <Dashboard />}
        </div>
        <div
          id="panel-forecasting"
          role="tabpanel"
          aria-labelledby="tab-forecasting"
          hidden={tab !== 'forecasting'}
        >
          {tab === 'forecasting' && <Forecasting />}
        </div>
        <div
          id="panel-model"
          role="tabpanel"
          aria-labelledby="tab-model"
          hidden={tab !== 'model'}
        >
          {tab === 'model' && <ModelInfoPage />}
        </div>
      </main>

      {/* Footer disclaimer */}
      <footer
        className="border-t border-clinical-border bg-white py-3"
        role="contentinfo"
      >
        <div className="max-w-screen-xl mx-auto px-5 flex items-center gap-2 text-xs text-clinical-text-muted">
          <AlertTriangle size={12} className="text-amber-500 flex-shrink-0" aria-hidden="true" />
          <span>
            <strong>Disclaimer:</strong> This is a clinical decision support tool.
            Not FDA-approved. Not for autonomous diagnostic use.
            All decisions must be made by qualified clinicians.
            XGBoost v1.0 — Macro F1 = 0.77 — Trained 2026-04-03
          </span>
        </div>
      </footer>
    </div>
  )
}
