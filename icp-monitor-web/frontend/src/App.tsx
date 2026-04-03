import { useState, useEffect } from 'react'
import { Activity, TrendingUp, Info, AlertTriangle, Sun, Moon, Keyboard } from 'lucide-react'
import { Toaster } from 'react-hot-toast'
import type { ActiveTab } from './types'
import Dashboard from './pages/Dashboard'
import Forecasting from './pages/Forecasting'
import ModelInfoPage from './pages/ModelInfo'
import KeyboardHelp from './components/KeyboardHelp'
import { useStore } from './store/useStore'

interface TabDef {
  id: ActiveTab
  label: string
  Icon: typeof Activity
  badge?: string
}

const TABS: TabDef[] = [
  { id: 'dashboard',   label: 'ICP Classification', Icon: Activity },
  { id: 'forecasting', label: 'ICP Forecasting',     Icon: TrendingUp, badge: 'v2.0' },
  { id: 'model',       label: 'Model Info',           Icon: Info },
]

export default function App() {
  const { isDark, toggleDark, setDark } = useStore()
  const [tab, setTab]           = useState<ActiveTab>('dashboard')
  const [time, setTime]         = useState('')
  const [showHelp, setShowHelp] = useState(false)

  // Apply dark class on mount + when isDark changes
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark)
  }, [isDark])

  // Init dark from localStorage if Zustand hasn't restored it yet
  useEffect(() => {
    const stored = localStorage.getItem('icp-monitor-store')
    if (!stored) {
      setDark(window.matchMedia('(prefers-color-scheme: dark)').matches)
    }
  }, [])

  // Live clock
  useEffect(() => {
    const fmt = () => new Date().toLocaleString('en-GB', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: false,
    }).replace(',', '')
    setTime(fmt())
    const id = setInterval(() => setTime(fmt()), 1000)
    return () => clearInterval(id)
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return
      if (e.key === '1') { e.preventDefault(); setTab('dashboard') }
      if (e.key === '2') { e.preventDefault(); setTab('forecasting') }
      if (e.key === '3') { e.preventDefault(); setTab('model') }
      if (e.key === 'd') { e.preventDefault(); toggleDark() }
      if (e.key === 'h') { e.preventDefault(); setShowHelp(true) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [toggleDark])

  return (
    <div className="min-h-screen bg-clinical-background dark:bg-slate-900 flex flex-col transition-colors duration-200">
      {/* Skip to content */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-clinical-primary focus:text-white focus:rounded"
      >
        Skip to content
      </a>

      {/* Top navigation */}
      <header className="bg-clinical-primary dark:bg-slate-800 text-white shadow-md transition-colors duration-200" role="banner">
        <div className="max-w-screen-xl mx-auto px-5 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity size={20} strokeWidth={2} aria-hidden="true" />
            <div>
              <span className="font-semibold text-sm tracking-tight">ICP Monitor</span>
              <span className="ml-2 text-xs text-blue-200 dark:text-slate-400 font-normal">
                Clinical Decision Support
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span
              className="text-xs font-mono text-blue-200 dark:text-slate-400 tabular-nums hidden sm:block"
              aria-label="Current timestamp"
              aria-live="polite"
            >
              {time}
            </span>
            <span className="text-xs bg-amber-500 text-white px-2 py-0.5 rounded font-medium hidden sm:inline">
              RESEARCH
            </span>
            <button
              onClick={() => setShowHelp(true)}
              aria-label="Keyboard shortcuts (Ctrl+H)"
              title="Keyboard shortcuts (Ctrl+H)"
              className="p-2 rounded-lg text-blue-200 dark:text-slate-400 hover:bg-white/10 dark:hover:bg-slate-700 transition-colors"
            >
              <Keyboard size={16} />
            </button>
            <button
              onClick={toggleDark}
              aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              title={isDark ? 'Light mode (Ctrl+D)' : 'Dark mode (Ctrl+D)'}
              className="p-2 rounded-lg text-blue-200 dark:text-slate-400 hover:bg-white/10 dark:hover:bg-slate-700 transition-colors"
            >
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </div>
      </header>

      {/* Tab bar */}
      <nav
        className="bg-white dark:bg-slate-800 border-b border-clinical-border dark:border-slate-700 shadow-sm transition-colors duration-200"
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
                    ? 'border-clinical-primary dark:border-blue-400 text-clinical-primary dark:text-blue-400'
                    : 'border-transparent text-clinical-text-secondary dark:text-slate-400 hover:text-clinical-text-primary dark:hover:text-slate-200 hover:border-gray-300 dark:hover:border-slate-600'}
                `}
              >
                <Icon size={15} aria-hidden="true" />
                {label}
                {badge && (
                  <span className="ml-1 text-2xs bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400 px-1.5 py-0.5 rounded font-medium">
                    {badge}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main
        id="main-content"
        className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5"
        role="main"
      >
        <div id="panel-dashboard" role="tabpanel" hidden={tab !== 'dashboard'}>
          {tab === 'dashboard' && <Dashboard />}
        </div>
        <div id="panel-forecasting" role="tabpanel" hidden={tab !== 'forecasting'}>
          {tab === 'forecasting' && <Forecasting />}
        </div>
        <div id="panel-model" role="tabpanel" hidden={tab !== 'model'}>
          {tab === 'model' && <ModelInfoPage />}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-clinical-border dark:border-slate-700 bg-white dark:bg-slate-800 py-3 transition-colors duration-200" role="contentinfo">
        <div className="max-w-screen-xl mx-auto px-5 flex items-center gap-2 text-xs text-clinical-text-muted dark:text-slate-400">
          <AlertTriangle size={12} className="text-amber-500 flex-shrink-0" aria-hidden="true" />
          <span>
            <strong className="text-clinical-text-secondary dark:text-slate-300">Disclaimer:</strong>{' '}
            Clinical decision support tool. Not FDA-approved. Not for autonomous diagnostic use.
            All decisions must be made by qualified clinicians.
            XGBoost v2.0 — F1 = 0.88, AUC = 0.96 — Binary (Normal / Abnormal) — Trained 2026-04-03
          </span>
        </div>
      </footer>

      {/* Keyboard help overlay */}
      {showHelp && <KeyboardHelp onClose={() => setShowHelp(false)} />}

      {/* Toast notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            fontSize: '13px',
            borderRadius: '8px',
            background: isDark ? '#2D3748' : '#fff',
            color: isDark ? '#E2E8F0' : '#1A202C',
            border: isDark ? '1px solid #4A5568' : '1px solid #E2E8F0',
          },
        }}
      />
    </div>
  )
}
