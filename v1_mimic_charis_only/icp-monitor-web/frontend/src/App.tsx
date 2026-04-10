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
  badgeVariant?: 'amber' | 'blue' | 'green'
}

const TABS: TabDef[] = [
  { id: 'dashboard',   label: 'ICP Classification', Icon: Activity },
  { id: 'forecasting', label: 'ICP Forecasting',     Icon: TrendingUp, badge: 'LSTM',  badgeVariant: 'blue' },
  { id: 'model',       label: 'Model Info',           Icon: Info },
]

export default function App() {
  const { isDark, toggleDark, setDark } = useStore()
  const [tab, setTab]           = useState<ActiveTab>('dashboard')
  const [time, setTime]         = useState('')
  const [showHelp, setShowHelp] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark)
  }, [isDark])

  useEffect(() => {
    const stored = localStorage.getItem('icp-monitor-store')
    if (!stored) setDark(window.matchMedia('(prefers-color-scheme: dark)').matches)
  }, [])

  useEffect(() => {
    const fmt = () => new Date().toLocaleString('en-GB', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    }).replace(',', '')
    setTime(fmt())
    const id = setInterval(() => setTime(fmt()), 1000)
    return () => clearInterval(id)
  }, [])

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

  const badgeClass: Record<string, string> = {
    amber: 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400',
    blue:  'bg-blue-100  dark:bg-blue-900/40  text-blue-700  dark:text-blue-400',
    green: 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400',
  }

  return (
    <div className="min-h-screen bg-clinical-background dark:bg-slate-900 flex flex-col transition-colors duration-200">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-clinical-primary focus:text-white focus:rounded">
        Skip to content
      </a>

      {/* Header */}
      <header className="bg-gradient-to-r from-[#1e3a5f] to-[#2C5282] dark:from-slate-900 dark:to-slate-800 text-white shadow-lg transition-colors duration-200" role="banner">
        <div className="max-w-screen-xl mx-auto px-5 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-1.5 bg-white/10 rounded-lg">
              <Activity size={18} strokeWidth={2} aria-hidden="true" />
            </div>
            <div>
              <span className="font-bold text-sm tracking-tight">ICP Monitor</span>
              <span className="ml-2 text-xs text-blue-200 dark:text-slate-400 font-normal hidden sm:inline">
                Clinical Decision Support
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-blue-200/70 dark:text-slate-500 tabular-nums hidden lg:block">
              {time}
            </span>
            <span className="text-xs bg-amber-500/90 text-white px-2 py-0.5 rounded-md font-semibold tracking-wide hidden sm:inline">
              RESEARCH
            </span>
            <div className="h-4 w-px bg-white/20 mx-1 hidden sm:block" />
            <button
              onClick={() => setShowHelp(true)}
              aria-label="Keyboard shortcuts (Ctrl+H)"
              className="p-1.5 rounded-lg text-blue-200/80 hover:bg-white/10 hover:text-white transition-colors"
            >
              <Keyboard size={15} />
            </button>
            <button
              onClick={toggleDark}
              aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              className="p-1.5 rounded-lg text-blue-200/80 hover:bg-white/10 hover:text-white transition-colors"
            >
              {isDark ? <Sun size={15} /> : <Moon size={15} />}
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
            {TABS.map(({ id, label, Icon, badge, badgeVariant = 'amber' }) => (
              <button
                key={id}
                role="tab"
                aria-selected={tab === id}
                onClick={() => setTab(id)}
                className={`
                  relative flex items-center gap-2 px-5 py-3.5 text-sm font-medium
                  focus:outline-none focus-visible:ring-2 focus-visible:ring-clinical-primary focus-visible:ring-inset
                  transition-colors duration-150 group
                  ${tab === id
                    ? 'text-clinical-primary dark:text-blue-400'
                    : 'text-clinical-text-secondary dark:text-slate-400 hover:text-clinical-text-primary dark:hover:text-slate-200'}
                `}
              >
                <Icon size={14} aria-hidden="true" className="flex-shrink-0" />
                {label}
                {badge && (
                  <span className={`ml-0.5 text-2xs px-1.5 py-0.5 rounded font-semibold ${badgeClass[badgeVariant]}`}>
                    {badge}
                  </span>
                )}
                {/* Active underline */}
                <span className={`
                  absolute bottom-0 left-0 right-0 h-0.5 rounded-t-full transition-all duration-200
                  ${tab === id ? 'bg-clinical-primary dark:bg-blue-400 opacity-100' : 'opacity-0 group-hover:opacity-30 bg-slate-400'}
                `} />
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main id="main-content" className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5" role="main">
        <div id="panel-dashboard"   role="tabpanel" hidden={tab !== 'dashboard'}   className="animate-fade-in-up">
          {tab === 'dashboard'   && <Dashboard />}
        </div>
        <div id="panel-forecasting" role="tabpanel" hidden={tab !== 'forecasting'} className="animate-fade-in-up">
          {tab === 'forecasting' && <Forecasting />}
        </div>
        <div id="panel-model"       role="tabpanel" hidden={tab !== 'model'}       className="animate-fade-in-up">
          {tab === 'model'       && <ModelInfoPage />}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-clinical-border dark:border-slate-700 bg-white dark:bg-slate-800 py-3 transition-colors duration-200" role="contentinfo">
        <div className="max-w-screen-xl mx-auto px-5 flex items-center gap-2 text-xs text-clinical-text-muted dark:text-slate-400">
          <AlertTriangle size={11} className="text-amber-500 flex-shrink-0" aria-hidden="true" />
          <span>
            <strong className="text-clinical-text-secondary dark:text-slate-300">Disclaimer:</strong>{' '}
            Research prototype. Not FDA-approved. Not for autonomous diagnostic use.
            All decisions must be made by qualified clinicians.
            {' '}XGBoost v3.0 — F1 0.854 · AUC 0.951 &nbsp;|&nbsp; LSTM v4.2 — F1 0.795 · AUC 0.890 &nbsp;|&nbsp; Trained 2026-04-09 · 166 patients
          </span>
        </div>
      </footer>

      {showHelp && <KeyboardHelp onClose={() => setShowHelp(false)} />}

      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            fontSize: '13px',
            borderRadius: '10px',
            background: isDark ? '#2D3748' : '#fff',
            color: isDark ? '#E2E8F0' : '#1A202C',
            border: isDark ? '1px solid #4A5568' : '1px solid #E2E8F0',
            boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
          },
        }}
      />
    </div>
  )
}
