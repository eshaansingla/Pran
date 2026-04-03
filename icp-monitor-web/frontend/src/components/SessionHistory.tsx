import { Clock, Trash2, RotateCcw, History } from 'lucide-react'
import { useState } from 'react'
import { useStore, type StoredSession } from '../store/useStore'
import type { BatchResult } from '../types'

interface Props {
  onLoad: (result: BatchResult) => void
}

function fmtDate(iso: string) {
  return new Date(iso).toLocaleString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

function sessionToBatchResult(s: StoredSession): BatchResult {
  return {
    predictions: s.predictions,
    parse_warnings: [],
    summary: {
      total:        s.windowCount,
      normal:       s.normal,
      abnormal:     s.abnormal,
      normal_pct:   +((s.normal / s.windowCount) * 100).toFixed(1),
      abnormal_pct: s.abnormalPct,
    },
    timestamp: s.date,
    feature_names: [],
  }
}

export default function SessionHistory({ onLoad }: Props) {
  const { sessions, removeSession } = useStore()
  const [open, setOpen] = useState(false)

  if (sessions.length === 0) return null

  return (
    <div className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg shadow-sm overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <History size={14} className="text-clinical-primary dark:text-blue-400" />
          <span className="text-xs font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide">
            Session History
          </span>
          <span className="text-2xs bg-slate-100 dark:bg-slate-700 text-clinical-text-muted dark:text-slate-400 px-1.5 py-0.5 rounded font-mono">
            {sessions.length}
          </span>
        </div>
        <span className="text-2xs text-clinical-text-muted dark:text-slate-500">
          {open ? 'hide' : 'show'}
        </span>
      </button>

      {open && (
        <div className="divide-y divide-clinical-border dark:divide-slate-700 max-h-64 overflow-y-auto">
          {sessions.map(s => (
            <div
              key={s.id}
              className="flex items-center gap-3 px-4 py-2.5 hover:bg-slate-50 dark:hover:bg-slate-700/40 transition-colors"
            >
              <Clock size={13} className="text-clinical-text-muted dark:text-slate-500 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-clinical-text-primary dark:text-slate-200 truncate">
                  {fmtDate(s.date)}
                </p>
                <p className="text-2xs text-clinical-text-muted dark:text-slate-400">
                  {s.windowCount} windows · {s.durationMin} min ·{' '}
                  <span style={{ color: s.abnormalPct > 0 ? '#DC2626' : '#059669' }}>
                    {s.abnormalPct}% abnormal
                  </span>
                </p>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <button
                  onClick={() => onLoad(sessionToBatchResult(s))}
                  aria-label="Reload session"
                  title="Reload session"
                  className="p-1 rounded text-clinical-primary dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
                >
                  <RotateCcw size={13} />
                </button>
                <button
                  onClick={() => removeSession(s.id)}
                  aria-label="Delete session"
                  title="Delete session"
                  className="p-1 rounded text-clinical-text-muted dark:text-slate-500 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
