import { AlertTriangle, CheckCircle, X, Loader2 } from 'lucide-react'
import { useState } from 'react'
import type { BatchSummary } from '../types'

interface Props {
  summary: BatchSummary | null
  loading: boolean
}

export default function AlertBanner({ summary, loading }: Props) {
  const [dismissed, setDismissed] = useState(false)

  if (loading) return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 animate-fade-in-up">
      <Loader2 size={15} className="text-blue-500 animate-spin flex-shrink-0" />
      <p className="text-sm text-blue-700 dark:text-blue-300 font-medium">Analysing windows…</p>
    </div>
  )

  if (!summary) return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-700 shadow-sm">
      <CheckCircle size={14} className="text-slate-300 dark:text-slate-600 flex-shrink-0" />
      <p className="text-sm text-clinical-text-muted dark:text-slate-500">
        No active session — upload a CSV file to begin ICP classification
      </p>
    </div>
  )

  if (!summary.abnormal) return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 animate-fade-in-up">
      <CheckCircle size={15} className="text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
      <p className="text-sm text-emerald-800 dark:text-emerald-200 font-medium">
        All <strong>{summary.total.toLocaleString()}</strong> windows Normal — no elevated ICP detected
      </p>
    </div>
  )

  if (dismissed) return (
    <button
      onClick={() => setDismissed(false)}
      className="w-full flex items-center gap-2 px-4 py-2.5 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-xs text-red-700 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
    >
      <AlertTriangle size={12} />
      Abnormal ICP alert dismissed — click to restore
    </button>
  )

  const pct = summary.abnormal_pct
  const severity = pct >= 50 ? 'critical' : pct >= 20 ? 'high' : 'moderate'
  const severityLabel = { critical: 'CRITICAL', high: 'HIGH ALERT', moderate: 'ALERT' }[severity]

  return (
    <div
      role="alert"
      className="flex items-start gap-3 px-4 py-3.5 rounded-xl border-l-4 border-red-500 dark:border-red-500 bg-gradient-to-r from-red-50 to-rose-50 dark:from-red-900/25 dark:to-rose-900/20 border border-red-200 dark:border-red-800 shadow-sm animate-fade-in-up"
    >
      <div className="flex-shrink-0 mt-0.5 animate-pulse-critical">
        <AlertTriangle size={18} className="text-red-600 dark:text-red-400" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="text-sm font-bold text-red-800 dark:text-red-200">
            ABNORMAL ICP DETECTED
          </p>
          <span className="text-2xs font-bold px-1.5 py-0.5 bg-red-600 text-white rounded-md tracking-wide">
            {severityLabel}
          </span>
        </div>
        <p className="text-xs text-red-700 dark:text-red-300 mt-0.5">
          <strong>{summary.abnormal.toLocaleString()}</strong> of {summary.total.toLocaleString()} windows abnormal
          ({pct}%) above threshold · Clinical review recommended
        </p>
      </div>
      <button
        onClick={() => setDismissed(true)}
        aria-label="Dismiss alert"
        className="text-red-400 hover:text-red-600 dark:hover:text-red-300 flex-shrink-0 transition-colors p-0.5 rounded"
      >
        <X size={15} />
      </button>
    </div>
  )
}
