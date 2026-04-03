import { AlertTriangle, CheckCircle, X } from 'lucide-react'
import { useState } from 'react'
import type { BatchSummary } from '../types'

interface Props {
  summary: BatchSummary | null
  loading: boolean
}

export default function AlertBanner({ summary, loading }: Props) {
  const [dismissed, setDismissed] = useState(false)

  if (loading) return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
      <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
      <p className="text-sm text-blue-700 dark:text-blue-300 font-medium">Analysing data…</p>
    </div>
  )

  if (!summary) return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-slate-50 dark:bg-slate-800 border border-clinical-border dark:border-slate-600">
      <CheckCircle size={16} className="text-slate-400 flex-shrink-0" />
      <p className="text-sm text-clinical-text-muted dark:text-slate-400">
        No session active — upload a CSV to begin analysis
      </p>
    </div>
  )

  const hasAbnormal = summary.abnormal > 0

  if (!hasAbnormal) return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800">
      <CheckCircle size={16} className="text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
      <p className="text-sm text-emerald-800 dark:text-emerald-200 font-medium">
        All {summary.total.toLocaleString()} windows classified Normal — no abnormal ICP detected
      </p>
    </div>
  )

  if (dismissed) return (
    <button
      onClick={() => setDismissed(false)}
      className="w-full flex items-center gap-2 px-4 py-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-xs text-red-700 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
    >
      <AlertTriangle size={13} />
      Abnormal ICP alert dismissed — click to show
    </button>
  )

  return (
    <div
      role="alert"
      className="flex items-start gap-3 px-4 py-3 rounded-lg bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 dark:border-red-600 border border-red-200 dark:border-red-800"
    >
      <AlertTriangle
        size={18}
        className="text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5"
        aria-hidden="true"
      />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-red-800 dark:text-red-200">
          ABNORMAL ICP DETECTED — Clinical Review Recommended
        </p>
        <p className="text-xs text-red-700 dark:text-red-300 mt-0.5">
          {summary.abnormal.toLocaleString()} of {summary.total.toLocaleString()} windows abnormal
          ({summary.abnormal_pct}%) · ICP likely ≥15 mmHg
        </p>
      </div>
      <button
        onClick={() => setDismissed(true)}
        aria-label="Dismiss alert"
        className="text-red-400 hover:text-red-600 dark:hover:text-red-200 flex-shrink-0 transition-colors"
      >
        <X size={16} />
      </button>
    </div>
  )
}
