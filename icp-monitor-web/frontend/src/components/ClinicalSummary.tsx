import { useState } from 'react'
import { ChevronDown, ChevronUp, FileText } from 'lucide-react'
import type { BatchSummary, TrendPoint } from '../types'

interface Props {
  summary: BatchSummary
  trend: TrendPoint[]
}

function episodeCount(trend: TrendPoint[]): number {
  let count = 0
  let inEp = false
  for (const p of trend) {
    if (p.class === 1 && !inEp) { count++; inEp = true }
    else if (p.class === 0) inEp = false
  }
  return count
}

function longestStreak(trend: TrendPoint[]): number {
  let max = 0, cur = 0
  for (const p of trend) {
    if (p.class === 1) { cur++; max = Math.max(max, cur) }
    else cur = 0
  }
  return max
}

function generateSummary(summary: BatchSummary, trend: TrendPoint[]): string {
  const durationMin = +(summary.total * 10 / 60).toFixed(0)
  const eps         = episodeCount(trend)
  const streakW     = longestStreak(trend)
  const streakMin   = +(streakW * 10 / 60).toFixed(1)

  let text = `${durationMin}-minute session — ${summary.total.toLocaleString()} windows analysed. `

  if (summary.abnormal === 0) {
    text += `All windows classified Normal (<15 mmHg). No abnormal ICP detected. Continued monitoring recommended per protocol.`
    return text
  }

  text += `${summary.normal_pct}% Normal, ${summary.abnormal_pct}% Abnormal (≥15 mmHg). `

  if (eps === 1) {
    text += `1 abnormal episode detected (${streakMin} min). `
  } else {
    text += `${eps} abnormal episode${eps > 1 ? 's' : ''} detected. Longest: ${streakMin} min (${streakW} windows). `
  }

  if (summary.abnormal_pct > 50) {
    text += `Majority of session shows elevated ICP. Urgent clinical review and imaging recommended.`
  } else if (summary.abnormal_pct > 20) {
    text += `Significant abnormal burden. Recommend clinical assessment and consider repeat imaging.`
  } else {
    text += `Low abnormal burden. Routine clinical review recommended.`
  }

  return text
}

export default function ClinicalSummary({ summary, trend }: Props) {
  const [open, setOpen] = useState(true)
  const text = generateSummary(summary, trend)

  return (
    <div className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg shadow-sm overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <FileText size={14} className="text-clinical-primary dark:text-blue-400" />
          <span className="text-xs font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide">
            Clinical Summary
          </span>
        </div>
        {open
          ? <ChevronUp size={14} className="text-clinical-text-muted dark:text-slate-400" />
          : <ChevronDown size={14} className="text-clinical-text-muted dark:text-slate-400" />
        }
      </button>
      {open && (
        <div className="px-4 pb-4 pt-1">
          <p className="text-sm text-clinical-text-primary dark:text-slate-200 leading-relaxed">
            {text}
          </p>
          <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-2 italic">
            Auto-generated interpretation. Clinical decisions must be made by qualified professionals.
          </p>
        </div>
      )}
    </div>
  )
}
