import { Activity, AlertTriangle, Clock, TrendingUp } from 'lucide-react'
import type { BatchSummary, TrendPoint } from '../types'

interface Props {
  summary: BatchSummary
  trend: TrendPoint[]
}

function longestAbnormalStreak(trend: TrendPoint[]): number {
  let max = 0, cur = 0
  for (const p of trend) {
    if (p.class === 1) { cur++; max = Math.max(max, cur) }
    else cur = 0
  }
  return max
}

interface CardProps {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  accent?: string
}

function StatCard({ icon, label, value, sub, accent }: CardProps) {
  return (
    <div className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg px-4 py-3 shadow-sm flex items-start gap-3">
      <div className="mt-0.5 text-clinical-text-muted dark:text-slate-400 flex-shrink-0">
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-2xs text-clinical-text-muted dark:text-slate-400 uppercase tracking-wide font-medium">
          {label}
        </p>
        <p
          className="text-xl font-bold tabular-nums mt-0.5 leading-tight"
          style={accent ? { color: accent } : undefined}
        >
          {value}
        </p>
        {sub && (
          <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-0.5">{sub}</p>
        )}
      </div>
    </div>
  )
}

export default function StatsCards({ summary, trend }: Props) {
  const streakWindows = longestAbnormalStreak(trend)
  const streakMin     = +(streakWindows * 10 / 60).toFixed(1)
  const durationMin   = +(summary.total * 10 / 60).toFixed(1)
  const abnormalHigh  = summary.abnormal_pct > 20

  return (
    <div className="grid grid-cols-4 gap-3">
      <StatCard
        icon={<Activity size={16} />}
        label="Total Windows"
        value={summary.total.toLocaleString()}
        sub={`${durationMin} min session`}
      />
      <StatCard
        icon={<TrendingUp size={16} />}
        label="Abnormal %"
        value={`${summary.abnormal_pct}%`}
        sub={abnormalHigh ? '↑ Above 20% threshold' : 'Within expected range'}
        accent={abnormalHigh ? '#DC2626' : undefined}
      />
      <StatCard
        icon={<AlertTriangle size={16} />}
        label="Longest Streak"
        value={streakWindows > 0 ? `${streakMin} min` : '—'}
        sub={streakWindows > 0 ? `${streakWindows} consecutive windows` : 'No abnormal episodes'}
        accent={streakWindows > 0 ? '#DC2626' : undefined}
      />
      <StatCard
        icon={<Clock size={16} />}
        label="Session Duration"
        value={`${durationMin} min`}
        sub={`${summary.total} × 10-second windows`}
      />
    </div>
  )
}
