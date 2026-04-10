import { Activity, AlertTriangle, Clock, TrendingUp } from 'lucide-react'
import type { BatchSummary, TrendPoint } from '../types'

interface Props {
  summary: BatchSummary
  trend: TrendPoint[]
}

function longestAbnormalStreak(trend: TrendPoint[]): number {
  let max = 0, cur = 0
  for (const p of trend) {
    if (p.class === 1) { cur++; max = Math.max(max, cur) } else cur = 0
  }
  return max
}

interface CardProps {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  accent?: string
  barPct?: number
  barColor?: string
}

function StatCard({ icon, label, value, sub, accent, barPct, barColor }: CardProps) {
  return (
    <div className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-xl px-4 py-3.5 shadow-sm card-hover overflow-hidden relative">
      {/* Subtle side accent line */}
      {accent && (
        <div className="absolute left-0 top-0 bottom-0 w-0.5 rounded-l-xl" style={{ backgroundColor: accent }} />
      )}
      <div className="flex items-start gap-3">
        <div className="mt-0.5 text-clinical-text-muted dark:text-slate-500 flex-shrink-0 p-1.5 bg-slate-100 dark:bg-slate-700 rounded-lg">
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-2xs text-clinical-text-muted dark:text-slate-500 uppercase tracking-widest font-medium">
            {label}
          </p>
          <p className="text-xl font-bold tabular-nums mt-0.5 leading-tight" style={accent ? { color: accent } : undefined}>
            {value}
          </p>
          {sub && <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-0.5 leading-snug">{sub}</p>}
          {barPct !== undefined && (
            <div className="mt-2 h-1 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${Math.min(barPct, 100)}%`, backgroundColor: barColor ?? '#2C5282' }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function StatsCards({ summary, trend }: Props) {
  const streakWindows = longestAbnormalStreak(trend)
  const streakMin     = +(streakWindows * 10 / 60).toFixed(1)
  const durationMin   = +(summary.total * 10 / 60).toFixed(1)
  const abnPct        = summary.abnormal_pct

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 animate-fade-in-up">
      <StatCard
        icon={<Activity size={14} />}
        label="Total Windows"
        value={summary.total.toLocaleString()}
        sub={`${durationMin} min session`}
        barPct={100}
        barColor="#2C5282"
      />
      <StatCard
        icon={<TrendingUp size={14} />}
        label="Abnormal %"
        value={`${abnPct}%`}
        sub={abnPct > 20 ? 'Above 20% threshold' : 'Within expected range'}
        accent={abnPct > 20 ? '#DC2626' : undefined}
        barPct={abnPct}
        barColor={abnPct > 20 ? '#DC2626' : '#059669'}
      />
      <StatCard
        icon={<AlertTriangle size={14} />}
        label="Longest Streak"
        value={streakWindows > 0 ? `${streakMin} min` : '—'}
        sub={streakWindows > 0 ? `${streakWindows} consecutive windows` : 'No abnormal episodes'}
        accent={streakWindows > 3 ? '#DC2626' : undefined}
        barPct={streakWindows > 0 ? Math.min(streakWindows / summary.total * 100 * 3, 100) : 0}
        barColor="#DC2626"
      />
      <StatCard
        icon={<Clock size={14} />}
        label="Session Duration"
        value={`${durationMin} min`}
        sub={`${summary.normal.toLocaleString()} normal · ${summary.abnormal.toLocaleString()} abnormal`}
        barPct={summary.normal_pct}
        barColor="#059669"
      />
    </div>
  )
}
