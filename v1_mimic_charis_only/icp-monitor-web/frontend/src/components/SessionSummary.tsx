import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'
import type { BatchSummary, TrendPoint } from '../types'
import { useStore } from '../store/useStore'

interface Props {
  summary: BatchSummary
  trend: TrendPoint[]
}

interface AbnormalEpisode { start: number; end: number; duration: number }

function findAbnormalEpisodes(trend: TrendPoint[]): AbnormalEpisode[] {
  const episodes: AbnormalEpisode[] = []
  let start: number | null = null
  for (let i = 0; i < trend.length; i++) {
    if (trend[i].class === 1 && start === null) {
      start = trend[i].windowId
    } else if (trend[i].class !== 1 && start !== null) {
      episodes.push({ start, end: trend[i - 1].windowId, duration: trend[i - 1].windowId - start + 1 })
      start = null
    }
  }
  if (start !== null) {
    const last = trend[trend.length - 1]
    episodes.push({ start, end: last.windowId, duration: last.windowId - start + 1 })
  }
  return episodes
}

export default function SessionSummary({ summary, trend }: Props) {
  const { isDark } = useStore()
  const normalColor   = isDark ? '#10B981' : '#059669'
  const abnormalColor = isDark ? '#EF4444' : '#DC2626'

  const pieData = [
    { name: 'Normal',   value: summary.normal,   color: normalColor },
    { name: 'Abnormal', value: summary.abnormal,  color: abnormalColor },
  ].filter(d => d.value > 0)

  const episodes = findAbnormalEpisodes(trend)

  const textPrimary   = isDark ? 'text-slate-200'  : 'text-clinical-text-primary'
  const textSecondary = isDark ? 'text-slate-300'  : 'text-clinical-text-secondary'
  const textMuted     = isDark ? 'text-slate-400'  : 'text-clinical-text-muted'
  const borderClass   = isDark ? 'border-slate-700' : 'border-clinical-border'

  return (
    <div className="space-y-5">
      <h3 className={`text-sm font-semibold uppercase tracking-wide ${textSecondary}`}>
        Session Summary
      </h3>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3">
        {[
          { label: 'Normal',   count: summary.normal,   pct: summary.normal_pct,   color: normalColor },
          { label: 'Abnormal', count: summary.abnormal, pct: summary.abnormal_pct, color: abnormalColor },
        ].map(({ label, count, pct, color }) => (
          <div
            key={label}
            className={`rounded-lg border ${borderClass} bg-white dark:bg-slate-800 px-3 py-3 text-center`}
          >
            <p className={`text-xs ${textMuted}`}>{label}</p>
            <p className="text-xl font-bold tabular-nums mt-0.5" style={{ color }}>
              {count.toLocaleString()}
            </p>
            <p className={`text-2xs ${textMuted}`}>{pct}%</p>
          </div>
        ))}
      </div>

      {/* Pie + legend */}
      <div className="flex items-center gap-4">
        <ResponsiveContainer width={120} height={120}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={32}
              outerRadius={52}
              dataKey="value"
              strokeWidth={2}
              stroke={isDark ? '#1A202C' : '#fff'}
            >
              {pieData.map((d, i) => (
                <Cell key={i} fill={d.color} />
              ))}
            </Pie>
            <Tooltip
              formatter={(v: number, name: string) => [`${v} windows`, name]}
              contentStyle={{
                fontSize: 11, borderRadius: 6,
                background: isDark ? '#2D3748' : '#fff',
                border: isDark ? '1px solid #4A5568' : '1px solid #E2E8F0',
                color: isDark ? '#E2E8F0' : '#1A202C',
              }}
            />
          </PieChart>
        </ResponsiveContainer>

        <div className="space-y-1.5 flex-1">
          <p className={`text-xs font-medium ${textPrimary}`}>
            Total: {summary.total.toLocaleString()} windows
          </p>
          {[
            { label: 'Normal',   pct: summary.normal_pct,   color: normalColor },
            { label: 'Abnormal', pct: summary.abnormal_pct, color: abnormalColor },
          ].map(({ label, pct, color }) => (
            <div key={label} className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
              <span className={`text-xs ${textSecondary} flex-1`}>{label}</span>
              <span className={`text-xs tabular-nums font-mono ${textPrimary}`}>{pct}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Abnormal episodes */}
      {episodes.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-clinical-critical dark:text-red-400 uppercase tracking-wide">
            Abnormal Episodes ({episodes.length})
          </h4>
          <div className="space-y-1.5 max-h-40 overflow-y-auto">
            {episodes.map((ep, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 px-3 py-2"
              >
                <span className="text-xs text-clinical-critical dark:text-red-400 font-medium">
                  Episode {i + 1}
                </span>
                <span className="text-xs text-red-600 dark:text-red-300 tabular-nums">
                  Windows {ep.start}–{ep.end}
                </span>
                <span className="text-xs text-red-500 dark:text-red-400">
                  {ep.duration} window{ep.duration > 1 ? 's' : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
