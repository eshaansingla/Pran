import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'
import type { BatchSummary, TrendPoint } from '../types'
import { CLASS_COLORS } from '../utils/formatters'

interface Props {
  summary: BatchSummary
  trend: TrendPoint[]
}

interface CriticalEpisode {
  start: number
  end: number
  duration: number
}

function findCriticalEpisodes(trend: TrendPoint[]): CriticalEpisode[] {
  const episodes: CriticalEpisode[] = []
  let start: number | null = null
  for (let i = 0; i < trend.length; i++) {
    if (trend[i].class === 2 && start === null) {
      start = trend[i].windowId
    } else if (trend[i].class !== 2 && start !== null) {
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

const PIE_DATA_KEYS: Array<{ key: keyof BatchSummary; label: string; cls: 0 | 1 | 2 }> = [
  { key: 'normal',   label: 'Normal',   cls: 0 },
  { key: 'elevated', label: 'Elevated', cls: 1 },
  { key: 'critical', label: 'Critical', cls: 2 },
]

export default function SessionSummary({ summary, trend }: Props) {
  const pieData = PIE_DATA_KEYS
    .map(d => ({ name: d.label, value: summary[d.key] as number, cls: d.cls }))
    .filter(d => d.value > 0)

  const episodes = findCriticalEpisodes(trend)

  return (
    <div className="space-y-5">
      <h3 className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide">
        Session Summary
      </h3>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        {PIE_DATA_KEYS.map(({ key, label, cls }) => {
          const count = summary[key] as number
          const pct = summary[`${key}_pct` as keyof BatchSummary] as number
          return (
            <div
              key={key}
              className="rounded-lg border border-clinical-border bg-white px-3 py-3 text-center"
            >
              <p className="text-xs text-clinical-text-muted">{label}</p>
              <p
                className="text-xl font-bold tabular-nums mt-0.5"
                style={{ color: CLASS_COLORS[cls] }}
              >
                {count.toLocaleString()}
              </p>
              <p className="text-2xs text-clinical-text-muted">{pct}%</p>
            </div>
          )
        })}
      </div>

      {/* Pie chart */}
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
              strokeWidth={1.5}
              stroke="#fff"
            >
              {pieData.map((d, i) => (
                <Cell key={i} fill={CLASS_COLORS[d.cls]} />
              ))}
            </Pie>
            <Tooltip
              formatter={(v: number, name: string) => [`${v} windows`, name]}
              contentStyle={{ fontSize: 11, borderRadius: 6 }}
            />
          </PieChart>
        </ResponsiveContainer>

        <div className="space-y-1.5 flex-1">
          <p className="text-xs font-medium text-clinical-text-primary">
            Total analysed: {summary.total.toLocaleString()} windows
          </p>
          {PIE_DATA_KEYS.map(({ key, label, cls }) => {
            const pct = summary[`${key}_pct` as keyof BatchSummary] as number
            return (
              <div key={key} className="flex items-center gap-2">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: CLASS_COLORS[cls] }}
                />
                <span className="text-xs text-clinical-text-secondary flex-1">{label}</span>
                <span className="text-xs tabular-nums font-mono text-clinical-text-primary">
                  {pct}%
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Critical episodes */}
      {episodes.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-clinical-critical uppercase tracking-wide">
            Critical Episodes ({episodes.length})
          </h4>
          <div className="space-y-1.5 max-h-40 overflow-y-auto">
            {episodes.map((ep, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded border border-red-200 bg-red-50 px-3 py-2"
              >
                <span className="text-xs text-clinical-critical font-medium">
                  Episode {i + 1}
                </span>
                <span className="text-xs text-red-600 tabular-nums">
                  Windows {ep.start}–{ep.end}
                </span>
                <span className="text-xs text-red-500">
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
