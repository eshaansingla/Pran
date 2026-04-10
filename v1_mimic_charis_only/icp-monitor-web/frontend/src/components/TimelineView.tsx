import { useRef } from 'react'
import type { TrendPoint } from '../types'
import { useStore } from '../store/useStore'

interface Segment {
  start: number
  end: number
  cls: 0 | 1
  count: number
}

function buildSegments(trend: TrendPoint[]): Segment[] {
  if (!trend.length) return []
  const segs: Segment[] = []
  let cur: Segment = { start: trend[0].windowId, end: trend[0].windowId, cls: trend[0].class as 0|1, count: 1 }
  for (let i = 1; i < trend.length; i++) {
    const p = trend[i]
    if (p.class === cur.cls) {
      cur.end = p.windowId
      cur.count++
    } else {
      segs.push(cur)
      cur = { start: p.windowId, end: p.windowId, cls: p.class as 0|1, count: 1 }
    }
  }
  segs.push(cur)
  return segs
}

interface Props {
  trend: TrendPoint[]
  onSelect: (windowIndex: number) => void
}

export default function TimelineView({ trend, onSelect }: Props) {
  const { isDark } = useStore()
  const containerRef = useRef<HTMLDivElement>(null)

  if (!trend.length) {
    return (
      <div className="h-10 flex items-center justify-center text-xs text-clinical-text-muted dark:text-slate-400">
        Upload data to see timeline
      </div>
    )
  }

  const segments = buildSegments(trend)
  const total    = trend.length

  const normalColor   = isDark ? '#10B981' : '#059669'
  const abnormalColor = isDark ? '#EF4444' : '#DC2626'

  return (
    <div className="space-y-1">
      <p className="text-2xs font-semibold uppercase tracking-wide text-clinical-text-muted dark:text-slate-400">
        Session Timeline
      </p>
      <div
        ref={containerRef}
        className="flex h-8 rounded overflow-hidden border border-clinical-border dark:border-slate-600 gap-px"
        aria-label="ICP session timeline"
        role="img"
      >
        {segments.map((seg, i) => {
          const widthPct = (seg.count / total) * 100
          const color    = seg.cls === 0 ? normalColor : abnormalColor
          const label    = seg.cls === 0 ? 'Normal' : 'Abnormal'
          return (
            <div
              key={i}
              style={{ width: `${widthPct}%`, backgroundColor: color, minWidth: 2 }}
              className="cursor-pointer opacity-80 hover:opacity-100 transition-opacity relative group"
              role="button"
              tabIndex={0}
              aria-label={`${label}: windows ${seg.start}–${seg.end}`}
              onClick={() => onSelect(seg.start - 1)}
              onKeyDown={e => e.key === 'Enter' && onSelect(seg.start - 1)}
            >
              {/* Tooltip */}
              {widthPct > 3 && (
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10 whitespace-nowrap">
                  <div className="bg-slate-800 dark:bg-slate-200 text-white dark:text-slate-900 text-2xs px-2 py-1 rounded shadow-lg">
                    {label} · #{seg.start}–#{seg.end} · {Math.round(seg.count * 10 / 60 * 10) / 10} min
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: normalColor }} />
          <span className="text-2xs text-clinical-text-muted dark:text-slate-400">Normal</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: abnormalColor }} />
          <span className="text-2xs text-clinical-text-muted dark:text-slate-400">Abnormal</span>
        </div>
        <span className="ml-auto text-2xs text-clinical-text-muted dark:text-slate-400">
          Click segment to inspect
        </span>
      </div>
    </div>
  )
}
