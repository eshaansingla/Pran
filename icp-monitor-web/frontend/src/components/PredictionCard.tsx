import { AlertTriangle, CheckCircle, Clock, ChevronRight } from 'lucide-react'
import type { WindowPrediction } from '../types'
import { CLASS_COLORS, CLASS_LABELS, fmtPct, fmtTimestamp } from '../utils/formatters'
import { useStore } from '../store/useStore'

interface Props {
  prediction: WindowPrediction & { timestamp?: string }
  windowIndex: number
  onClick?: () => void
}

const ICP_RANGES: Record<number, string> = { 0: '< 15 mmHg', 1: '>= 15 mmHg' }
const DARK_BG:     Record<number, string> = { 0: '#064E3B', 1: '#450A0A' }
const DARK_BORDER: Record<number, string> = { 0: '#065F46', 1: '#7F1D1D' }
const DARK_TEXT:   Record<number, string> = { 0: '#6EE7B7', 1: '#FCA5A5' }
const DARK_COLOR:  Record<number, string> = { 0: '#10B981', 1: '#EF4444' }
const LIGHT_BG:    Record<number, string> = { 0: '#F0FDF4', 1: '#FFF1F2' }
const LIGHT_BORDER:Record<number, string> = { 0: '#BBF7D0', 1: '#FECDD3' }
const LIGHT_TEXT:  Record<number, string> = { 0: '#065F46', 1: '#9F1239' }

export default function PredictionCard({ prediction, windowIndex, onClick }: Props) {
  const { isDark } = useStore()
  const cls    = prediction.class
  const bg     = isDark ? DARK_BG[cls]     : LIGHT_BG[cls]
  const border = isDark ? DARK_BORDER[cls] : LIGHT_BORDER[cls]
  const text   = isDark ? DARK_TEXT[cls]   : LIGHT_TEXT[cls]
  const color  = isDark ? DARK_COLOR[cls]  : CLASS_COLORS[cls]
  const ts     = prediction.timestamp ? fmtTimestamp(prediction.timestamp) : '—'

  // Confidence band color
  const confColor = prediction.confidence >= 0.85
    ? color
    : isDark ? '#F59E0B' : '#D97706'

  return (
    <div
      role={onClick ? 'button' : 'status'}
      tabIndex={onClick ? 0 : undefined}
      aria-live={onClick ? undefined : 'polite'}
      aria-label={`Window ${windowIndex}: ${CLASS_LABELS[cls]} ICP`}
      onClick={onClick}
      onKeyDown={onClick ? e => e.key === 'Enter' && onClick() : undefined}
      className={`rounded-xl border shadow-sm overflow-hidden transition-all duration-200 ${onClick ? 'cursor-pointer hover:shadow-md hover:-translate-y-px' : ''} animate-fade-in-up`}
      style={{ borderColor: border, backgroundColor: bg }}
    >
      {/* Top accent band */}
      <div className="h-1" style={{ backgroundColor: color, opacity: 0.7 }} />

      {/* Header */}
      <div className="px-4 py-2.5 flex items-center justify-between" style={{ borderBottom: `1px solid ${border}` }}>
        <span className="text-xs font-semibold tabular-nums" style={{ color: text }}>
          Window #{windowIndex}
        </span>
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1 text-2xs" style={{ color: text, opacity: 0.7 }}>
            <Clock size={10} />
            {ts}
          </span>
          {onClick && <ChevronRight size={12} style={{ color: text, opacity: 0.5 }} />}
        </div>
      </div>

      {/* Status row */}
      <div className="px-4 py-3.5 flex items-center gap-3">
        <span style={{ color }} className={cls === 1 ? 'animate-pulse-critical' : ''}>
          {cls === 0
            ? <CheckCircle size={26} strokeWidth={2} />
            : <AlertTriangle size={26} strokeWidth={2} />
          }
        </span>
        <div className="flex-1">
          <p className="text-lg font-bold leading-tight" style={{ color: text }}>
            {cls === 0 ? 'Normal ICP' : 'ABNORMAL ICP'}
          </p>
          <p className="text-2xs mt-0.5 opacity-70" style={{ color: text }}>
            Est. range: {ICP_RANGES[cls]}
          </p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold tabular-nums leading-none" style={{ color: confColor }}>
            {(prediction.confidence * 100).toFixed(0)}%
          </p>
          <p className="text-2xs mt-0.5" style={{ color: text, opacity: 0.7 }}>confidence</p>
        </div>
      </div>

      {/* Probability bars */}
      <div className="px-4 pb-3.5 space-y-1.5">
        {prediction.probabilities.map((p, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="w-14 text-2xs text-right tabular-nums font-medium" style={{ color: text, opacity: 0.8 }}>
              {CLASS_LABELS[i as 0|1]}
            </span>
            <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ backgroundColor: border }}>
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${(p * 100).toFixed(1)}%`,
                  backgroundColor: isDark ? DARK_COLOR[i as 0|1] : CLASS_COLORS[i as 0|1],
                  opacity: i === cls ? 1 : 0.35,
                }}
              />
            </div>
            <span className="w-10 text-2xs tabular-nums text-right font-mono" style={{ color: text }}>
              {fmtPct(p)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
