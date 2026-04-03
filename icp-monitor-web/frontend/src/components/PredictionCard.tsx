import { AlertTriangle, CheckCircle, Clock, TrendingUp } from 'lucide-react'
import type { WindowPrediction } from '../types'
import {
  CLASS_BG, CLASS_BORDER, CLASS_COLORS, CLASS_LABELS, CLASS_TEXT,
  fmtPct, fmtTimestamp,
} from '../utils/formatters'

interface Props {
  prediction: WindowPrediction & { timestamp?: string }
  windowIndex: number
}

const ICP_RANGES: Record<number, string> = {
  0: '< 15 mmHg',
  1: '15 – 20 mmHg',
  2: '\u2265 20 mmHg',
}

function classIcon(cls: number, size = 20) {
  if (cls === 0) return <CheckCircle size={size} aria-hidden="true" />
  if (cls === 1) return <TrendingUp size={size} aria-hidden="true" />
  return <AlertTriangle size={size} aria-hidden="true" />
}

export default function PredictionCard({ prediction, windowIndex }: Props) {
  const cls = prediction.class
  const bg     = CLASS_BG[cls]
  const border = CLASS_BORDER[cls]
  const text   = CLASS_TEXT[cls]
  const color  = CLASS_COLORS[cls]
  const label  = CLASS_LABELS[cls]
  const ts     = prediction.timestamp
    ? fmtTimestamp(prediction.timestamp)
    : '—'

  return (
    <div
      role="status"
      aria-live="polite"
      aria-label={`ICP classification: ${label}`}
      className="rounded-lg border shadow-sm overflow-hidden"
      style={{ borderColor: border, backgroundColor: bg }}
    >
      {/* Header */}
      <div
        className="px-4 py-3 flex items-center justify-between"
        style={{ borderBottom: `1px solid ${border}` }}
      >
        <span className="text-xs font-semibold" style={{ color: text }}>
          Window #{windowIndex}
        </span>
        <span className="flex items-center gap-1 text-xs" style={{ color: text }}>
          <Clock size={11} />
          {ts}
        </span>
      </div>

      {/* Status */}
      <div className="px-4 py-4 flex items-center gap-3">
        <span style={{ color }}>{classIcon(cls, 28)}</span>
        <div>
          <p className="text-xl font-bold leading-tight" style={{ color: text }}>
            {cls === 2 ? 'CRITICAL ICP' : cls === 1 ? 'ELEVATED ICP' : 'Normal ICP'}
          </p>
          <p className="text-xs mt-0.5" style={{ color: text }}>
            Estimated range: {ICP_RANGES[cls]}
          </p>
        </div>
        <div className="ml-auto text-right">
          <p className="text-2xl font-bold tabular-nums" style={{ color }}>
            {(prediction.confidence * 100).toFixed(1)}%
          </p>
          <p className="text-2xs" style={{ color: text }}>confidence</p>
        </div>
      </div>

      {/* Probability bars */}
      <div
        className="px-4 pb-4 space-y-2"
        aria-label="Probability distribution"
      >
        <p className="text-2xs font-semibold uppercase tracking-wide" style={{ color: text }}>
          Probability Distribution
        </p>
        {prediction.probabilities.map((p, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="w-16 text-xs text-right tabular-nums" style={{ color: text }}>
              {CLASS_LABELS[i]}
            </span>
            <div className="flex-1 h-3 rounded-full bg-white/50 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${(p * 100).toFixed(1)}%`,
                  backgroundColor: CLASS_COLORS[i as 0|1|2],
                  opacity: i === cls ? 1 : 0.45,
                }}
                role="progressbar"
                aria-valuenow={Math.round(p * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`${CLASS_LABELS[i]} probability`}
              />
            </div>
            <span className="w-12 text-xs tabular-nums text-right" style={{ color: text }}>
              {fmtPct(p)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
