import { useEffect } from 'react'
import { X, ChevronLeft, ChevronRight, Flag } from 'lucide-react'
import type { WindowPrediction } from '../types'
import { useStore } from '../store/useStore'
import { fmtPct } from '../utils/formatters'

const FEATURE_LABELS: Record<string, string> = {
  cardiac_amplitude:      'Cardiac Amplitude',
  cardiac_frequency:      'Cardiac Frequency',
  respiratory_amplitude:  'Respiratory Amplitude',
  slow_wave_power:        'Slow Wave Power',
  cardiac_power:          'Cardiac Power',
  mean_arterial_pressure: 'Mean Arterial Pressure',
  head_angle:             'Head Angle',
  motion_artifact_flag:   'Motion Artifact Flag',
}

const FEATURE_UNITS: Record<string, string> = {
  cardiac_amplitude:      'μm',
  cardiac_frequency:      'Hz',
  respiratory_amplitude:  'μm',
  slow_wave_power:        '',
  cardiac_power:          '',
  mean_arterial_pressure: 'mmHg',
  head_angle:             '°',
  motion_artifact_flag:   '',
}

interface Props {
  prediction: WindowPrediction
  windowIndex: number
  total: number
  onClose: () => void
  onPrev: () => void
  onNext: () => void
  featureNames: string[]
}

export default function InspectionModal({
  prediction, windowIndex, total, onClose, onPrev, onNext, featureNames,
}: Props) {
  const { toggleFlag, isFlagged } = useStore()
  const flagged = isFlagged(prediction.window_id)

  // Keyboard: Esc = close, ArrowLeft = prev, ArrowRight = next
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape')      { e.preventDefault(); onClose() }
      if (e.key === 'ArrowLeft')   { e.preventDefault(); onPrev() }
      if (e.key === 'ArrowRight')  { e.preventDefault(); onNext() }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, onPrev, onNext])

  const isAbnormal = prediction.class === 1
  const accentColor = isAbnormal ? '#DC2626' : '#059669'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label={`Window ${prediction.window_id} inspection`}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 dark:bg-black/70"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div className="relative bg-white dark:bg-slate-800 rounded-lg shadow-xl w-full max-w-md border border-clinical-border dark:border-slate-600 overflow-hidden">
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-4 border-b border-clinical-border dark:border-slate-700"
          style={{ borderLeftColor: accentColor, borderLeftWidth: 4 }}
        >
          <div>
            <h2 className="text-sm font-semibold text-clinical-text-primary dark:text-slate-100">
              Window #{prediction.window_id}
            </h2>
            <p className="text-xs mt-0.5" style={{ color: accentColor }}>
              {prediction.class_name} — {fmtPct(prediction.confidence)} confidence
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => toggleFlag(prediction.window_id)}
              aria-label={flagged ? 'Remove flag' : 'Flag for review'}
              title={flagged ? 'Remove flag' : 'Flag for review'}
              className={`p-1.5 rounded transition-colors ${
                flagged
                  ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400'
                  : 'text-clinical-text-muted dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              <Flag size={14} fill={flagged ? 'currentColor' : 'none'} />
            </button>
            <button
              onClick={onClose}
              aria-label="Close"
              className="p-1.5 rounded text-clinical-text-muted dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Probabilities */}
        <div className="px-5 py-4 space-y-2 border-b border-clinical-border dark:border-slate-700">
          {(['Normal', 'Abnormal'] as const).map((label, i) => {
            const p = prediction.probabilities[i]
            const color = i === 0 ? '#059669' : '#DC2626'
            return (
              <div key={label} className="flex items-center gap-2">
                <span className="w-20 text-xs text-right text-clinical-text-secondary dark:text-slate-300">{label}</span>
                <div className="flex-1 h-2.5 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${(p * 100).toFixed(1)}%`, backgroundColor: color, opacity: i === prediction.class ? 1 : 0.45 }}
                  />
                </div>
                <span className="w-12 text-xs tabular-nums text-right text-clinical-text-primary dark:text-slate-200">
                  {fmtPct(p)}
                </span>
              </div>
            )
          })}
        </div>

        {/* Feature values (if feature names provided) */}
        {featureNames.length > 0 && (
          <div className="px-5 py-3 border-b border-clinical-border dark:border-slate-700">
            <p className="text-2xs font-semibold text-clinical-text-muted dark:text-slate-400 uppercase tracking-wide mb-2">
              Feature Reference Ranges
            </p>
            <div className="space-y-1 max-h-44 overflow-y-auto">
              {featureNames.map(f => (
                <div key={f} className="flex items-center justify-between text-xs">
                  <span className="text-clinical-text-secondary dark:text-slate-300">
                    {FEATURE_LABELS[f] ?? f}
                  </span>
                  <span className="font-mono text-clinical-text-muted dark:text-slate-400">
                    {FEATURE_UNITS[f] ? `(${FEATURE_UNITS[f]})` : ''}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer navigation */}
        <div className="flex items-center justify-between px-5 py-3 bg-slate-50 dark:bg-slate-800/50">
          <button
            onClick={onPrev}
            disabled={windowIndex === 0}
            aria-label="Previous window"
            className="flex items-center gap-1 text-xs text-clinical-text-secondary dark:text-slate-300 disabled:opacity-40 hover:text-clinical-primary dark:hover:text-blue-400 transition-colors disabled:cursor-not-allowed"
          >
            <ChevronLeft size={14} />
            Prev
          </button>
          <span className="text-2xs text-clinical-text-muted dark:text-slate-400 tabular-nums">
            {windowIndex + 1} / {total}
          </span>
          <button
            onClick={onNext}
            disabled={windowIndex === total - 1}
            aria-label="Next window"
            className="flex items-center gap-1 text-xs text-clinical-text-secondary dark:text-slate-300 disabled:opacity-40 hover:text-clinical-primary dark:hover:text-blue-400 transition-colors disabled:cursor-not-allowed"
          >
            Next
            <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  )
}
