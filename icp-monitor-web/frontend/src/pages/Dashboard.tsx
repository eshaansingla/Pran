import { useState, useEffect, useCallback } from 'react'
import { AlertTriangle } from 'lucide-react'
import type { BatchResult, TrendPoint, WindowPrediction } from '../types'
import { predictBatch } from '../utils/api'
import { fmtTimestamp } from '../utils/formatters'
import UploadZone from '../components/UploadZone'
import PredictionCard from '../components/PredictionCard'
import TrendChart from '../components/TrendChart'
import FeatureExplainer from '../components/FeatureExplainer'
import SessionSummary from '../components/SessionSummary'
import ReportExporter from '../components/ReportExporter'

export default function Dashboard() {
  const [loading, setLoading]       = useState(false)
  const [errors, setErrors]         = useState<string[]>([])
  const [result, setResult]         = useState<BatchResult | null>(null)
  const [current, setCurrent]       = useState<(WindowPrediction & { timestamp: string }) | null>(null)
  const [currentIdx, setCurrentIdx] = useState(0)
  const [trend, setTrend]           = useState<TrendPoint[]>([])
  const [hasAbnormal, setHasAbnormal] = useState(false)

  const handleFile = useCallback(async (file: File) => {
    setLoading(true)
    setErrors([])
    setResult(null)
    setTrend([])
    setCurrent(null)
    setHasAbnormal(false)
    try {
      const res = await predictBatch(file)
      setResult(res)

      if (res.parse_warnings.length > 0) setErrors(res.parse_warnings)

      const points: TrendPoint[] = res.predictions.map((p, i) => ({
        windowId: p.window_id,
        timestamp: fmtTimestamp(new Date(Date.now() - (res.predictions.length - i) * 10000).toISOString()),
        class: p.class,
        confidence: p.confidence,
        label: p.class_name,
      }))
      setTrend(points)
      setHasAbnormal(res.summary.abnormal > 0)

      // Show first window by default
      if (res.predictions.length > 0) {
        const first = res.predictions[0]
        setCurrent({ ...first, timestamp: new Date().toISOString() })
        setCurrentIdx(0)
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setErrors(msg.split('\n').filter(Boolean))
    } finally {
      setLoading(false)
    }
  }, [])

  const handleClear = () => {
    setResult(null)
    setErrors([])
    setTrend([])
    setCurrent(null)
    setHasAbnormal(false)
    setCurrentIdx(0)
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault()
        document.getElementById('file-input-trigger')?.click()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const selectWindow = (idx: number) => {
    if (!result) return
    const p = result.predictions[idx]
    setCurrent({ ...p, timestamp: new Date().toISOString() })
    setCurrentIdx(idx)
  }

  return (
    <div className="space-y-5">
      {/* Abnormal alert banner */}
      {hasAbnormal && (
        <div
          role="alert"
          className="flex items-center gap-3 px-4 py-3 bg-red-600 text-white rounded-lg shadow-sm"
        >
          <AlertTriangle size={18} className="flex-shrink-0" />
          <p className="text-sm font-semibold">
            Abnormal ICP detected in this session — {result?.summary.abnormal} window
            {(result?.summary.abnormal ?? 0) > 1 ? 's' : ''} ({result?.summary.abnormal_pct}%).
            Clinical review indicated.
          </p>
        </div>
      )}

      <div className="grid grid-cols-12 gap-5">
        {/* Left column */}
        <div className="col-span-4 space-y-5">
          {/* Upload */}
          <section
            aria-label="Data Upload"
            className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm"
          >
            <UploadZone
              onFile={handleFile}
              loading={loading}
              errors={errors}
              onClear={handleClear}
              hasData={!!result}
            />
          </section>

          {/* Current prediction */}
          {current && (
            <section aria-label="Current Window Prediction">
              <PredictionCard prediction={current} windowIndex={currentIdx + 1} />
            </section>
          )}

          {/* Feature explainer */}
          {current && (
            <section
              aria-label="Feature Attribution"
              className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm"
            >
              {/* We don't have SHAP for batch — show placeholder for first window */}
              <FeatureExplainer features={[]} predictedClass={current.class} />
              <p className="text-2xs text-clinical-text-muted mt-2">
                Per-window SHAP attribution available via single-window /api/predict endpoint.
              </p>
            </section>
          )}
        </div>

        {/* Right column */}
        <div className="col-span-8 space-y-5">
          {/* Trend chart */}
          <section
            aria-label="ICP Trend"
            className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide">
                ICP Classification Trend
              </h2>
              {result && (
                <span className="text-xs text-clinical-text-muted">
                  {result.predictions.length} windows · {(result.predictions.length * 10 / 60).toFixed(1)} min
                </span>
              )}
            </div>
            <TrendChart data={trend} />

            {/* Window selector */}
            {result && (
              <div className="mt-4">
                <label className="text-xs text-clinical-text-muted" htmlFor="window-select">
                  Inspect window:
                </label>
                <select
                  id="window-select"
                  value={currentIdx}
                  onChange={e => selectWindow(Number(e.target.value))}
                  className="ml-2 text-xs border border-clinical-border rounded px-2 py-1 bg-white text-clinical-text-primary focus:outline-none focus:ring-1 focus:ring-clinical-primary"
                  aria-label="Select window to inspect"
                >
                  {result.predictions.map((p, i) => (
                    <option key={i} value={i}>
                      #{i + 1} — {p.class_name} ({(p.confidence * 100).toFixed(0)}%)
                    </option>
                  ))}
                </select>
              </div>
            )}
          </section>

          {/* Session summary */}
          {result && (
            <section
              aria-label="Session Summary"
              className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm"
            >
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide">
                  Session Summary
                </span>
                <ReportExporter result={result} />
              </div>
              <SessionSummary summary={result.summary} trend={trend} />
            </section>
          )}

          {/* Empty state */}
          {!result && !loading && (
            <div className="flex flex-col items-center justify-center h-64 text-clinical-text-muted space-y-2">
              <p className="text-sm">Upload a CSV file to begin analysis</p>
              <p className="text-xs">Keyboard shortcut: Ctrl+U</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
