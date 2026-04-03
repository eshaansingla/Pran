import { useState, useEffect, useCallback } from 'react'
import toast from 'react-hot-toast'
import type { BatchResult, TrendPoint, WindowPrediction } from '../types'
import { predictBatch } from '../utils/api'
import { fmtTimestamp } from '../utils/formatters'
import { useStore } from '../store/useStore'

import UploadZone      from '../components/UploadZone'
import PredictionCard  from '../components/PredictionCard'
import TrendChart      from '../components/TrendChart'
import FeatureExplainer from '../components/FeatureExplainer'
import SessionSummary  from '../components/SessionSummary'
import AlertBanner     from '../components/AlertBanner'
import StatsCards      from '../components/StatsCards'
import TimelineView    from '../components/TimelineView'
import ClinicalSummary from '../components/ClinicalSummary'
import SessionHistory  from '../components/SessionHistory'
import InspectionModal from '../components/InspectionModal'
import ExportMenu      from '../components/ExportMenu'

export default function Dashboard() {
  const { addSession }                 = useStore()
  const [loading, setLoading]          = useState(false)
  const [errors, setErrors]            = useState<string[]>([])
  const [result, setResult]            = useState<BatchResult | null>(null)
  const [current, setCurrent]          = useState<(WindowPrediction & { timestamp: string }) | null>(null)
  const [currentIdx, setCurrentIdx]    = useState(0)
  const [trend, setTrend]              = useState<TrendPoint[]>([])
  const [inspecting, setInspecting]    = useState(false)

  const applyResult = useCallback((res: BatchResult) => {
    setResult(res)
    setErrors(res.parse_warnings.length > 0 ? res.parse_warnings : [])

    const points: TrendPoint[] = res.predictions.map((p, i) => ({
      windowId:   p.window_id,
      timestamp:  fmtTimestamp(new Date(Date.now() - (res.predictions.length - i) * 10000).toISOString()),
      class:      p.class,
      confidence: p.confidence,
      label:      p.class_name,
    }))
    setTrend(points)

    if (res.predictions.length > 0) {
      setCurrent({ ...res.predictions[0], timestamp: new Date().toISOString() })
      setCurrentIdx(0)
    }
  }, [])

  const handleFile = useCallback(async (file: File) => {
    setLoading(true)
    setErrors([])
    setResult(null)
    setTrend([])
    setCurrent(null)
    setInspecting(false)
    try {
      const res = await predictBatch(file)
      applyResult(res)
      addSession(res)
      toast.success(`Analysis complete — ${res.predictions.length.toLocaleString()} windows processed`)
      if (res.parse_warnings.length > 0) {
        toast(`${res.parse_warnings.length} row warning(s) — check validation panel`, {
          icon: '⚠️',
        })
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setErrors(msg.split('\n').filter(Boolean))
      toast.error('Upload failed — ' + msg.split('\n')[0])
    } finally {
      setLoading(false)
    }
  }, [applyResult, addSession])

  const handleClear = () => {
    setResult(null)
    setErrors([])
    setTrend([])
    setCurrent(null)
    setCurrentIdx(0)
    setInspecting(false)
  }

  const selectWindow = useCallback((idx: number) => {
    if (!result) return
    const clamped = Math.max(0, Math.min(idx, result.predictions.length - 1))
    const p = result.predictions[clamped]
    setCurrent({ ...p, timestamp: new Date().toISOString() })
    setCurrentIdx(clamped)
  }, [result])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault()
        document.getElementById('file-input-trigger')?.click()
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault()
        if (result) {
          document.getElementById('export-trigger')?.click()
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [result])

  return (
    <div className="space-y-4">
      {/* Alert banner — full width */}
      <AlertBanner summary={result?.summary ?? null} loading={loading} />

      {/* Stats row — only when data loaded */}
      {result && (
        <StatsCards summary={result.summary} trend={trend} />
      )}

      <div className="grid grid-cols-12 gap-4">
        {/* ── Left column ─────────────────────────────────── */}
        <div className="col-span-12 lg:col-span-4 space-y-4">

          {/* Upload */}
          <section
            aria-label="Data Upload"
            className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg p-5 shadow-sm transition-colors duration-200"
          >
            <UploadZone
              onFile={handleFile}
              loading={loading}
              errors={errors}
              onClear={handleClear}
              hasData={!!result}
            />
          </section>

          {/* Clinical summary */}
          {result && (
            <ClinicalSummary summary={result.summary} trend={trend} />
          )}

          {/* Session history */}
          <SessionHistory onLoad={res => { applyResult(res); toast.success('Session reloaded') }} />

          {/* Current window card */}
          {current && (
            <section aria-label="Current Window Prediction">
              <PredictionCard
                prediction={current}
                windowIndex={currentIdx + 1}
                onClick={() => setInspecting(true)}
              />
              <p className="text-2xs text-clinical-text-muted dark:text-slate-500 text-center mt-1">
                Click card to inspect · use ← → in modal
              </p>
            </section>
          )}

          {/* SHAP feature explainer */}
          {current && (
            <section
              aria-label="Feature Attribution"
              className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg p-5 shadow-sm transition-colors duration-200"
            >
              <FeatureExplainer features={[]} predictedClass={current.class} />
              <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-2">
                Per-window SHAP available via single-window /api/predict endpoint.
              </p>
            </section>
          )}
        </div>

        {/* ── Right column ────────────────────────────────── */}
        <div className="col-span-12 lg:col-span-8 space-y-4">

          {/* Timeline */}
          {result && (
            <section
              aria-label="Session Timeline"
              className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg p-5 shadow-sm transition-colors duration-200"
            >
              <TimelineView trend={trend} onSelect={selectWindow} />
            </section>
          )}

          {/* Trend chart */}
          <section
            aria-label="ICP Trend"
            className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg p-5 shadow-sm transition-colors duration-200"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide">
                ICP Classification Trend
              </h2>
              {result && (
                <span className="text-xs text-clinical-text-muted dark:text-slate-400">
                  {result.predictions.length} windows · {(result.predictions.length * 10 / 60).toFixed(1)} min
                </span>
              )}
            </div>

            <TrendChart data={trend} />

            {/* Window selector */}
            {result && (
              <div className="mt-4 flex items-center gap-2">
                <label className="text-xs text-clinical-text-muted dark:text-slate-400" htmlFor="window-select">
                  Inspect window:
                </label>
                <select
                  id="window-select"
                  value={currentIdx}
                  onChange={e => selectWindow(Number(e.target.value))}
                  className="text-xs border border-clinical-border dark:border-slate-600 rounded px-2 py-1 bg-white dark:bg-slate-700 text-clinical-text-primary dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-clinical-primary dark:focus:ring-blue-400"
                  aria-label="Select window to inspect"
                >
                  {result.predictions.map((p, i) => (
                    <option key={i} value={i}>
                      #{i + 1} — {p.class_name} ({(p.confidence * 100).toFixed(0)}%)
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => setInspecting(true)}
                  className="text-xs text-clinical-primary dark:text-blue-400 hover:underline ml-1"
                >
                  Open inspector
                </button>
              </div>
            )}
          </section>

          {/* Session summary */}
          {result && (
            <section
              aria-label="Session Summary"
              className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg p-5 shadow-sm transition-colors duration-200"
            >
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide">
                  Session Summary
                </span>
                <div id="export-trigger">
                  <ExportMenu result={result} />
                </div>
              </div>
              <SessionSummary summary={result.summary} trend={trend} />
            </section>
          )}

          {/* Empty state */}
          {!result && !loading && (
            <div className="flex flex-col items-center justify-center h-64 text-clinical-text-muted dark:text-slate-400 space-y-2">
              <p className="text-sm">Upload a CSV file to begin analysis</p>
              <p className="text-xs">Keyboard shortcut: Ctrl+U</p>
            </div>
          )}
        </div>
      </div>

      {/* Inspection modal */}
      {inspecting && result && current && (
        <InspectionModal
          prediction={current}
          windowIndex={currentIdx}
          total={result.predictions.length}
          onClose={() => setInspecting(false)}
          onPrev={() => selectWindow(currentIdx - 1)}
          onNext={() => selectWindow(currentIdx + 1)}
          featureNames={result.feature_names}
        />
      )}
    </div>
  )
}
