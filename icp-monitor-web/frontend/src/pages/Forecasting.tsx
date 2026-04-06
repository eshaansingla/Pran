import { useState, useRef, useCallback, useEffect } from 'react'
import {
  Upload, TrendingUp, AlertTriangle, CheckCircle2,
  RotateCcw, ChevronDown, ChevronUp, Info, Clock,
} from 'lucide-react'
import toast from 'react-hot-toast'
import type { ForecastResult } from '../types'
import { predictForecast, fetchModelInfo } from '../utils/api'
import { probToICP } from '../utils/formatters'
import { fmtFeatureName } from '../utils/formatters'
import { useStore } from '../store/useStore'
import ForecastChart      from '../components/ForecastChart'
import AttentionHeatmap   from '../components/AttentionHeatmap'
import ForecastExportMenu from '../components/ForecastExportMenu'
import ForecastHistory    from '../components/ForecastHistory'

const FEATURE_NAMES = [
  'cardiac_amplitude', 'cardiac_frequency', 'respiratory_amplitude',
  'slow_wave_power',   'cardiac_power',     'mean_arterial_pressure',
]

// ─── CSV parse ────────────────────────────────────────────────────────────────

function parseSequenceCsv(text: string): { rows: number[][] | null; error: string | null } {
  const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0)
  const dataLines = lines[0].toLowerCase().includes('cardiac') ? lines.slice(1) : lines

  if (dataLines.length < 30) {
    return { rows: null, error: `Need at least 30 data rows (got ${dataLines.length}). Each row = one 10-second window.` }
  }

  const rows: number[][] = []
  for (let i = 0; i < dataLines.length; i++) {
    const cols = dataLines[i].split(',').map(c => parseFloat(c.trim()))
    if (cols.length !== 6 || cols.some(isNaN)) {
      return { rows: null, error: `Row ${i + 1}: expected 6 numeric columns, got "${dataLines[i]}"` }
    }
    rows.push(cols)
  }
  return { rows, error: null }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ForecastCard({ result }: { result: ForecastResult }) {
  const isAbnormal = result.class === 1
  const pct        = (result.probability * 100).toFixed(1)
  const normalPct  = (result.probabilities[0] * 100).toFixed(1)
  const abnPct     = (result.probabilities[1] * 100).toFixed(1)
  const ciLoPct    = (result.ci_lower  * 100).toFixed(1)
  const ciHiPct    = (result.ci_upper  * 100).toFixed(1)
  const estICP     = probToICP(result.probability)
  const icpLo      = probToICP(result.ci_lower)
  const icpHi      = probToICP(result.ci_upper)

  const cardBg     = isAbnormal
    ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700'
    : 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700'
  const titleColor = isAbnormal
    ? 'text-red-700 dark:text-red-400'
    : 'text-emerald-700 dark:text-emerald-400'
  const barColor   = isAbnormal ? 'bg-red-500' : 'bg-emerald-500'

  return (
    <div className={`rounded-xl border p-4 ${cardBg}`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-xs text-clinical-text-muted dark:text-slate-400 uppercase tracking-wide">
            {result.horizon_minutes}-min ICP Forecast
          </p>
          <p className={`text-2xl font-bold mt-0.5 ${titleColor}`}>{result.class_name}</p>
          <p className="text-xs text-clinical-text-muted dark:text-slate-500 mt-0.5">
            Confidence: {result.confidence_label}
          </p>
        </div>
        <div className={`p-2.5 rounded-lg ${isAbnormal ? 'bg-red-100 dark:bg-red-900/40' : 'bg-emerald-100 dark:bg-emerald-900/40'}`}>
          {isAbnormal
            ? <AlertTriangle size={24} className="text-red-600 dark:text-red-400" />
            : <CheckCircle2  size={24} className="text-emerald-600 dark:text-emerald-400" />
          }
        </div>
      </div>

      {/* Probability bar */}
      <div className="mb-3">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-emerald-700 dark:text-emerald-400 font-medium">Normal {normalPct}%</span>
          <span className="text-red-700 dark:text-red-400 font-medium">Abnormal {abnPct}%</span>
        </div>
        <div className="h-2 rounded-full bg-emerald-200 dark:bg-emerald-900/40 overflow-hidden">
          <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${pct}%` }} />
        </div>
        <p className="text-xs text-clinical-text-muted dark:text-slate-500 text-center mt-1">
          P(Abnormal) = {pct}%  ·  95% CI [{ciLoPct}%, {ciHiPct}%]
        </p>
      </div>

      {/* Estimated ICP */}
      <div className={`rounded-lg px-3 py-2 mb-3 ${isAbnormal ? 'bg-red-100/60 dark:bg-red-900/30' : 'bg-emerald-100/60 dark:bg-emerald-900/30'}`}>
        <p className="text-xs text-clinical-text-muted dark:text-slate-400 mb-0.5">Estimated ICP (mmHg)</p>
        <p className={`text-xl font-bold tabular-nums ${titleColor}`}>
          ~{estICP.toFixed(0)} mmHg
        </p>
        <p className="text-2xs text-clinical-text-muted dark:text-slate-500">
          95% CI: ~{icpLo.toFixed(0)}–{icpHi.toFixed(0)} mmHg · threshold = 15 mmHg
        </p>
      </div>

      {/* Interpretation */}
      <p className="text-xs text-clinical-text-secondary dark:text-slate-300 leading-relaxed">
        {result.interpretation}
      </p>
    </div>
  )
}


function EarlyWarningBanner({ result }: { result: ForecastResult }) {
  if (result.class !== 1 || result.probability < 0.6) return null
  return (
    <div role="alert" className="rounded-xl border border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/30 p-4">
      <div className="flex items-center gap-2 mb-2">
        <AlertTriangle size={18} className="text-red-600 dark:text-red-400 flex-shrink-0 animate-pulse-critical" />
        <h3 className="text-sm font-bold text-red-700 dark:text-red-400 uppercase tracking-wide">
          Early ICP Elevation Warning
        </h3>
      </div>
      <p className="text-sm text-red-700 dark:text-red-300 mb-3">
        LSTM model forecasts <strong>Abnormal ICP</strong> within approximately{' '}
        <strong>{result.horizon_minutes} minutes</strong> with{' '}
        {(result.probability * 100).toFixed(0)}% probability
        (est. ~{probToICP(result.probability).toFixed(0)} mmHg).
      </p>
      <ul className="text-xs text-red-700 dark:text-red-400 space-y-1 list-disc list-inside">
        <li>Verify patient positioning (head elevated 30°)</li>
        <li>Check vital signs and neurological status</li>
        <li>Review current medications and fluid balance</li>
        <li>Prepare for potential clinical intervention</li>
        <li>Confirm with qualified medical professional before acting</li>
      </ul>
      <p className="text-2xs text-red-500 dark:text-red-500 mt-3">
        Research prototype — NOT FDA-approved. All decisions require qualified clinician judgement.
      </p>
    </div>
  )
}


function FeatureHighlights({ result }: { result: ForecastResult }) {
  return (
    <div>
      <h4 className="text-xs font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide mb-2">
        Key Driving Features
      </h4>
      <div className="space-y-1.5">
        {result.feature_highlights.map((fh, i) => (
          <div key={fh.name} className="flex items-center gap-2">
            <span className="w-4 h-4 rounded-full text-2xs flex items-center justify-center font-semibold
              bg-clinical-primary dark:bg-blue-600 text-white flex-shrink-0">
              {i + 1}
            </span>
            <span className="text-xs text-clinical-text-secondary dark:text-slate-300 flex-1">
              {fmtFeatureName(fh.name)}
            </span>
            <div className="w-16 h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
              <div className="h-full rounded-full bg-clinical-primary dark:bg-blue-500"
                style={{ width: `${(fh.importance * 100).toFixed(0)}%` }} />
            </div>
            <span className="text-2xs tabular-nums text-clinical-text-muted dark:text-slate-500 w-8 text-right">
              {(fh.importance * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}


function SessionMeta({ result, sequence, fileName }: {
  result: ForecastResult
  sequence: number[][]
  fileName: string
}) {
  const now = new Date().toLocaleString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: false,
  })
  return (
    <div className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-xl p-4 shadow-sm">
      <h4 className="text-xs font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide mb-2">
        Session Info
      </h4>
      <div className="space-y-1">
        {[
          { label: 'File',       value: fileName },
          { label: 'Timestamp', value: now },
          { label: 'Windows',   value: `${sequence.length} (${(sequence.length * 10 / 60).toFixed(1)} min history)` },
          { label: 'Model',     value: `LSTM v${result.model_version}` },
          { label: 'Threshold', value: `${result.threshold.toFixed(4)} (F1-opt.)` },
        ].map(({ label, value }) => (
          <div key={label} className="flex justify-between text-xs">
            <span className="text-clinical-text-muted dark:text-slate-500">{label}</span>
            <span className="font-medium text-clinical-text-primary dark:text-slate-200 font-mono text-right max-w-[60%] truncate">
              {value}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}


// ─── Main page ────────────────────────────────────────────────────────────────

type PageState = 'idle' | 'loading' | 'done' | 'error'

export default function Forecasting() {
  const { addForecast } = useStore()

  const [state,       setState]      = useState<PageState>('idle')
  const [sequence,    setSequence]   = useState<number[][] | null>(null)
  const [fileName,    setFileName]   = useState<string>('')
  const [result,      setResult]     = useState<ForecastResult | null>(null)
  const [errorMsg,    setErrorMsg]   = useState<string>('')
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [globalImportances, setGlobalImportances] = useState<Record<string, number>>({})

  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchModelInfo()
      .then(info => setGlobalImportances(info.global_importances))
      .catch(() => {})
  }, [])

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      toast.error('Only .csv files are accepted')
      return
    }
    const text = await file.text()
    const { rows, error } = parseSequenceCsv(text)
    if (error || !rows) { setErrorMsg(error ?? 'Parse error'); setState('error'); return }

    setSequence(rows)
    setFileName(file.name)
    setErrorMsg('')
    setState('loading')

    try {
      const res = await predictForecast(rows)
      setResult(res)
      setState('done')
      addForecast(res, file.name, rows.length)
      toast.success(`Forecast complete — ${res.class_name} in ${res.horizon_minutes} min`)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setErrorMsg(msg)
      setState('error')
      toast.error('Forecast failed: ' + msg)
    }
  }, [addForecast])

  function onFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    e.target.value = ''
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  function reset() {
    setSequence(null); setResult(null); setFileName(''); setErrorMsg(''); setState('idle')
  }

  function reloadForecast(res: ForecastResult, fname: string, seqLen: number) {
    // Reconstruct a placeholder sequence of the right length (no raw data stored)
    setResult(res)
    setFileName(fname)
    setSequence(Array.from({ length: seqLen }, () => Array(6).fill(0) as number[]))
    setState('done')
    toast.success('Forecast reloaded from history')
  }

  // ── Idle / upload screen ──────────────────────────────────────────────────
  if (state === 'idle' || state === 'error') {
    return (
      <div className="max-w-2xl space-y-4">
        <div>
          <h1 className="text-base font-semibold text-clinical-text-primary dark:text-slate-100">
            ICP Trend Forecasting
          </h1>
          <p className="text-sm text-clinical-text-muted dark:text-slate-400 mt-0.5">
            LSTM — {15}-minute ahead prediction · BiLSTM(64→32) + self-attention · MC Dropout
          </p>
        </div>

        {/* Drop zone */}
        <div
          onDrop={onDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-clinical-border dark:border-slate-600 rounded-xl
            p-10 text-center cursor-pointer hover:border-purple-400 dark:hover:border-purple-500
            transition-colors bg-white dark:bg-slate-800"
          role="button"
          aria-label="Upload sequence CSV"
        >
          <Upload size={32} className="mx-auto mb-3 text-clinical-text-muted dark:text-slate-500" />
          <p className="text-sm font-medium text-clinical-text-primary dark:text-slate-200">
            Drop sequence CSV here or click to browse
          </p>
          <p className="text-xs text-clinical-text-muted dark:text-slate-400 mt-1">
            Minimum 30 consecutive rows (each = one 10-second window) · 6 feature columns
          </p>
          <p className="text-xs font-mono text-clinical-text-muted dark:text-slate-500 mt-2">
            {FEATURE_NAMES.join(', ')}
          </p>
        </div>
        <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={onFileInput} />

        {state === 'error' && (
          <div className="rounded-xl border border-red-200 dark:border-red-700 bg-red-50 dark:bg-red-900/20 p-4">
            <p className="text-sm font-semibold text-red-700 dark:text-red-400 mb-1">Upload error</p>
            <p className="text-xs text-red-600 dark:text-red-300">{errorMsg}</p>
          </div>
        )}

        {/* CSV format guide */}
        <div className="rounded-xl border border-clinical-border dark:border-slate-700 bg-white dark:bg-slate-800 p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Info size={13} className="text-purple-600 dark:text-purple-400" />
            <h3 className="text-xs font-semibold text-clinical-text-secondary dark:text-slate-300">CSV Format</h3>
          </div>
          <pre className="text-xs font-mono text-clinical-text-muted dark:text-slate-400 bg-slate-50 dark:bg-slate-900 rounded p-3 overflow-x-auto">
{`cardiac_amplitude,cardiac_frequency,respiratory_amplitude,slow_wave_power,cardiac_power,mean_arterial_pressure
32.4,1.2,8.7,1.30,2.10,95.0
28.1,1.1,7.2,1.65,2.55,92.0
45.6,1.3,12.3,1.80,3.20,98.0
...  (minimum 30 rows)`}
          </pre>
          <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-2">
            Header optional. Same column order as ICP Classification tab.
            Export 30+ rows from a classification session to use here.
          </p>
        </div>

        {/* Forecast history */}
        <ForecastHistory onLoad={reloadForecast} />
      </div>
    )
  }

  // ── Loading ───────────────────────────────────────────────────────────────
  if (state === 'loading') {
    return (
      <div className="max-w-2xl flex flex-col items-center justify-center py-20 space-y-4">
        <TrendingUp size={40} className="text-purple-600 dark:text-purple-400 animate-pulse" />
        <p className="text-sm font-medium text-clinical-text-primary dark:text-slate-200">
          Running LSTM forecast…
        </p>
        <p className="text-xs text-clinical-text-muted dark:text-slate-400">
          Monte Carlo dropout · 20 stochastic passes · computing uncertainty
        </p>
      </div>
    )
  }

  // ── Results ───────────────────────────────────────────────────────────────
  if (state === 'done' && result && sequence) {
    return (
      <div className="max-w-4xl space-y-4 animate-fade-in-up">

        {/* Header row */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-base font-semibold text-clinical-text-primary dark:text-slate-100">
              ICP Trend Forecast
            </h1>
            <p className="text-xs text-clinical-text-muted dark:text-slate-400 mt-0.5 flex items-center gap-1">
              <Clock size={10} />
              {fileName} · {sequence.length} windows ({(sequence.length * 10 / 60).toFixed(1)} min)
              · LSTM v{result.model_version} · {result.horizon_minutes}-min ahead
            </p>
          </div>
          <div className="flex items-center gap-2">
            <ForecastExportMenu result={result} sequence={sequence} fileName={fileName} />
            <button
              onClick={reset}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium
                text-clinical-text-secondary dark:text-slate-400 border border-clinical-border
                dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
            >
              <RotateCcw size={12} /> New Forecast
            </button>
          </div>
        </div>

        {/* Early warning */}
        <EarlyWarningBanner result={result} />

        {/* Main 5-col grid: cards (2) + chart (3) */}
        <div className="grid grid-cols-5 gap-4">
          <div className="col-span-2 space-y-4">
            <ForecastCard result={result} />
            <div className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-xl p-4 shadow-sm">
              <FeatureHighlights result={result} />
            </div>
            <SessionMeta result={result} sequence={sequence} fileName={fileName} />
          </div>

          <div className="col-span-3 bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-xl p-4 shadow-sm">
            <h3 className="text-xs font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide mb-3">
              ICP Probability &amp; Estimated mmHg Trajectory
            </h3>
            <ForecastChart sequence={sequence} result={result} />
          </div>
        </div>

        {/* Attention heatmap (collapsible) */}
        <div className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-xl shadow-sm overflow-hidden">
          <button
            onClick={() => setShowHeatmap(h => !h)}
            className="w-full flex items-center justify-between px-4 py-3
              text-xs font-semibold text-clinical-text-secondary dark:text-slate-300
              uppercase tracking-wide hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
          >
            <span>Attention Heatmap — Temporal Feature Importance</span>
            {showHeatmap ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {showHeatmap && (
            <div className="px-4 pb-4">
              <AttentionHeatmap
                attentionWeights={result.attention_weights}
                featureNames={FEATURE_NAMES}
                globalImportances={globalImportances}
              />
            </div>
          )}
        </div>

        {/* Forecast history */}
        <ForecastHistory onLoad={reloadForecast} />

        {/* Disclaimer */}
        <div className="rounded-xl border border-amber-200 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20 px-4 py-3">
          <p className="text-xs text-amber-700 dark:text-amber-400">
            <strong>Research prototype.</strong> LSTM trained on CHARIS + MIMIC-III research data.
            Estimated ICP values are probabilistic approximations — not direct measurements.
            Requires clinical validation. Not FDA-approved. All decisions must be made by qualified clinicians.
          </p>
        </div>
      </div>
    )
  }

  return null
}
