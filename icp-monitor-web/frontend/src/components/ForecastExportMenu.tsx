import { useState, useRef, useEffect } from 'react'
import { FileDown, ChevronDown, FileText, Table } from 'lucide-react'
import type { ForecastResult } from '../types'
import { downloadBlob, probToICP } from '../utils/formatters'
import { fmtFeatureName } from '../utils/formatters'

interface Props {
  result:   ForecastResult
  sequence: number[][]
  fileName: string
}

const FEATURE_NAMES = [
  'cardiac_amplitude', 'cardiac_frequency', 'respiratory_amplitude',
  'slow_wave_power',   'cardiac_power',     'mean_arterial_pressure',
]

function exportForecastCSV(result: ForecastResult, sequence: number[][], fileName: string) {
  const now = new Date().toLocaleString('en-GB', { hour12: false })

  // Summary rows
  const summary = [
    '# ICP FORECAST EXPORT',
    `# Source file: ${fileName}`,
    `# Generated: ${now}`,
    `# Model: LSTM v${result.model_version}`,
    `# Horizon: ${result.horizon_minutes} min ahead`,
    '',
    '## FORECAST RESULT',
    'field,value',
    `predicted_class,${result.class_name}`,
    `probability_abnormal,${result.probability.toFixed(4)}`,
    `probability_normal,${result.probabilities[0].toFixed(4)}`,
    `ci_lower,${result.ci_lower.toFixed(4)}`,
    `ci_upper,${result.ci_upper.toFixed(4)}`,
    `confidence_label,${result.confidence_label}`,
    `estimated_icp_mmhg,${probToICP(result.probability).toFixed(1)}`,
    `ci_icp_lower_mmhg,${probToICP(result.ci_lower).toFixed(1)}`,
    `ci_icp_upper_mmhg,${probToICP(result.ci_upper).toFixed(1)}`,
    `horizon_minutes,${result.horizon_minutes}`,
    `threshold,${result.threshold}`,
    `timestamp,${result.timestamp}`,
    '',
    '## FEATURE HIGHLIGHTS',
    'rank,feature,importance_pct',
    ...result.feature_highlights.map((fh, i) =>
      `${i + 1},${fmtFeatureName(fh.name)},${(fh.importance * 100).toFixed(1)}`
    ),
    '',
    '## ATTENTION WEIGHTS (most to least attended)',
    'timestep_relative_s,attention_weight',
    ...result.attention_weights.map((w, i) => {
      const tSec = -(result.seq_len - 1 - i) * 10
      return `${tSec},${w.toFixed(6)}`
    }),
    '',
    '## INPUT SEQUENCE',
    FEATURE_NAMES.join(','),
    ...sequence.map(row => row.map(v => v.toFixed(4)).join(',')),
  ]

  downloadBlob(
    new Blob([summary.join('\n')], { type: 'text/csv' }),
    `ICP_Forecast_${fileName.replace('.csv', '')}_${Date.now()}.csv`
  )
}

async function exportForecastPDF(result: ForecastResult, sequence: number[][], fileName: string) {
  const { jsPDF } = await import('jspdf')
  const doc = new jsPDF({ unit: 'mm', format: 'a4' })
  const now = new Date().toLocaleString('en-GB', { hour12: false })
  const isAbn = result.class === 1
  const estICP = probToICP(result.probability)

  let y = 18
  const L = 14
  const W = 182

  const line = (text: string, opts?: { bold?: boolean; size?: number; color?: [number,number,number] }) => {
    doc.setFont('helvetica', opts?.bold ? 'bold' : 'normal')
    doc.setFontSize(opts?.size ?? 10)
    if (opts?.color) doc.setTextColor(...opts.color)
    else doc.setTextColor(26, 32, 44)
    doc.text(text, L, y)
    y += (opts?.size ?? 10) * 0.45 + 2
  }

  const rule = () => {
    doc.setDrawColor(226, 232, 240)
    doc.line(L, y, L + W, y)
    y += 4
  }

  const gap = (n = 4) => { y += n }

  // ── Header bar ──────────────────────────────────────────────────────────
  doc.setFillColor(88, 28, 135)   // purple for LSTM
  doc.rect(0, 0, 210, 14, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(255, 255, 255)
  doc.text('ICP FORECAST REPORT — LSTM EARLY WARNING SYSTEM', 14, 9)
  doc.setFontSize(7)
  doc.setFont('helvetica', 'normal')
  doc.text(`NOT FOR DIAGNOSTIC USE  |  Research Prototype  |  LSTM v${result.model_version}`, 14, 12.5)
  y = 22

  line('GENERATED: ' + now, { bold: true, size: 9 })
  line(`SOURCE FILE: ${fileName}`, { size: 9, color: [74, 85, 104] })
  line(`FORECAST HORIZON: ${result.horizon_minutes} minutes ahead`, { size: 9, color: [74, 85, 104] })
  gap(3); rule()

  // ── Forecast result ─────────────────────────────────────────────────────
  line('FORECAST RESULT', { bold: true, size: 12 })
  gap(2)
  const predColor: [number,number,number] = isAbn ? [220, 38, 38] : [5, 150, 105]
  line(`Predicted Class       : ${result.class_name}`, { bold: true, size: 11, color: predColor })
  line(`P(Abnormal ICP)       : ${(result.probability * 100).toFixed(1)}%`, { size: 10 })
  line(`P(Normal ICP)         : ${(result.probabilities[0] * 100).toFixed(1)}%`, { size: 10 })
  line(`95% Confidence Interval: [${(result.ci_lower * 100).toFixed(1)}%, ${(result.ci_upper * 100).toFixed(1)}%]`, { size: 10 })
  line(`Confidence Level      : ${result.confidence_label}`, { size: 10 })
  gap(2)
  line(`Estimated ICP         : ~${estICP.toFixed(0)} mmHg`, { bold: true, size: 11, color: predColor })
  line(`ICP CI Range          : ~${probToICP(result.ci_lower).toFixed(0)}–${probToICP(result.ci_upper).toFixed(0)} mmHg`, { size: 10 })
  line(`ICP Threshold         : ${result.threshold.toFixed(4)} probability (= 15 mmHg clinical threshold)`, { size: 9, color: [74, 85, 104] })
  gap(2)
  line(`Interpretation: ${result.interpretation}`, { size: 9, color: [74, 85, 104] })
  gap(3); rule()

  // ── Early warning ────────────────────────────────────────────────────────
  if (isAbn && result.probability >= 0.6) {
    line('EARLY WARNING TRIGGERED', { bold: true, size: 11, color: [220, 38, 38] })
    gap(2)
    line(`Abnormal ICP forecast within ${result.horizon_minutes} minutes with ${(result.probability * 100).toFixed(0)}% probability.`, { size: 10 })
    line('Clinical actions to consider:', { size: 10 })
    gap(1)
    const actions = [
      'Verify patient positioning (head elevated 30 degrees)',
      'Check vital signs and neurological status',
      'Review current medications and fluid balance',
      'Consider neurology/neurosurgery consultation',
      'Confirm findings with qualified medical professional before acting',
    ]
    actions.forEach(a => {
      line('  - ' + a, { size: 9, color: [153, 27, 27] })
    })
    gap(3); rule()
  }

  // ── Input sequence summary ───────────────────────────────────────────────
  line('INPUT SEQUENCE SUMMARY', { bold: true, size: 11 })
  gap(2)
  line(`Sequence Length   : ${sequence.length} windows (${(sequence.length * 10 / 60).toFixed(1)} min history)`, { size: 10 })
  line(`Window Duration   : 10 seconds`, { size: 10 })
  line(`Features Used     : ${result.seq_len} timesteps × 6 physiological features`, { size: 10 })

  // Feature means
  gap(2)
  const featureMeans = FEATURE_NAMES.map((name, j) => {
    const vals = sequence.map(row => row[j])
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length
    return { name, mean }
  })
  line('Feature Means (over input window):', { bold: true, size: 9 })
  gap(1)
  featureMeans.forEach(({ name, mean }) => {
    line(`  ${fmtFeatureName(name).padEnd(30)} : ${mean.toFixed(3)}`, { size: 9 })
  })
  gap(3); rule()

  // ── Feature highlights ────────────────────────────────────────────────────
  line('KEY DRIVING FEATURES (LSTM Attention)', { bold: true, size: 11 })
  gap(2)
  result.feature_highlights.forEach((fh, i) => {
    line(`${i + 1}. ${fmtFeatureName(fh.name).padEnd(35)} ${(fh.importance * 100).toFixed(1)}%`, { size: 10 })
  })
  gap(3); rule()

  // ── Top attended timesteps ────────────────────────────────────────────────
  line('TOP ATTENDED TIMESTEPS', { bold: true, size: 11 })
  gap(2)
  const attnWithTime = result.attention_weights
    .map((w, i) => ({ tSec: -(result.seq_len - 1 - i) * 10, weight: w }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 5)
  attnWithTime.forEach((a, i) => {
    const label = a.tSec === 0 ? 'Now' : `${a.tSec}s ago`
    line(`${i + 1}. t = ${label.padEnd(12)} attention = ${(a.weight * 100).toFixed(2)}%`, { size: 10 })
  })
  gap(3); rule()

  // ── Model info ────────────────────────────────────────────────────────────
  line('MODEL INFORMATION', { bold: true, size: 11 })
  gap(2)
  line(`Model Type        : Bidirectional LSTM with Self-Attention`, { size: 10 })
  line(`Version           : ${result.model_version}`, { size: 10 })
  line(`Forecast Horizon  : ${result.horizon_minutes} minutes`, { size: 10 })
  line(`Uncertainty Est.  : Monte Carlo Dropout (20 passes)`, { size: 10 })
  line(`Decision Threshold: ${result.threshold.toFixed(4)} (F1-optimised)`, { size: 10 })
  line(`Training Data     : CHARIS + MIMIC-III (patient-stratified)`, { size: 10 })
  gap(3); rule()

  // ── Disclaimer ────────────────────────────────────────────────────────────
  line('IMPORTANT DISCLAIMER', { bold: true, size: 11, color: [220, 38, 38] })
  gap(2)
  const disc = [
    'This system is a clinical decision SUPPORT tool only.',
    'NOT FDA-approved. NOT for autonomous diagnostic use.',
    'All clinical decisions must be made by qualified medical professionals.',
    'Estimated ICP values are probabilistic approximations, not direct measurements.',
    'Validate rigorously before any clinical deployment.',
  ]
  disc.forEach(d => line(d, { size: 9, color: [153, 27, 27] }))

  doc.save(`ICP_Forecast_Report_${fileName.replace('.csv', '')}_${Date.now()}.pdf`)
}

export default function ForecastExportMenu({ result, sequence, fileName }: Props) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(o => !o)}
        aria-label="Export forecast"
        aria-haspopup="true"
        aria-expanded={open}
        className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium
          bg-purple-600 dark:bg-purple-700 text-white rounded-lg
          hover:bg-purple-700 dark:hover:bg-purple-600
          focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2
          transition-colors"
      >
        <FileDown size={13} />
        Export
        <ChevronDown size={11} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute right-0 mt-1 w-44 bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg shadow-lg z-20 overflow-hidden">
          <button
            onClick={() => { exportForecastCSV(result, sequence, fileName); setOpen(false) }}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-clinical-text-primary dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-left"
          >
            <Table size={14} className="text-clinical-text-muted dark:text-slate-400" />
            Export CSV
          </button>
          <button
            onClick={() => { exportForecastPDF(result, sequence, fileName); setOpen(false) }}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-clinical-text-primary dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-left border-t border-clinical-border dark:border-slate-700"
          >
            <FileText size={14} className="text-clinical-text-muted dark:text-slate-400" />
            Export PDF
          </button>
        </div>
      )}
    </div>
  )
}
