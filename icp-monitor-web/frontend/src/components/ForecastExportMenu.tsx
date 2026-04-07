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
    `estimated_icp_mmhg,${probToICP(result.probability, result.threshold).toFixed(1)}`,
    `ci_icp_lower_mmhg,${probToICP(result.ci_lower, result.threshold).toFixed(1)}`,
    `ci_icp_upper_mmhg,${probToICP(result.ci_upper, result.threshold).toFixed(1)}`,
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

// ── MAP z-score reconstruction (mirrors ForecastChart/ForecastWindowAnalysis) ─
const MAP_MEAN_PDF = 88.352
const MAP_STD_PDF  = 7.850
function toRawMAPpdf(v: number): number { return v > 50 ? v : v * MAP_STD_PDF + MAP_MEAN_PDF }

// ── Lundberg grade helper ─────────────────────────────────────────────────────
function icpGradePDF(icp: number): { label: string; color: [number,number,number] } {
  if (icp < 7)  return { label: 'Sub-normal',  color: [37, 99, 235] }
  if (icp < 15) return { label: 'Normal',       color: [5, 150, 105] }
  if (icp < 20) return { label: 'Grade I ⚠',   color: [217, 119, 6] }
  if (icp < 40) return { label: 'Grade II ✗',  color: [220, 38, 38] }
  return           { label: 'Grade III ✗✗',    color: [124, 45, 18] }
}

async function exportForecastPDF(result: ForecastResult, sequence: number[][], fileName: string) {
  const { jsPDF } = await import('jspdf')
  const doc    = new jsPDF({ unit: 'mm', format: 'a4' })
  const now    = new Date().toLocaleString('en-GB', { hour12: false })
  const isAbn  = result.class === 1
  const thr    = result.threshold
  const estICP = probToICP(result.probability, thr)
  const ciLoICP = probToICP(result.ci_lower, thr)
  const ciHiICP = probToICP(result.ci_upper, thr)
  const grade  = icpGradePDF(estICP)
  const predColor: [number,number,number] = isAbn ? [220, 38, 38] : [5, 150, 105]

  // Compute MAP stats from sequence
  const mapVals  = sequence.map(r => toRawMAPpdf(r[5])).filter(v => v > 0)
  const mapMean  = mapVals.length ? mapVals.reduce((a, b) => a + b, 0) / mapVals.length : 0
  const mapMin   = mapVals.length ? Math.min(...mapVals) : 0
  const mapMax   = mapVals.length ? Math.max(...mapVals) : 0
  const icpMean  = mapMean > 0 ? Math.max(5, Math.min(40, mapMean - 70)) : estICP

  let y = 18
  const L = 14
  const W = 182
  const R = L + W   // right margin

  // ── helpers ────────────────────────────────────────────────────────────────
  type TextOpts = { bold?: boolean; size?: number; color?: [number,number,number]; align?: 'left'|'right'; x?: number }
  const text = (t: string, opts?: TextOpts) => {
    doc.setFont('helvetica', opts?.bold ? 'bold' : 'normal')
    doc.setFontSize(opts?.size ?? 10)
    doc.setTextColor(...(opts?.color ?? [26, 32, 44]))
    const xPos = opts?.x ?? (opts?.align === 'right' ? R : L)
    doc.text(t, xPos, y, opts?.align === 'right' ? { align: 'right' } : {})
  }
  const line = (t: string, opts?: TextOpts) => {
    text(t, opts)
    y += (opts?.size ?? 10) * 0.45 + 2
  }
  const rule = (color: [number,number,number] = [226, 232, 240]) => {
    doc.setDrawColor(...color); doc.line(L, y, R, y); y += 4
  }
  const gap = (n = 4) => { y += n }
  const twoCol = (left: string, right: string, opts?: TextOpts, rightOpts?: TextOpts) => {
    text(left, opts)
    text(right, { ...(rightOpts ?? opts), align: 'right' })
    y += ((opts?.size ?? 10) * 0.45 + 2)
  }

  // ── Cover header bar ───────────────────────────────────────────────────────
  doc.setFillColor(88, 28, 135)
  doc.rect(0, 0, 210, 18, 'F')
  doc.setFont('helvetica', 'bold'); doc.setFontSize(13); doc.setTextColor(255, 255, 255)
  doc.text('NON-INVASIVE ICP FORECAST REPORT', 14, 9)
  doc.setFont('helvetica', 'normal'); doc.setFontSize(8)
  doc.text('LSTM Early Warning System  |  Research Prototype  |  NOT FOR DIAGNOSTIC USE', 14, 14)
  doc.text(`v${result.model_version}`, 196, 9, { align: 'right' })
  y = 24

  // Report meta
  twoCol('Generated:', now, { size: 9, color: [74, 85, 104], bold: true }, { size: 9, color: [74, 85, 104] })
  twoCol('Source file:', fileName, { size: 9, color: [74, 85, 104], bold: true }, { size: 9, color: [74, 85, 104] })
  gap(1); rule()

  // ── Section 1: FORECAST SUMMARY (prominent) ─────────────────────────────
  line('1. FORECAST SUMMARY', { bold: true, size: 12 })
  gap(2)

  // Status box
  const boxH = 22
  const boxColor: [number,number,number] = isAbn ? [254, 226, 226] : [209, 250, 229]
  const boxBorder: [number,number,number] = isAbn ? [220, 38, 38] : [5, 150, 105]
  doc.setFillColor(...boxColor)
  doc.setDrawColor(...boxBorder)
  doc.roundedRect(L, y, W, boxH, 2, 2, 'FD')
  const saveY = y
  y += 5
  line(`Predicted Class:  ${result.class_name}`, { bold: true, size: 13, color: predColor })
  text(`${result.horizon_minutes}-min horizon  |  Confidence: ${result.confidence_label}`, { size: 9, color: [74, 85, 104] })
  text(`P(Abnormal) = ${(result.probability * 100).toFixed(1)}%`, { size: 9, color: [74, 85, 104], align: 'right' })
  y = saveY + boxH + 4

  // ICP result table with reference ranges
  gap(1)
  line('ICP Estimates vs. Clinical Reference Ranges:', { bold: true, size: 10 })
  gap(1)

  // Table header
  doc.setFillColor(241, 245, 249)
  doc.rect(L, y - 1, W, 6, 'F')
  text('Parameter',          { size: 9, bold: true, color: [51, 65, 85] })
  text('Patient Value',      { size: 9, bold: true, color: [51, 65, 85], x: L + 60 })
  text('Reference Range',    { size: 9, bold: true, color: [51, 65, 85], x: L + 100 })
  text('Status',             { size: 9, bold: true, color: [51, 65, 85], x: L + 145 })
  y += 6

  // Helper to draw one reference row
  const refRow = (
    param: string, value: string, refRange: string, status: string,
    abnormal: boolean
  ) => {
    doc.setDrawColor(226, 232, 240); doc.line(L, y - 1, R, y - 1)
    const rowColor: [number,number,number] = abnormal ? [254, 242, 242] : [240, 253, 244]
    doc.setFillColor(...rowColor)
    doc.rect(L, y - 1, W, 6, 'F')
    text(param,    { size: 9, color: [51, 65, 85] })
    text(value,    { size: 9, bold: abnormal, color: abnormal ? [220, 38, 38] : [5, 150, 105], x: L + 60 })
    text(refRange, { size: 9, color: [100, 116, 139], x: L + 100 })
    text(status,   { size: 9, bold: abnormal, color: abnormal ? [220, 38, 38] : [5, 150, 105], x: L + 145 })
    y += 6
  }

  const icpAbn  = estICP >= 15
  const icpLoAbn = ciLoICP >= 15
  const icpHiAbn = ciHiICP >= 20
  const mapAbn  = mapMean > 0 && (mapMean < 70 || mapMean > 105)
  const icpMeanAbn = icpMean >= 15

  refRow('Est. ICP (LSTM model)',     `~${estICP.toFixed(0)} mmHg`,              '7–15 mmHg (supine adult)',    icpAbn  ? `${grade.label} — OUT OF RANGE`  : 'WITHIN NORMAL RANGE',    icpAbn)
  refRow('ICP 95% CI lower',          `~${ciLoICP.toFixed(0)} mmHg`,             '7–15 mmHg',                   icpLoAbn ? 'Elevated'    : 'Normal',                     icpLoAbn)
  refRow('ICP 95% CI upper',          `~${ciHiICP.toFixed(0)} mmHg`,             '< 20 mmHg (mild concern)',    icpHiAbn ? 'ELEVATED'    : 'Acceptable',                  icpHiAbn)
  if (mapVals.length > 0) {
    refRow('Mean MAP (history)',       `${mapMean.toFixed(1)} mmHg`,              '70–105 mmHg',                 mapAbn   ? 'OUT OF RANGE' : 'Normal',                    mapAbn)
    refRow('Est. ICP (MAP−70, mean)', `~${icpMean.toFixed(0)} mmHg`,             '7–15 mmHg',                   icpMeanAbn ? 'Elevated'  : 'Normal',                     icpMeanAbn)
  }
  gap(1)

  // Lundberg classification table
  line('Lundberg ICP Classification:', { bold: true, size: 9 })
  gap(1)
  const lundRows: Array<[string, string, boolean]> = [
    ['Normal',  '7–15 mmHg  (asymptomatic)',         !icpAbn],
    ['Grade I', '15–20 mmHg  (mild hypertension)',    estICP >= 15 && estICP < 20],
    ['Grade II','20–40 mmHg  (moderate, treat)',       estICP >= 20 && estICP < 40],
    ['Grade III','>40 mmHg   (critical, Lundberg A)', estICP >= 40],
  ]
  lundRows.forEach(([grade, desc, current]) => {
    const hl: [number,number,number] = current ? [88, 28, 135] : [148, 163, 184]
    text(current ? '▶ ' + grade : '  ' + grade, { size: 8, bold: current, color: hl })
    text(desc, { size: 8, color: hl, x: L + 28 })
    y += 5
  })
  gap(2); rule()

  // ── Section 2: PROBABILITY & UNCERTAINTY ──────────────────────────────────
  line('2. PROBABILITY & UNCERTAINTY', { bold: true, size: 12 })
  gap(2)
  twoCol('P(Abnormal ICP):', `${(result.probability * 100).toFixed(1)}%  ${isAbn ? '← ELEVATED' : '← within normal'}`,
    { size: 10 }, { size: 10, bold: isAbn, color: predColor })
  twoCol('P(Normal ICP):',   `${(result.probabilities[0] * 100).toFixed(1)}%`,
    { size: 10 }, { size: 10 })
  twoCol('95% CI (probability):', `[${(result.ci_lower * 100).toFixed(1)}%, ${(result.ci_upper * 100).toFixed(1)}%]`,
    { size: 10 }, { size: 10 })
  twoCol('Confidence level:',    result.confidence_label,
    { size: 10 }, { size: 10, bold: result.confidence_label === 'Low', color: result.confidence_label === 'Low' ? [220, 38, 38] : [26, 32, 44] })
  twoCol('Decision threshold:',  `${thr.toFixed(4)} (P ≥ threshold → Abnormal)`,
    { size: 10 }, { size: 9, color: [74, 85, 104] })
  gap(1)
  line(`Interpretation: ${result.interpretation}`, { size: 9, color: [74, 85, 104] })
  gap(3); rule()

  // ── Section 3: CLINICAL ALERT ─────────────────────────────────────────────
  if (isAbn && result.probability >= result.threshold) {
    doc.setFillColor(254, 226, 226)
    doc.setDrawColor(220, 38, 38)
    const alertStart = y
    y += 5
    line('⚠  EARLY ICP ELEVATION WARNING', { bold: true, size: 11, color: [153, 27, 27] })
    line(`LSTM forecasts Abnormal ICP within ${result.horizon_minutes} min  |  P = ${(result.probability * 100).toFixed(0)}%  |  Est. ICP ~${estICP.toFixed(0)} mmHg (${grade.label})`,
      { size: 9, color: [153, 27, 27] })
    gap(1)
    line('Recommended clinical actions:', { size: 9, bold: true, color: [153, 27, 27] })
    const actions = [
      'Head of bed elevated 30° — reduces ICP by 5–10 mmHg',
      'Verify airway patency; ensure adequate oxygenation (SpO₂ > 95%)',
      'Review sedation, analgesia, and osmotherapy orders',
      'Consider neurosurgery / neurology consultation immediately',
      'Confirm all findings with a qualified medical professional before acting',
    ]
    actions.forEach(a => { line('  • ' + a, { size: 9, color: [153, 27, 27] }) })
    const alertEnd = y + 2
    doc.roundedRect(L, alertStart, W, alertEnd - alertStart, 2, 2, 'FD')
    y = alertEnd + 4; rule()
  } else {
    rule()
  }

  // ── Section 4: INPUT SEQUENCE ANALYSIS ───────────────────────────────────
  // Add new page if needed
  if (y > 220) { doc.addPage(); y = 18 }
  line('3. INPUT SEQUENCE ANALYSIS', { bold: true, size: 12 })
  gap(2)
  twoCol('Sequence length:', `${sequence.length} windows × 10 s = ${(sequence.length * 10 / 60).toFixed(1)} min history`, { size: 10 }, { size: 10 })
  twoCol('Timesteps used:',  `${result.seq_len} (most recent ${(result.seq_len * 10 / 60).toFixed(1)} min)`, { size: 10 }, { size: 10 })
  gap(2)

  const featureMeans = FEATURE_NAMES.map((name, j) => {
    const vals = sequence.map(row => row[j]).filter(v => v !== 0)
    const mean = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0
    const std  = vals.length > 1
      ? Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (vals.length - 1))
      : 0
    return { name, mean, std }
  })

  line('Feature Statistics (mean ± std over input window):', { bold: true, size: 9 })
  gap(1)
  doc.setFillColor(241, 245, 249)
  doc.rect(L, y - 1, W, 6, 'F')
  text('Feature',    { size: 9, bold: true, color: [51, 65, 85] })
  text('Mean',       { size: 9, bold: true, color: [51, 65, 85], x: L + 90 })
  text('Std Dev',    { size: 9, bold: true, color: [51, 65, 85], x: L + 125 })
  text('Min → Max',  { size: 9, bold: true, color: [51, 65, 85], x: L + 155 })
  y += 6

  featureMeans.forEach(({ name, mean, std }, j) => {
    const vals = sequence.map(r => r[j]).filter(v => v !== 0)
    const mn = vals.length ? Math.min(...vals) : 0
    const mx = vals.length ? Math.max(...vals) : 0
    const isMapFeature = j === 5
    if (isMapFeature) {
      doc.setFillColor(255, 251, 235)
    } else {
      doc.setFillColor(248, 250, 252)
    }
    doc.rect(L, y - 1, W, 6, 'F')
    text(fmtFeatureName(name), { size: 9, color: isMapFeature ? [180, 83, 9] : [51, 65, 85], bold: isMapFeature })
    text(mean.toFixed(2),      { size: 9, color: [51, 65, 85], x: L + 90 })
    text(`±${std.toFixed(2)}`, { size: 9, color: [100, 116, 139], x: L + 125 })
    text(`${mn.toFixed(1)} → ${mx.toFixed(1)}`, { size: 9, color: [100, 116, 139], x: L + 155 })
    y += 6
  })
  gap(3); rule()

  // ── Section 5: ATTENTION & FEATURE IMPORTANCE ─────────────────────────────
  if (y > 220) { doc.addPage(); y = 18 }
  line('4. KEY DRIVING FEATURES (LSTM Attention)', { bold: true, size: 12 })
  gap(2)
  doc.setFillColor(241, 245, 249)
  doc.rect(L, y - 1, W, 6, 'F')
  text('Rank', { size: 9, bold: true, color: [51, 65, 85] })
  text('Feature', { size: 9, bold: true, color: [51, 65, 85], x: L + 16 })
  text('Importance', { size: 9, bold: true, color: [51, 65, 85], align: 'right' })
  y += 6

  result.feature_highlights.forEach((fh, i) => {
    doc.setFillColor(i % 2 === 0 ? 248 : 255, 250, 252)
    doc.rect(L, y - 1, W, 6, 'F')
    text(`${i + 1}.`, { size: 9, color: [100, 116, 139] })
    text(fmtFeatureName(fh.name), { size: 9, bold: i === 0, color: i === 0 ? [88, 28, 135] : [51, 65, 85], x: L + 16 })
    // Bar visualisation
    const barW = 60
    const barH = 2.5
    const barX = R - barW - 20
    doc.setFillColor(226, 232, 240)
    doc.rect(barX, y - 2, barW, barH, 'F')
    doc.setFillColor(88, 28, 135)
    doc.rect(barX, y - 2, barW * fh.importance, barH, 'F')
    text(`${(fh.importance * 100).toFixed(1)}%`, { size: 9, bold: i === 0, color: [88, 28, 135], align: 'right' })
    y += 6
  })
  gap(2)

  line('Top 5 Most-Attended Timesteps:', { bold: true, size: 9 })
  gap(1)
  const top5 = result.attention_weights
    .map((w, i) => ({ tSec: -(result.seq_len - 1 - i) * 10, weight: w }))
    .sort((a, b) => b.weight - a.weight).slice(0, 5)

  doc.setFillColor(241, 245, 249)
  doc.rect(L, y - 1, W / 2, 6, 'F')
  text('Timestep',  { size: 9, bold: true, color: [51, 65, 85] })
  text('Attention Weight', { size: 9, bold: true, color: [88, 28, 135], x: L + 35 })
  y += 6

  top5.forEach((a, i) => {
    const label = a.tSec === 0 ? 'Now (t = 0)' : `${Math.abs(a.tSec)}s ago`
    text(`${i + 1}. ${label}`, { size: 9, color: [51, 65, 85] })
    text(`${(a.weight * 100).toFixed(2)}%  ${i === 0 ? '← most attended' : ''}`, { size: 9, color: [88, 28, 135], x: L + 35 })
    y += 5
  })
  gap(3); rule()

  // ── Section 6: MODEL INFORMATION ─────────────────────────────────────────
  if (y > 240) { doc.addPage(); y = 18 }
  line('5. MODEL INFORMATION', { bold: true, size: 12 })
  gap(2)
  const modelRows: Array<[string, string]> = [
    ['Model Type',          'Bidirectional LSTM + Self-Attention'],
    ['Version',             result.model_version],
    ['Architecture',        'BiLSTM(64→32) + Dense(32) + Sigmoid'],
    ['Forecast Horizon',    `${result.horizon_minutes} min ahead`],
    ['Uncertainty Method',  'Monte Carlo Dropout (20 stochastic passes)'],
    ['Decision Threshold',  `${thr.toFixed(4)} (F1-optimised on validation set)`],
    ['ICP Formula',         'probToICP(p, thr) = 15 + 3×(logit(p)−logit(thr)) mmHg'],
    ['MAP→ICP Formula',     'ICP = MAP − CPP; CPP assumed = 70 mmHg (Rosner 1990)'],
    ['Training Data',       'CHARIS + MIMIC-III (PhysioNet), patient-stratified'],
    ['Model Performance',   'AUC = 0.984  |  F1 = 0.966  |  Sensitivity = 96.4%'],
    ['Train / Val / Test',  '279,660 / 9,546 / 156,690 sequences'],
  ]
  modelRows.forEach(([k, v]) => twoCol(k + ':', v, { size: 9, bold: true, color: [74, 85, 104] }, { size: 9, color: [26, 32, 44] }))
  gap(3); rule()

  // ── Disclaimer ────────────────────────────────────────────────────────────
  doc.setFillColor(254, 243, 199)
  doc.setDrawColor(217, 119, 6)
  const discStart = y; y += 4
  line('IMPORTANT DISCLAIMER', { bold: true, size: 10, color: [120, 53, 15] })
  const disc = [
    '• This system is a RESEARCH PROTOTYPE and a clinical decision SUPPORT tool only.',
    '• NOT FDA-cleared. NOT CE-marked. NOT for autonomous diagnostic or treatment decisions.',
    '• Estimated ICP values are probabilistic approximations derived from the LSTM model output',
    '  via logit-space calibration and the MAP−CPP formula — not direct ICP measurements.',
    '• All clinical decisions must be made and verified by qualified medical professionals.',
    '• Validate rigorously in a prospective clinical trial before any patient-care deployment.',
    '• Literature: Czosnyka & Pickard (Brain 2004), Rosner & Daughton (Neurosurg 1990).',
  ]
  disc.forEach(d => line(d, { size: 8, color: [120, 53, 15] }))
  y += 2
  doc.roundedRect(L, discStart, W, y - discStart, 2, 2, 'FD')

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
