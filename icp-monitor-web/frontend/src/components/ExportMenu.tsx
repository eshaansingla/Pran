import { useState, useRef, useEffect } from 'react'
import { FileDown, ChevronDown, FileText, Table } from 'lucide-react'
import type { BatchResult, ModelInfo } from '../types'
import { sessionId, downloadBlob } from '../utils/formatters'
import { fetchModelInfo } from '../utils/api'

interface Props {
  result: BatchResult
  disabled?: boolean
}

function exportCSV(result: BatchResult) {
  const header = 'window_id,class,class_name,probability,confidence,normal_pct,abnormal_pct'
  const rows = result.predictions.map(p =>
    [
      p.window_id,
      p.class,
      p.class_name,
      p.probability.toFixed(4),
      p.confidence.toFixed(4),
      (p.probabilities[0] * 100).toFixed(1),
      (p.probabilities[1] * 100).toFixed(1),
    ].join(',')
  )
  const csv = [header, ...rows].join('\n')
  downloadBlob(new Blob([csv], { type: 'text/csv' }), `ICP_Predictions_${sessionId()}.csv`)
}

async function exportPDF(result: BatchResult) {
  const [{ jsPDF }, modelInfo] = await Promise.all([
    import('jspdf'),
    fetchModelInfo().catch(() => null as ModelInfo | null),
  ])
  const doc = new jsPDF({ unit: 'mm', format: 'a4' })
  const mi = modelInfo
  const sid = sessionId()
  const now = new Date().toLocaleString('en-GB', { hour12: false })
  const { summary } = result

  let y = 18
  const L = 14
  const W = 182
  const R = L + W

  // ── helpers ──
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

  // ── Header bar ──
  doc.setFillColor(44, 82, 130)
  doc.rect(0, 0, 210, 18, 'F')
  doc.setFont('helvetica', 'bold'); doc.setFontSize(13); doc.setTextColor(255, 255, 255)
  doc.text('NON-INVASIVE ICP CLASSIFICATION REPORT', 14, 9)
  doc.setFont('helvetica', 'normal'); doc.setFontSize(8)
  doc.text('XGBoost Clinical Decision Support  |  Research Prototype  |  NOT FOR DIAGNOSTIC USE', 14, 14)
  doc.text(`v${mi?.version ?? '2.2'}`, 196, 9, { align: 'right' })
  y = 24

  // Report metadata
  twoCol('Report Date:', now, { size: 9, bold: true, color: [74, 85, 104] }, { size: 9, color: [74, 85, 104] })
  twoCol('Session ID:', sid, { size: 9, bold: true, color: [74, 85, 104] }, { size: 9, color: [74, 85, 104] })
  gap(1); rule()

  // ── Section 1: Classification Summary ──
  line('1. CLASSIFICATION SUMMARY', { bold: true, size: 12 })
  gap(2)

  const pctAbn = summary.abnormal_pct
  const isHighRisk = pctAbn >= 30
  const summaryColor: [number,number,number] = isHighRisk ? [254, 226, 226] : [209, 250, 229]
  const summaryBorder: [number,number,number] = isHighRisk ? [220, 38, 38] : [5, 150, 105]
  doc.setFillColor(...summaryColor); doc.setDrawColor(...summaryBorder)
  const boxStart = y; y += 5
  line(`Overall Assessment:  ${isHighRisk ? 'ELEVATED RISK — Abnormal ICP episodes detected' : 'LOW RISK — ICP predominantly within normal range'}`,
    { bold: true, size: 11, color: isHighRisk ? [220, 38, 38] : [5, 150, 105] })
  twoCol('Windows Analysed:', summary.total.toLocaleString(), { size: 10 }, { size: 10 })
  twoCol('Normal ICP:', `${summary.normal.toLocaleString()} (${summary.normal_pct}%)`, { size: 10, color: [5, 150, 105] }, { size: 10, color: [5, 150, 105] })
  twoCol('Abnormal ICP:', `${summary.abnormal.toLocaleString()} (${summary.abnormal_pct}%)`,
    { size: 10, color: isHighRisk ? [220, 38, 38] : [26, 32, 44], bold: isHighRisk },
    { size: 10, color: isHighRisk ? [220, 38, 38] : [26, 32, 44], bold: isHighRisk })
  y += 2
  doc.roundedRect(L, boxStart, W, y - boxStart, 2, 2, 'FD')
  gap(3); rule()

  // ── Section 2: Abnormal Events ──
  const abnormals = result.predictions.filter(p => p.class === 1)
  if (abnormals.length > 0) {
    line('2. ABNORMAL ICP EPISODES', { bold: true, size: 12, color: [220, 38, 38] })
    gap(2)
    const episodes: Array<{ start: number; end: number }> = []
    let start: number | null = null
    result.predictions.forEach((p, i) => {
      if (p.class === 1 && start === null) start = i + 1
      else if (p.class !== 1 && start !== null) { episodes.push({ start, end: i }); start = null }
    })
    if (start !== null) episodes.push({ start, end: result.predictions.length })

    // Table header
    doc.setFillColor(254, 242, 242)
    doc.rect(L, y - 1, W, 6, 'F')
    text('Episode', { size: 9, bold: true, color: [153, 27, 27] })
    text('Duration', { size: 9, bold: true, color: [153, 27, 27], x: L + 50 })
    text('Avg Confidence', { size: 9, bold: true, color: [153, 27, 27], x: L + 95 })
    text('Severity', { size: 9, bold: true, color: [153, 27, 27], align: 'right' })
    y += 6

    episodes.slice(0, 12).forEach((ep, i) => {
      const dur = ep.end - ep.start + 1
      const avgConf = result.predictions.slice(ep.start - 1, ep.end)
        .reduce((s, p) => s + p.confidence, 0) / dur
      const durMin = (dur * 10 / 60).toFixed(1)
      const severity = avgConf > 0.85 ? 'CRITICAL' : avgConf > 0.7 ? 'HIGH' : 'MODERATE'
      const sevColor: [number,number,number] = avgConf > 0.85 ? [153, 27, 27] : avgConf > 0.7 ? [220, 38, 38] : [180, 83, 9]
      doc.setFillColor(i % 2 === 0 ? 255 : 254, 250, 250)
      doc.rect(L, y - 1, W, 6, 'F')
      text(`Windows ${ep.start}–${ep.end}`, { size: 9, color: [51, 65, 85] })
      text(`${dur} windows (${durMin} min)`, { size: 9, color: [51, 65, 85], x: L + 50 })
      text(`${(avgConf * 100).toFixed(0)}%`, { size: 9, bold: true, color: sevColor, x: L + 95 })
      text(severity, { size: 9, bold: true, color: sevColor, align: 'right' })
      y += 6
    })
    if (episodes.length > 12) line(`   … and ${episodes.length - 12} more episodes`, { size: 9, color: [100, 116, 139] })
    gap(3); rule()
  }

  // ── Section 3: Prediction Log ──
  if (y > 220) { doc.addPage(); y = 18 }
  line(`${abnormals.length > 0 ? '3' : '2'}. WINDOW-BY-WINDOW CLASSIFICATION (first 30)`, { bold: true, size: 12 })
  gap(2)
  const colX = [L, L + 30, L + 65, L + 105, L + 140]
  const headers = ['Window', 'Class', 'P(Normal)', 'P(Abnormal)', 'Status']
  doc.setFillColor(241, 245, 249)
  doc.rect(L, y - 1, W, 6, 'F')
  headers.forEach((h, i) => {
    doc.setFont('helvetica', 'bold'); doc.setFontSize(8)
    doc.setTextColor(51, 65, 85)
    doc.text(h, colX[i], y)
  })
  y += 6
  doc.setFont('helvetica', 'normal')
  result.predictions.slice(0, 30).forEach(p => {
    const isAbn = p.class === 1
    const rowBg: [number,number,number] = isAbn ? [254, 242, 242] : [248, 250, 252]
    doc.setFillColor(...rowBg)
    doc.rect(L, y - 4, W, 5, 'F')
    doc.setTextColor(26, 32, 44); doc.text(String(p.window_id), colX[0], y)
    doc.setTextColor(...(isAbn ? [220, 38, 38] as [number,number,number] : [5, 150, 105] as [number,number,number]))
    doc.setFont('helvetica', isAbn ? 'bold' : 'normal')
    doc.text(p.class_name, colX[1], y)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(26, 32, 44)
    doc.text((p.probabilities[0] * 100).toFixed(1) + '%', colX[2], y)
    doc.setTextColor(...(isAbn ? [220, 38, 38] as [number,number,number] : [26, 32, 44] as [number,number,number]))
    doc.setFont('helvetica', isAbn ? 'bold' : 'normal')
    doc.text((p.probabilities[1] * 100).toFixed(1) + '%', colX[3], y)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(...(isAbn ? [220, 38, 38] as [number,number,number] : [5, 150, 105] as [number,number,number]))
    doc.text(isAbn ? 'ABNORMAL' : 'Normal', colX[4], y)
    y += 5
    if (y > 270) { doc.addPage(); y = 18 }
  })
  gap(3); rule()

  // ── Section 4: Model Information ──
  if (y > 230) { doc.addPage(); y = 18 }
  const secNum = abnormals.length > 0 ? '4' : '3'
  line(`${secNum}. MODEL INFORMATION`, { bold: true, size: 12 })
  gap(2)
  const mRows: Array<[string, string]> = [
    ['Model Type',        mi?.model_type ?? 'XGBoost Binary Classifier'],
    ['Version',           `v${mi?.version ?? '2.2'}`],
    ['Training Data',     `CHARIS (${mi?.training_data.charis_patients ?? 13}) + MIMIC-III (${mi?.training_data.mimic_patients ?? 87}) patients`],
    ['Calibration',       mi?.calibrated ? 'Isotonic Regression (cross-validated)' : 'Uncalibrated'],
    ['ICP Threshold',     '15 mmHg (Czosnyka & Pickard, Brain 2004)'],
    ['F1-Score (test)',   (mi?.metrics.f1 ?? 0.877).toFixed(4)],
    ['AUC (test)',        (mi?.metrics.auc ?? 0.949).toFixed(4)],
    ['Sensitivity (test)',(mi?.metrics.recall ?? 0.819).toFixed(4)],
    ['Specificity (test)',(mi?.metrics.specificity ?? 0.951).toFixed(4)],
    ['Data Integrity',    'No data leakage — patient-level GroupShuffleSplit'],
  ]
  mRows.forEach(([k, v]) => twoCol(k + ':', v, { size: 9, bold: true, color: [74, 85, 104] }, { size: 9, color: [26, 32, 44] }))
  gap(3); rule()

  // ── Disclaimer ──
  doc.setFillColor(254, 243, 199); doc.setDrawColor(217, 119, 6)
  const discStart = y; y += 4
  line('IMPORTANT DISCLAIMER', { bold: true, size: 10, color: [120, 53, 15] })
  const disc = [
    '• This system is a RESEARCH PROTOTYPE and a clinical decision SUPPORT tool only.',
    '• NOT FDA-cleared. NOT CE-marked. NOT for autonomous diagnostic or treatment decisions.',
    '• All clinical decisions must be made and verified by qualified medical professionals.',
    '• Model trained on CHARIS + MIMIC-III research datasets — requires prospective validation.',
    '• Literature: Czosnyka & Pickard (Brain 2004), Rosner & Daughton (Neurosurg 1990).',
  ]
  disc.forEach(d => line(d, { size: 8, color: [120, 53, 15] }))
  y += 2
  doc.roundedRect(L, discStart, W, y - discStart, 2, 2, 'FD')

  doc.save(`ICP_Classification_Report_${sid}.pdf`)
}

export default function ExportMenu({ result, disabled }: Props) {
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
        disabled={disabled}
        aria-label="Export data"
        aria-haspopup="true"
        aria-expanded={open}
        className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-clinical-primary dark:bg-blue-600 text-white rounded-lg
          hover:bg-blue-700 dark:hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-clinical-primary focus:ring-offset-2
          disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        <FileDown size={15} />
        Export
        <ChevronDown size={13} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute right-0 mt-1 w-44 bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg shadow-lg z-20 overflow-hidden">
          <button
            onClick={() => { exportCSV(result); setOpen(false) }}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-clinical-text-primary dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-left"
          >
            <Table size={14} className="text-clinical-text-muted dark:text-slate-400" />
            Export CSV
          </button>
          <button
            onClick={() => { exportPDF(result); setOpen(false) }}
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
