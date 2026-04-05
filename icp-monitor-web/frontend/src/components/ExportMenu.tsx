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

  // Header bar
  doc.setFillColor(44, 82, 130)
  doc.rect(0, 0, 210, 14, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.setTextColor(255, 255, 255)
  doc.text('ICP MONITORING REPORT — CLINICAL DECISION SUPPORT', 14, 9)
  doc.setFontSize(7)
  doc.setFont('helvetica', 'normal')
  doc.text(`NOT FOR DIAGNOSTIC USE | Research Prototype v${mi?.version ?? '2.1'}`, 14, 12.5)
  y = 22

  line('GENERATED: ' + now, { bold: true, size: 9 })
  line('SESSION ID: ' + sid, { size: 9, color: [74, 85, 104] })
  gap(3)
  rule()

  // Summary
  line('SUMMARY STATISTICS', { bold: true, size: 11 })
  gap(2)
  line(`Total Windows Analysed : ${summary.total.toLocaleString()}`, { size: 10 })
  line(`Normal ICP             : ${summary.normal.toLocaleString()} windows (${summary.normal_pct}%)`, { size: 10, color: [5, 150, 105] })
  line(`Abnormal ICP           : ${summary.abnormal.toLocaleString()} windows (${summary.abnormal_pct}%)`, { size: 10, color: [220, 38, 38] })
  gap(3)
  rule()

  // Abnormal events
  const abnormals = result.predictions.filter(p => p.class === 1)
  if (abnormals.length > 0) {
    line('ABNORMAL EVENTS', { bold: true, size: 11, color: [220, 38, 38] })
    gap(2)
    const episodes: Array<{ start: number; end: number }> = []
    let start: number | null = null
    result.predictions.forEach((p, i) => {
      if (p.class === 1 && start === null) start = i + 1
      else if (p.class !== 1 && start !== null) { episodes.push({ start, end: i }); start = null }
    })
    if (start !== null) episodes.push({ start, end: result.predictions.length })

    episodes.slice(0, 10).forEach((ep, i) => {
      const dur     = ep.end - ep.start + 1
      const avgConf = result.predictions.slice(ep.start - 1, ep.end)
        .reduce((s, p) => s + p.confidence, 0) / dur
      line(
        `${i + 1}. Windows ${ep.start}–${ep.end}  (${dur} window${dur > 1 ? 's' : ''}, avg confidence ${(avgConf * 100).toFixed(0)}%)`,
        { size: 9, color: [153, 27, 27] }
      )
    })
    if (episodes.length > 10) line(`   … and ${episodes.length - 10} more episodes`, { size: 9 })
    gap(3)
    rule()
  }

  // Prediction log
  line('PREDICTION LOG (first 30 windows)', { bold: true, size: 11 })
  gap(2)
  const colX = [L, 55, 105, 150]
  const headers = ['Window', 'Class', 'Normal%', 'Abnormal%']
  const colColors: Array<[number,number,number]> = [[26,32,44],[26,32,44],[5,150,105],[220,38,38]]
  doc.setFont('helvetica', 'bold'); doc.setFontSize(8)
  headers.forEach((h, i) => { doc.setTextColor(...colColors[i]); doc.text(h, colX[i], y) })
  y += 5
  doc.setFont('helvetica', 'normal')
  result.predictions.slice(0, 30).forEach(p => {
    doc.setTextColor(26,32,44); doc.text(String(p.window_id), colX[0], y)
    const clsColor: [number,number,number][] = [[5,150,105],[220,38,38]]
    doc.setTextColor(...clsColor[p.class]); doc.text(p.class_name, colX[1], y)
    doc.setTextColor(26,32,44)
    doc.text((p.probabilities[0]*100).toFixed(1)+'%', colX[2], y)
    doc.text((p.probabilities[1]*100).toFixed(1)+'%', colX[3], y)
    y += 5
    if (y > 270) { doc.addPage(); y = 18 }
  })
  gap(3)
  rule()

  // Model info
  line('MODEL INFORMATION', { bold: true, size: 11 })
  gap(2)
  line(`Model Type          : ${mi?.model_type ?? 'XGBoost Binary Classifier'}`, { size: 10 })
  line(`Version             : ${mi?.version ?? '2.1'}`, { size: 10 })
  line(`Training Date       : ${mi?.training_date ?? 'N/A'}`, { size: 10 })
  line(`Training Data       : CHARIS (${mi?.training_data.charis_patients ?? 13} patients) + MIMIC-III (${mi?.training_data.mimic_patients ?? '?'} patients)`, { size: 10 })
  line(`F1-Score            : ${(mi?.metrics.f1 ?? 0.8796).toFixed(4)}`, { size: 10 })
  line(`AUC                 : ${(mi?.metrics.auc ?? 0.9623).toFixed(4)}`, { size: 10 })
  line(`Balanced Accuracy   : ${(mi?.metrics.balanced_accuracy ?? 0.8831).toFixed(4)}`, { size: 10 })
  if (mi?.calibrated) line('Calibration         : Isotonic (probabilities calibrated)', { size: 10 })
  gap(3)
  rule()

  // Disclaimer
  line('IMPORTANT DISCLAIMER', { bold: true, size: 11, color: [220, 38, 38] })
  gap(2)
  const disc = [
    'This system is a clinical decision SUPPORT tool only.',
    'It is NOT FDA-approved and NOT intended for autonomous diagnostic use.',
    'All clinical decisions must be made by qualified medical professionals.',
    'Model trained on research datasets — validate before clinical deployment.',
  ]
  disc.forEach(d => line(d, { size: 9, color: [153, 27, 27] }))

  doc.save(`ICP_Report_${sid}.pdf`)
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
