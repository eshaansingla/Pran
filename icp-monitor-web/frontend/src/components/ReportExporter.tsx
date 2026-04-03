import { FileDown } from 'lucide-react'
import type { BatchResult } from '../types'
import { sessionId } from '../utils/formatters'

interface Props {
  result: BatchResult
  disabled?: boolean
}

export default function ReportExporter({ result, disabled }: Props) {
  const generate = async () => {
    const { jsPDF } = await import('jspdf')
    const doc = new jsPDF({ unit: 'mm', format: 'a4' })
    const sid = sessionId()
    const now = new Date().toLocaleString('en-GB', { hour12: false })
    const { summary } = result

    const LINE = '─'.repeat(60)
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

    // Header
    doc.setFillColor(44, 82, 130)
    doc.rect(0, 0, 210, 14, 'F')
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(11)
    doc.setTextColor(255, 255, 255)
    doc.text('ICP MONITORING REPORT — CLINICAL DECISION SUPPORT', 14, 9)
    doc.setFontSize(7)
    doc.setFont('helvetica', 'normal')
    doc.text('NOT FOR DIAGNOSTIC USE | Research Prototype v1.0', 14, 12.5)
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
    line(`Elevated ICP           : ${summary.elevated.toLocaleString()} windows (${summary.elevated_pct}%)`, { size: 10, color: [217, 119, 6] })
    line(`Critical ICP           : ${summary.critical.toLocaleString()} windows (${summary.critical_pct}%)`, { size: 10, color: [220, 38, 38] })
    gap(3)
    rule()

    // Critical events
    const criticals = result.predictions.filter(p => p.class === 2)
    if (criticals.length > 0) {
      line('CRITICAL EVENTS', { bold: true, size: 11, color: [220, 38, 38] })
      gap(2)

      // Find episodes
      const episodes: Array<{ start: number; end: number }> = []
      let start: number | null = null
      result.predictions.forEach((p, i) => {
        if (p.class === 2 && start === null) start = i + 1
        else if (p.class !== 2 && start !== null) {
          episodes.push({ start, end: i })
          start = null
        }
      })
      if (start !== null) episodes.push({ start, end: result.predictions.length })

      episodes.slice(0, 10).forEach((ep, i) => {
        const dur = ep.end - ep.start + 1
        const avgConf = result.predictions
          .slice(ep.start - 1, ep.end)
          .reduce((s, p) => s + p.confidence, 0) / dur
        line(
          `${i + 1}. Windows ${ep.start}–${ep.end}  (${dur} windows, avg confidence ${(avgConf * 100).toFixed(0)}%)`,
          { size: 9, color: [153, 27, 27] }
        )
      })
      if (episodes.length > 10) line(`   … and ${episodes.length - 10} more episodes`, { size: 9 })
      gap(3)
      rule()
    }

    // Predictions table (first 30)
    line('PREDICTION LOG (first 30 windows)', { bold: true, size: 11 })
    gap(2)
    const colX = [L, 50, 90, 130, 162]
    const headers = ['Window', 'Class', 'Normal%', 'Elevated%', 'Critical%']
    const colColors: Array<[number,number,number]> = [
      [26,32,44],[26,32,44],[5,150,105],[217,119,6],[220,38,38]
    ]
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(8)
    headers.forEach((h, i) => {
      doc.setTextColor(...colColors[i])
      doc.text(h, colX[i], y)
    })
    y += 5
    doc.setFont('helvetica', 'normal')
    result.predictions.slice(0, 30).forEach(p => {
      doc.setTextColor(26,32,44); doc.text(String(p.window_id), colX[0], y)
      const clsColor: [number,number,number][] = [[5,150,105],[217,119,6],[220,38,38]]
      doc.setTextColor(...clsColor[p.class]); doc.text(p.class_name, colX[1], y)
      doc.setTextColor(26,32,44)
      doc.text((p.probabilities[0]*100).toFixed(1)+'%', colX[2], y)
      doc.text((p.probabilities[1]*100).toFixed(1)+'%', colX[3], y)
      doc.text((p.probabilities[2]*100).toFixed(1)+'%', colX[4], y)
      y += 5
      if (y > 270) { doc.addPage(); y = 18 }
    })
    gap(3)
    rule()

    // Model info
    line('MODEL INFORMATION', { bold: true, size: 11 })
    gap(2)
    line('Model Type          : XGBoost (Gradient Boosting)', { size: 10 })
    line('Version             : 1.0', { size: 10 })
    line('Training Date       : 2026-04-03', { size: 10 })
    line('Training Data       : CHARIS (13 patients) + MIMIC-III (36 patients)', { size: 10 })
    line('Macro F1-Score      : 0.7667', { size: 10 })
    line('Balanced Accuracy   : 0.7686', { size: 10 })
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

  return (
    <button
      onClick={generate}
      disabled={disabled}
      aria-label="Export session report as PDF"
      className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-clinical-primary text-white rounded-lg
        hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-clinical-primary focus:ring-offset-2
        disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
    >
      <FileDown size={16} />
      Export PDF Report
    </button>
  )
}
