import { useState } from 'react'
import { Activity, AlertTriangle, CheckCircle2 } from 'lucide-react'
import type { ForecastResult } from '../types'
import { useStore } from '../store/useStore'
import { mapToICP, icpGrade } from '../utils/formatters'

// Scaler constants from lstm_meta.json — used to reconstruct raw MAP from z-scores
const MAP_MEAN = 88.352
const MAP_STD  = 7.850

/** Auto-detect z-score vs raw MAP. Physiological MAP ≥ 50 mmHg always. */
function toRawMAP(val: number): number {
  if (val > 50) return val
  return val * MAP_STD + MAP_MEAN
}

const FEATURE_DEFS = [
  { idx: 0, label: 'Cardiac Amplitude',      unit: 'μm'   },
  { idx: 1, label: 'Cardiac Frequency',       unit: 'Hz'   },
  { idx: 2, label: 'Respiratory Amplitude',   unit: 'μm'   },
  { idx: 3, label: 'Slow Wave Power',         unit: '—'    },
  { idx: 4, label: 'Cardiac Power',           unit: '—'    },
  { idx: 5, label: 'Mean Arterial Pressure',  unit: 'mmHg' },
]

interface Props {
  sequence: number[][]   // raw N × 6 windows (all-zero when reloaded from history)
  result: ForecastResult
}

export default function ForecastWindowAnalysis({ sequence, result }: Props) {
  const { isDark }    = useStore()
  const [open, setOpen]         = useState(true)   // start open — visible by default
  const [selected, setSelected] = useState<number | null>(null)

  const seqLen      = sequence.length
  const hasRealData = sequence.some(row => row.some(v => v !== 0))

  // Per-window derived values
  // Classification uses clinical ICP threshold (15 mmHg), NOT LSTM probability threshold.
  // MAP-based ICP is a physiological proxy independent of the LSTM classifier.
  const windows = Array.from({ length: seqLen }, (_, i) => {
    const row    = sequence[i]
    const rawMap = toRawMAP(row[5])      // auto-reconstruct if z-scored
    const icp    = mapToICP(rawMap)      // MAP − 70, clamped [5, 40]
    const isAbn  = icp >= 15            // clinical ICP threshold
    const attnW  = result.attention_weights[i] ?? 0
    const tMin   = -(seqLen - 1 - i) * 10 / 60
    return { idx: i, tMin, icp, isAbn, attnW, features: row, rawMap }
  })

  const attnMax = Math.max(...windows.map(w => w.attnW), 0.001)
  const sel     = selected !== null ? windows[selected] : null

  const abnCount = windows.filter(w => w.isAbn).length

  return (
    <div className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-xl shadow-sm overflow-hidden">

      {/* Collapsible header */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3
          text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <Activity size={13} className="text-purple-500 dark:text-purple-400 flex-shrink-0" />
          <span className="text-xs font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide">
            Input Window Analysis
          </span>
          <span className="text-2xs bg-slate-100 dark:bg-slate-700 text-clinical-text-muted dark:text-slate-400 px-1.5 py-0.5 rounded font-mono">
            {seqLen}
          </span>
          {abnCount > 0 && hasRealData && (
            <span className="text-2xs bg-red-100 dark:bg-red-900/40 text-red-600 dark:text-red-400 px-1.5 py-0.5 rounded font-semibold">
              {abnCount} abnormal
            </span>
          )}
        </div>
        <span className="text-2xs text-clinical-text-muted dark:text-slate-500">
          {open ? 'hide' : 'show'}
        </span>
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-4">

          {!hasRealData && (
            <div className="rounded-lg border border-amber-200 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20 px-3 py-2">
              <p className="text-xs text-amber-700 dark:text-amber-400">
                Raw sequence data not available for reloaded forecasts.
                Upload the original CSV to see per-window analysis.
              </p>
            </div>
          )}

          {/* Timeline bar */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <p className="text-2xs font-semibold text-clinical-text-muted dark:text-slate-400 uppercase tracking-wide">
                Session Timeline — {seqLen} windows · {(seqLen * 10 / 60).toFixed(1)} min
              </p>
              <p className="text-2xs text-clinical-text-muted dark:text-slate-500">
                Colour: MAP-based ICP ≥15 mmHg = abnormal
              </p>
            </div>
            <div className="flex h-8 rounded-lg overflow-hidden border border-clinical-border dark:border-slate-600 gap-px">
              {windows.map(w => {
                const normalColor = isDark ? '#10B981' : '#059669'
                const abnColor    = isDark ? '#EF4444' : '#DC2626'
                const color       = w.isAbn ? abnColor : normalColor
                const opacity     = hasRealData
                  ? 0.45 + 0.55 * (w.attnW / attnMax)
                  : 0.5
                const isSelected  = selected === w.idx
                return (
                  <div
                    key={w.idx}
                    style={{ flex: 1, backgroundColor: color, opacity }}
                    className={`cursor-pointer hover:opacity-100 transition-opacity relative group ${
                      isSelected ? 'ring-2 ring-white ring-inset' : ''
                    }`}
                    onClick={() => setSelected(selected === w.idx ? null : w.idx)}
                  >
                    {/* Hover tooltip */}
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block z-20 pointer-events-none">
                      <div className="bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 text-2xs px-2 py-1 rounded shadow-lg whitespace-nowrap">
                        #{w.idx + 1} · t={w.tMin.toFixed(1)}m
                        {hasRealData && ` · ICP~${w.icp.toFixed(0)} mmHg · Attn ${(w.attnW * 100).toFixed(1)}%`}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
            <div className="flex items-center justify-between text-2xs text-clinical-text-muted dark:text-slate-500">
              <span>t = −{(seqLen * 10 / 60).toFixed(0)} min</span>
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm inline-block bg-emerald-500" />Normal
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm inline-block bg-red-500" />Abnormal
                </span>
                {hasRealData && <span>· brightness ∝ attention</span>}
              </div>
              <span>t = Now</span>
            </div>
          </div>

          {/* Window selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-clinical-text-muted dark:text-slate-400 flex-shrink-0">
              Inspect window:
            </label>
            <select
              value={selected ?? ''}
              onChange={e => setSelected(e.target.value === '' ? null : Number(e.target.value))}
              className="flex-1 text-xs border border-clinical-border dark:border-slate-600 rounded-lg px-2 py-1.5
                bg-white dark:bg-slate-700 text-clinical-text-primary dark:text-slate-200
                focus:outline-none focus:ring-2 focus:ring-purple-400 dark:focus:ring-purple-500"
            >
              <option value="">— select a window —</option>
              {windows.map(w => (
                <option key={w.idx} value={w.idx}>
                  #{w.idx + 1} · t = {w.tMin.toFixed(1)} min
                  {hasRealData
                    ? ` · ICP ~${w.icp.toFixed(0)} mmHg · ${w.isAbn ? 'Abnormal' : 'Normal'} · Attn ${(w.attnW * 100).toFixed(1)}%`
                    : ''}
                </option>
              ))}
            </select>
            {selected !== null && (
              <button
                onClick={() => setSelected(null)}
                className="text-2xs text-clinical-text-muted dark:text-slate-500 hover:text-red-600 dark:hover:text-red-400 px-1"
              >
                clear
              </button>
            )}
          </div>

          {/* Detail panel */}
          {sel && (
            <div className="grid grid-cols-2 gap-3">

              {/* Left: status cards */}
              <div className="space-y-2">

                {/* Classification */}
                <div className={`rounded-xl border px-3 py-2.5 ${
                  sel.isAbn
                    ? 'bg-red-50 dark:bg-red-900/25 border-red-200 dark:border-red-700'
                    : 'bg-emerald-50 dark:bg-emerald-900/25 border-emerald-200 dark:border-emerald-700'
                }`}>
                  <div className="flex items-center gap-2 mb-0.5">
                    {sel.isAbn
                      ? <AlertTriangle size={12} className="text-red-500 flex-shrink-0" />
                      : <CheckCircle2  size={12} className="text-emerald-500 flex-shrink-0" />
                    }
                    <p className={`text-sm font-bold ${sel.isAbn ? 'text-red-600 dark:text-red-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                      {sel.isAbn ? 'Abnormal' : 'Normal'}
                    </p>
                  </div>
                  <p className="text-2xs text-clinical-text-muted dark:text-slate-400">
                    Window #{sel.idx + 1} · t = {sel.tMin.toFixed(1)} min
                  </p>
                  <p className="text-2xs text-clinical-text-muted dark:text-slate-500">
                    {Math.abs(Math.round(sel.tMin * 60))}s before present
                  </p>
                </div>

                {/* ICP estimate */}
                {hasRealData && (
                  <div className="rounded-xl border border-amber-200 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20 px-3 py-2.5">
                    <p className="text-2xs text-clinical-text-muted dark:text-slate-400 mb-0.5">Est. ICP (MAP − 70)</p>
                    <p className="text-xl font-bold tabular-nums text-amber-700 dark:text-amber-400">
                      ~{sel.icp.toFixed(0)} mmHg
                    </p>
                    <p className="text-xs font-semibold" style={{ color: icpGrade(sel.icp).color }}>
                      {icpGrade(sel.icp).label}
                    </p>
                  </div>
                )}

                {/* Attention weight */}
                <div className="rounded-xl border border-purple-200 dark:border-purple-700 bg-purple-50 dark:bg-purple-900/20 px-3 py-2.5">
                  <p className="text-2xs text-clinical-text-muted dark:text-slate-400 mb-1.5">
                    LSTM Attention Weight
                  </p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-purple-500 dark:bg-purple-400"
                        style={{ width: `${Math.min(100, (sel.attnW / attnMax) * 100)}%` }}
                      />
                    </div>
                    <span className="text-xs font-mono tabular-nums text-purple-700 dark:text-purple-300 w-10 text-right">
                      {(sel.attnW * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-1">
                    {sel.attnW >= attnMax * 0.8
                      ? 'High importance to model decision'
                      : sel.attnW >= attnMax * 0.4
                      ? 'Moderate importance'
                      : 'Lower importance timestep'}
                  </p>
                </div>
              </div>

              {/* Right: feature table */}
              <div className="bg-slate-50 dark:bg-slate-900/40 rounded-xl border border-clinical-border dark:border-slate-700 overflow-hidden">
                <div className="px-3 py-2 border-b border-clinical-border dark:border-slate-700 bg-white dark:bg-slate-800/80">
                  <p className="text-2xs font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide">
                    Feature Values
                  </p>
                </div>
                <div className="divide-y divide-clinical-border dark:divide-slate-700/70">
                  {FEATURE_DEFS.map(f => {
                    const raw   = sel.features[f.idx]
                    const isMap = f.idx === 5
                    // For MAP: always show reconstructed raw mmHg, not z-score
                    const displayVal = isMap ? sel.rawMap : raw
                    const hasVal = raw !== undefined && (isMap ? true : raw !== 0)
                    return (
                      <div key={f.idx} className={`flex items-center justify-between px-3 py-1.5 ${
                        isMap ? 'bg-amber-50/50 dark:bg-amber-900/10' : ''
                      }`}>
                        <span className={`text-2xs truncate mr-2 ${
                          isMap
                            ? 'text-amber-700 dark:text-amber-400 font-medium'
                            : 'text-clinical-text-muted dark:text-slate-400'
                        }`}>
                          {f.label}
                        </span>
                        <span className="text-2xs font-mono tabular-nums text-clinical-text-primary dark:text-slate-200 flex-shrink-0">
                          {hasVal
                            ? `${displayVal.toFixed(f.idx === 1 ? 3 : 1)}${f.unit !== '—' ? ' ' + f.unit : ''}`
                            : <span className="text-clinical-text-muted dark:text-slate-600">—</span>
                          }
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Attention summary table (top 5) */}
          {hasRealData && (
            <div>
              <p className="text-2xs font-semibold text-clinical-text-muted dark:text-slate-400 uppercase tracking-wide mb-1.5">
                Top 5 Attended Timesteps
              </p>
              <div className="rounded-lg border border-clinical-border dark:border-slate-700 overflow-hidden">
                <table className="w-full text-2xs">
                  <thead className="bg-slate-50 dark:bg-slate-900/40">
                    <tr>
                      <th className="px-3 py-1.5 text-left font-semibold text-clinical-text-muted dark:text-slate-400">#</th>
                      <th className="px-3 py-1.5 text-left font-semibold text-clinical-text-muted dark:text-slate-400">Time</th>
                      <th className="px-3 py-1.5 text-right font-semibold text-amber-600 dark:text-amber-400">ICP est.</th>
                      <th className="px-3 py-1.5 text-right font-semibold text-purple-600 dark:text-purple-400">Attention</th>
                      <th className="px-3 py-1.5 text-left font-semibold text-clinical-text-muted dark:text-slate-400">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-clinical-border dark:divide-slate-700">
                    {[...windows]
                      .sort((a, b) => b.attnW - a.attnW)
                      .slice(0, 5)
                      .map((w, rank) => (
                        <tr
                          key={w.idx}
                          className={`cursor-pointer transition-colors ${
                            selected === w.idx
                              ? 'bg-purple-50 dark:bg-purple-900/20'
                              : 'hover:bg-slate-50 dark:hover:bg-slate-700/30'
                          }`}
                          onClick={() => setSelected(selected === w.idx ? null : w.idx)}
                        >
                          <td className="px-3 py-1.5 font-mono text-clinical-text-muted dark:text-slate-500">
                            {rank + 1}
                          </td>
                          <td className="px-3 py-1.5 text-clinical-text-secondary dark:text-slate-300">
                            #{w.idx + 1} · {w.tMin.toFixed(1)}m
                          </td>
                          <td className="px-3 py-1.5 text-right tabular-nums font-mono text-amber-600 dark:text-amber-400">
                            ~{w.icp.toFixed(0)} mmHg
                          </td>
                          <td className="px-3 py-1.5 text-right tabular-nums font-mono text-purple-600 dark:text-purple-400">
                            {(w.attnW * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-1.5">
                            <span className={`px-1.5 py-0.5 rounded text-2xs font-semibold ${
                              w.isAbn
                                ? 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-400'
                                : 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-400'
                            }`}>
                              {w.isAbn ? 'Abn' : 'Norm'}
                            </span>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
