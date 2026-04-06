import { useStore } from '../store/useStore'
import { fmtFeatureName } from '../utils/formatters'

interface Props {
  attentionWeights:  number[]    // (30,) — timestep importance, sums to 1
  featureNames:      string[]    // (6,) — feature column labels
  globalImportances: Record<string, number>  // from XGBoost model info (optional)
}

/**
 * 30 × 6 attention heatmap.
 * Row    = timestep (t-30 to t-1, most recent at bottom)
 * Column = feature
 * Cell   = attention_weight[t] × feature_importance[f]  (normalised 0–1)
 */
export default function AttentionHeatmap({
  attentionWeights,
  featureNames,
  globalImportances,
}: Props) {
  const { isDark } = useStore()

  const nTimesteps = attentionWeights.length   // 30
  const nFeatures  = featureNames.length       // 6

  // Per-feature weights: use globalImportances if available, else uniform
  const featImportance = featureNames.map(f => {
    const k = Object.keys(globalImportances).find(
      k => k === f || k.toLowerCase().includes(f.split('_')[0])
    )
    return k ? (globalImportances[k] ?? 1 / nFeatures) : 1 / nFeatures
  })
  const featTotal = featImportance.reduce((s, v) => s + v, 0) || 1

  // Build heatmap cells: value = attn[t] × feat_imp[f] (linear 0–1)
  const cells: number[][] = attentionWeights.map(attn =>
    featImportance.map(fi => attn * (fi / featTotal))
  )
  const maxCell = Math.max(...cells.flat()) || 1

  function cellColor(val: number): string {
    const norm = val / maxCell
    if (isDark) {
      // Dark: deep navy → electric blue
      const r = Math.round(29  + (59  - 29)  * norm)
      const g = Math.round(78  + (130 - 78)  * norm)
      const b = Math.round(137 + (246 - 137) * norm)
      return `rgb(${r},${g},${b})`
    } else {
      // Light: white → clinical blue
      const r = Math.round(255 - (255 - 44)  * norm)
      const g = Math.round(255 - (255 - 82)  * norm)
      const b = Math.round(255 - (255 - 130) * norm)
      return `rgb(${r},${g},${b})`
    }
  }

  function cellTextColor(val: number): string {
    const norm = val / maxCell
    return norm > 0.55
      ? (isDark ? '#F0F4FF' : '#fff')
      : (isDark ? '#94A3B8' : '#4A5568')
  }

  // Show every 5th row label + first + last
  function showRowLabel(t: number): boolean {
    return t === 0 || t === nTimesteps - 1 || (t + 1) % 5 === 0
  }

  const cellH = 9   // px per row — keep compact
  const cellW = `${100 / nFeatures}%`

  return (
    <div>
      {/* Column headers */}
      <div className="flex mb-1 pl-8">
        {featureNames.map(f => (
          <div
            key={f}
            style={{ width: cellW }}
            className="text-center text-2xs text-clinical-text-muted dark:text-slate-500 truncate px-0.5"
            title={fmtFeatureName(f)}
          >
            {fmtFeatureName(f).split(' ').map(w => w[0]).join('')}
          </div>
        ))}
      </div>

      {/* Heatmap grid */}
      <div className="flex">
        {/* Row labels (timestep) */}
        <div className="flex flex-col" style={{ width: 32, minWidth: 32 }}>
          {cells.map((_, t) => (
            <div
              key={t}
              style={{ height: cellH, fontSize: 7, lineHeight: `${cellH}px` }}
              className="text-right pr-1 text-clinical-text-muted dark:text-slate-600 tabular-nums"
            >
              {showRowLabel(t) ? `t-${nTimesteps - t}` : ''}
            </div>
          ))}
        </div>

        {/* Grid body */}
        <div className="flex-1">
          {cells.map((row, t) => (
            <div key={t} className="flex">
              {row.map((val, f) => (
                <div
                  key={f}
                  style={{
                    width:           cellW,
                    height:          cellH,
                    backgroundColor: cellColor(val),
                    color:           cellTextColor(val),
                    fontSize:        6,
                    lineHeight:      `${cellH}px`,
                    textAlign:       'center',
                    borderRight:     f < nFeatures - 1
                      ? `1px solid ${isDark ? '#1A202C' : '#fff'}`
                      : 'none',
                    borderBottom:    t < nTimesteps - 1
                      ? `1px solid ${isDark ? '#1A202C' : '#fff'}`
                      : 'none',
                  }}
                  title={`t-${nTimesteps - t}, ${fmtFeatureName(featureNames[f])}: ${(val / maxCell * 100).toFixed(0)}%`}
                />
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Colour scale legend */}
      <div className="flex items-center gap-2 mt-2 pl-8">
        <span className="text-2xs text-clinical-text-muted dark:text-slate-500">Low</span>
        <div
          className="flex-1 h-2 rounded"
          style={{
            background: isDark
              ? 'linear-gradient(to right, rgb(29,78,137), rgb(59,130,246))'
              : 'linear-gradient(to right, #fff, rgb(44,82,130))',
            border: `1px solid ${isDark ? '#4A5568' : '#E2E8F0'}`,
          }}
        />
        <span className="text-2xs text-clinical-text-muted dark:text-slate-500">High</span>
      </div>

      <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-1.5">
        Cell intensity = timestep attention × feature importance. Most recent windows at bottom.
        Hover a cell for details.
      </p>
    </div>
  )
}
