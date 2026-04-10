import type { FeatureContribution } from '../types'
import { fmtFeatureName } from '../utils/formatters'

interface Props {
  features: FeatureContribution[]
  predictedClass: number
}

const CLASS_LABELS = ['Normal', 'Abnormal']
const STATUS_COLORS: Record<string, string> = {
  HIGH:   '#DC2626',
  NORMAL: '#059669',
  LOW:    '#2563EB',
}

export default function FeatureExplainer({ features, predictedClass }: Props) {
  if (features.length === 0) return null

  const barColor = predictedClass === 1 ? '#DC2626' : '#059669'

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide">
        Key Contributing Factors
      </h3>
      <p className="text-xs text-clinical-text-muted dark:text-slate-400">
        SHAP attribution for predicted class:{' '}
        <span className="font-medium text-clinical-text-primary dark:text-slate-200">
          {CLASS_LABELS[predictedClass]}
        </span>
      </p>

      <div className="space-y-3">
        {features.map((f, i) => {
          const positive = f.shap > 0
          const color    = positive ? barColor : '#6B7280'

          return (
            <div key={i} className="space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-clinical-text-primary dark:text-slate-200">
                    {fmtFeatureName(f.name)}
                  </span>
                  <span
                    className="text-2xs font-semibold px-1.5 py-0.5 rounded"
                    style={{
                      color: STATUS_COLORS[f.status],
                      backgroundColor: STATUS_COLORS[f.status] + '18',
                    }}
                  >
                    {f.status}
                  </span>
                </div>
                <span className="text-xs tabular-nums font-mono text-clinical-text-secondary dark:text-slate-400">
                  {f.value.toFixed(f.unit === 'Hz' ? 2 : 1)}{f.unit}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <div className="flex-1 h-2.5 bg-gray-100 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${f.impact_pct}%`, backgroundColor: color }}
                    role="progressbar"
                    aria-valuenow={f.impact_pct}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={`${fmtFeatureName(f.name)} impact ${f.impact_pct}%`}
                  />
                </div>
                <span className="text-2xs tabular-nums text-clinical-text-muted dark:text-slate-400 w-9 text-right">
                  {f.impact_pct.toFixed(0)}%
                </span>
                <span className="text-2xs font-medium w-8 text-right" style={{ color }}>
                  {positive ? '+' : ''}{(f.shap * 100).toFixed(1)}
                </span>
              </div>
            </div>
          )
        })}
      </div>

      <p className="text-2xs text-clinical-text-muted dark:text-slate-500 border-t border-clinical-border dark:border-slate-700 pt-2">
        SHAP values indicate contribution to the log-odds of the predicted class.
        Positive = increases likelihood; negative = decreases.
      </p>
    </div>
  )
}
