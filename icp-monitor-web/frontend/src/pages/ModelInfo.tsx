import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { fetchModelInfo } from '../utils/api'
import type { ModelInfo } from '../types'
import { fmtFeatureName } from '../utils/formatters'

export default function ModelInfoPage() {
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch(e => setError(String(e)))
  }, [])

  if (error) return <p className="text-sm text-clinical-critical p-4">{error}</p>
  if (!info) return <p className="text-sm text-clinical-text-muted p-4">Loading model information…</p>

  const importanceData = Object.entries(info.global_importances)
    .sort((a, b) => b[1] - a[1])
    .map(([name, val]) => ({ name: fmtFeatureName(name), value: +(val * 100).toFixed(1) }))

  const metrics: Array<{ label: string; value: number; fmt: (v: number) => string }> = [
    { label: 'F1-Score',          value: info.metrics.f1,                fmt: (v: number) => v.toFixed(4) },
    { label: 'AUC',               value: info.metrics.auc,               fmt: (v: number) => v.toFixed(4) },
    { label: 'Precision',         value: info.metrics.precision,         fmt: (v: number) => v.toFixed(4) },
    { label: 'Recall',            value: info.metrics.recall,            fmt: (v: number) => v.toFixed(4) },
    { label: 'Specificity',       value: info.metrics.specificity,       fmt: (v: number) => v.toFixed(4) },
    { label: 'Balanced Accuracy', value: info.metrics.balanced_accuracy, fmt: (v: number) => v.toFixed(4) },
  ]

  return (
    <div className="space-y-5 max-w-4xl">
      <div>
        <h1 className="text-base font-semibold text-clinical-text-primary">Model Information</h1>
        <p className="text-sm text-clinical-text-muted mt-0.5">
          XGBoost v{info.version} — trained {info.training_date}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-5">
        {/* Performance metrics */}
        <section
          aria-label="Model performance metrics"
          className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm"
        >
          <h2 className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide mb-4">
            Performance Metrics
          </h2>
          <div className="space-y-2">
            {metrics.map(m => (
              <div key={m.label} className="flex items-center justify-between py-1.5 border-b border-clinical-border last:border-0">
                <span className="text-sm text-clinical-text-secondary">{m.label}</span>
                <span className="text-sm font-semibold tabular-nums font-mono text-clinical-text-primary">
                  {m.fmt(m.value)}
                </span>
              </div>
            ))}
          </div>
          <p className="text-2xs text-clinical-text-muted mt-3">
            Evaluated on held-out test set: {info.training_data.charis_patients + info.training_data.mimic_patients} patients,
            patient-stratified 20% hold-out.
          </p>
        </section>

        {/* Training data */}
        <section
          aria-label="Training data statistics"
          className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm"
        >
          <h2 className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide mb-4">
            Training Data
          </h2>
          <div className="space-y-2">
            {[
              { label: 'Dataset',          value: 'CHARIS + MIMIC-III' },
              { label: 'Total Patients',   value: String(info.training_data.charis_patients + info.training_data.mimic_patients) },
              { label: 'CHARIS (TBI ICU)', value: String(info.training_data.charis_patients) + ' patients' },
              { label: 'MIMIC (Gen. ICU)', value: String(info.training_data.mimic_patients) + ' patients' },
              { label: 'Total Windows',    value: info.training_data.total_windows.toLocaleString() },
              { label: 'Window Duration',  value: '10 seconds' },
              { label: 'Sampling Rate',    value: '125 Hz' },
            ].map(({ label, value }) => (
              <div key={label} className="flex items-center justify-between py-1.5 border-b border-clinical-border last:border-0">
                <span className="text-sm text-clinical-text-secondary">{label}</span>
                <span className="text-sm font-medium text-clinical-text-primary">{value}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Feature importance */}
        <section
          aria-label="Global feature importance"
          className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm col-span-2"
        >
          <h2 className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide mb-4">
            Global Feature Importance (Gain)
          </h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={importanceData} layout="vertical" margin={{ left: 0, right: 16, top: 0, bottom: 0 }}>
              <XAxis type="number" tickFormatter={v => `${v}%`} tick={{ fontSize: 10, fill: '#718096' }} tickLine={false} axisLine={false} />
              <YAxis dataKey="name" type="category" width={150} tick={{ fontSize: 10, fill: '#4A5568' }} tickLine={false} axisLine={false} />
              <Tooltip formatter={(v: number) => [`${v}%`, 'Importance']} contentStyle={{ fontSize: 11 }} />
              <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                {importanceData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? '#2C5282' : i === 1 ? '#4A5568' : '#718096'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-2xs text-clinical-text-muted mt-2">
            Gain importance: average improvement in split criterion when a feature is used for splitting.
            Features with zero gain were not used by the model.
          </p>
        </section>

        {/* Hyperparameters */}
        <section
          aria-label="Model hyperparameters"
          className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm"
        >
          <h2 className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide mb-4">
            Hyperparameters
          </h2>
          <div className="space-y-2">
            {Object.entries(info.hyperparameters).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between py-1.5 border-b border-clinical-border last:border-0">
                <span className="text-sm text-clinical-text-secondary font-mono">{k}</span>
                <span className="text-sm tabular-nums font-mono text-clinical-text-primary">{String(v)}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Feature ranges */}
        <section
          aria-label="Feature physiological ranges"
          className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm"
        >
          <h2 className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide mb-4">
            Feature Definitions
          </h2>
          <div className="space-y-1.5">
            {info.features.map(f => {
              const [lo, hi] = info.feature_ranges[f]
              const unit = info.feature_units[f]
              return (
                <div key={f} className="flex items-center justify-between py-1 border-b border-clinical-border last:border-0">
                  <span className="text-xs text-clinical-text-secondary">{fmtFeatureName(f)}</span>
                  <span className="text-xs tabular-nums font-mono text-clinical-text-muted">
                    [{lo}, {hi}]{unit ? ` ${unit}` : ''}
                  </span>
                </div>
              )
            })}
          </div>
        </section>
      </div>

      {/* Disclaimer */}
      <div className="rounded-lg border border-red-200 bg-red-50 p-4">
        <h3 className="text-xs font-semibold text-clinical-critical uppercase tracking-wide mb-2">
          Clinical Disclaimer
        </h3>
        <p className="text-xs text-red-700 leading-relaxed">
          This is a <strong>research prototype</strong> developed as a capstone project.
          It is <strong>NOT FDA-approved</strong> and is <strong>not intended for autonomous diagnostic use</strong>.
          All clinical decisions must be made by qualified medical professionals.
          Model performance (Macro F1 = 0.77) has been evaluated on held-out research data only.
          Validate rigorously before any clinical deployment.
        </p>
      </div>
    </div>
  )
}
