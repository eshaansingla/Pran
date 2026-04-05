import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { fetchModelInfo } from '../utils/api'
import type { ModelInfo } from '../types'
import { fmtFeatureName } from '../utils/formatters'
import { useStore } from '../store/useStore'

export default function ModelInfoPage() {
  const { isDark }    = useStore()
  const [info, setInfo]   = useState<ModelInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchModelInfo().then(setInfo).catch(e => setError(String(e)))
  }, [])

  if (error) return <p className="text-sm text-clinical-critical dark:text-red-400 p-4">{error}</p>
  if (!info)  return <p className="text-sm text-clinical-text-muted dark:text-slate-400 p-4">Loading model information…</p>

  const importanceData = Object.entries(info.global_importances)
    .sort((a, b) => b[1] - a[1])
    .map(([name, val]) => ({ name: fmtFeatureName(name), value: +(val * 100).toFixed(1) }))

  const metrics: Array<{ label: string; value: number; fmt: (v: number) => string }> = [
    { label: 'F1-Score',          value: info.metrics.f1,                fmt: v => v.toFixed(4) },
    { label: 'AUC',               value: info.metrics.auc,               fmt: v => v.toFixed(4) },
    { label: 'Precision',         value: info.metrics.precision,         fmt: v => v.toFixed(4) },
    { label: 'Recall',            value: info.metrics.recall,            fmt: v => v.toFixed(4) },
    { label: 'Specificity',       value: info.metrics.specificity,       fmt: v => v.toFixed(4) },
    { label: 'Balanced Accuracy', value: info.metrics.balanced_accuracy, fmt: v => v.toFixed(4) },
  ]

  const panelCls  = 'bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg p-5 shadow-sm'
  const divCls    = 'border-b border-clinical-border dark:border-slate-700 last:border-0'
  const labelCls  = 'text-sm text-clinical-text-secondary dark:text-slate-300'
  const valueCls  = 'text-sm font-semibold tabular-nums font-mono text-clinical-text-primary dark:text-slate-100'
  const headingCls = 'text-sm font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide mb-4'

  const barColors = isDark
    ? ['#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE', '#DBEAFE', '#EFF6FF', '#1D4ED8', '#1E40AF']
    : ['#2C5282', '#4A5568', '#718096', '#A0AEC0', '#CBD5E0', '#E2E8F0', '#1A202C', '#2D3748']

  return (
    <div className="space-y-5 max-w-4xl">
      <div>
        <h1 className="text-base font-semibold text-clinical-text-primary dark:text-slate-100">
          Model Information
        </h1>
        <p className="text-sm text-clinical-text-muted dark:text-slate-400 mt-0.5">
          XGBoost Binary v{info.version} — trained {info.training_date} —
          Normal (&lt;15 mmHg) vs Abnormal (≥15 mmHg)
        </p>
      </div>

      <div className="grid grid-cols-2 gap-5">
        {/* Performance metrics */}
        <section aria-label="Model performance metrics" className={panelCls}>
          <h2 className={headingCls}>Performance Metrics</h2>
          <div className="space-y-2">
            {metrics.map(m => (
              <div key={m.label} className={`flex items-center justify-between py-1.5 ${divCls}`}>
                <span className={labelCls}>{m.label}</span>
                <span className={valueCls}>{m.fmt(m.value)}</span>
              </div>
            ))}
          </div>
          <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-3">
            Evaluated on patient-stratified 20% hold-out ({info.training_data.charis_patients + info.training_data.mimic_patients} patients).
          </p>
        </section>

        {/* Training data */}
        <section aria-label="Training data statistics" className={panelCls}>
          <h2 className={headingCls}>Training Data</h2>
          <div className="space-y-2">
            {[
              { label: 'Dataset',          value: 'CHARIS + MIMIC-III' },
              { label: 'Total Patients',   value: String(info.training_data.charis_patients + info.training_data.mimic_patients) },
              { label: 'CHARIS (TBI ICU)', value: info.training_data.charis_patients + ' patients' },
              { label: 'MIMIC (Gen. ICU)', value: info.training_data.mimic_patients + ' patients' },
              { label: 'Total Windows',    value: info.training_data.total_windows.toLocaleString() },
              { label: 'Window Duration',  value: '10 seconds' },
              { label: 'Sampling Rate',    value: '125 Hz' },
            ].map(({ label, value }) => (
              <div key={label} className={`flex items-center justify-between py-1.5 ${divCls}`}>
                <span className={labelCls}>{label}</span>
                <span className={`text-sm font-medium text-clinical-text-primary dark:text-slate-100`}>{value}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Feature importance — full width */}
        <section aria-label="Global feature importance" className={`${panelCls} col-span-2`}>
          <h2 className={headingCls}>Global Feature Importance (Gain)</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={importanceData} layout="vertical" margin={{ left: 0, right: 16, top: 0, bottom: 0 }}>
              <XAxis
                type="number"
                tickFormatter={v => `${v}%`}
                tick={{ fontSize: 10, fill: isDark ? '#718096' : '#718096' }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                dataKey="name"
                type="category"
                width={150}
                tick={{ fontSize: 10, fill: isDark ? '#A0AEC0' : '#4A5568' }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                formatter={(v: number) => [`${v}%`, 'Importance']}
                contentStyle={{
                  fontSize: 11,
                  background: isDark ? '#2D3748' : '#fff',
                  border: isDark ? '1px solid #4A5568' : '1px solid #E2E8F0',
                  color: isDark ? '#E2E8F0' : '#1A202C',
                  borderRadius: 6,
                }}
              />
              <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                {importanceData.map((_, i) => (
                  <Cell key={i} fill={barColors[Math.min(i, barColors.length - 1)]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-2">
            Gain importance: average improvement in split criterion when a feature is used for splitting.
          </p>
        </section>

        {/* Hyperparameters */}
        <section aria-label="Model hyperparameters" className={panelCls}>
          <h2 className={headingCls}>Hyperparameters</h2>
          <div className="space-y-2">
            {Object.entries(info.hyperparameters).map(([k, v]) => (
              <div key={k} className={`flex items-center justify-between py-1.5 ${divCls}`}>
                <span className={`${labelCls} font-mono`}>{k}</span>
                <span className={`text-sm tabular-nums font-mono text-clinical-text-primary dark:text-slate-100`}>{String(v)}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Feature ranges */}
        <section aria-label="Feature physiological ranges" className={panelCls}>
          <h2 className={headingCls}>Feature Definitions</h2>
          <div className="space-y-1.5">
            {info.features.map(f => {
              const [lo, hi] = info.feature_ranges[f]
              const unit = info.feature_units[f]
              return (
                <div key={f} className={`flex items-center justify-between py-1 ${divCls}`}>
                  <span className={`text-xs text-clinical-text-secondary dark:text-slate-300`}>{fmtFeatureName(f)}</span>
                  <span className="text-xs tabular-nums font-mono text-clinical-text-muted dark:text-slate-400">
                    [{lo}, {hi}]{unit ? ` ${unit}` : ''}
                  </span>
                </div>
              )
            })}
          </div>
        </section>
      </div>

      {/* Disclaimer */}
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 p-4">
        <h3 className="text-xs font-semibold text-clinical-critical dark:text-red-400 uppercase tracking-wide mb-2">
          Clinical Disclaimer
        </h3>
        <p className="text-xs text-red-700 dark:text-red-300 leading-relaxed">
          This is a <strong>research prototype</strong> developed as a capstone project.
          It is <strong>NOT FDA-approved</strong> and is <strong>not intended for autonomous diagnostic use</strong>.
          All clinical decisions must be made by qualified medical professionals.
          Binary model v2.2 — 6 features, F1 = 0.89, AUC = 0.97 — evaluated on held-out research data only.
          Validate rigorously before any clinical deployment.
        </p>
      </div>
    </div>
  )
}
