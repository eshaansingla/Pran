import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { fetchModelInfo, fetchLstmInfo } from '../utils/api'
import type { ModelInfo } from '../types'
import { fmtFeatureName } from '../utils/formatters'
import { useStore } from '../store/useStore'
import { Cpu, Zap, Database, TrendingUp, CheckCircle, AlertTriangle, Layers, FlaskConical } from 'lucide-react'

// ─── Types ────────────────────────────────────────────────────────────────────

interface LstmMeta {
  version: string
  training_date: string
  seq_len: number
  horizon: number
  horizon_minutes: number
  n_features: number
  feature_names: string[]
  threshold: number
  metrics: {
    auc: number
    f1: number
    precision: number
    recall: number
    specificity: number
    balanced_accuracy: number
    early_warning_rate: number
    tn: number; fp: number; fn: number; tp: number
  }
  training_data: {
    total_sequences: number
    train_sequences: number
    val_sequences: number
    test_sequences: number
  }
  architecture: {
    bilstm_units: number
    dense_units: number
    dropout: number
    batch_size: number
    optimizer: string
    learning_rate: number
  }
}

// ─── Primitives ───────────────────────────────────────────────────────────────

const card = 'bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-700 rounded-xl shadow-sm'

function SectionHeader({ icon: Icon, title, badge, color = 'blue' }: {
  icon: typeof Cpu
  title: string
  badge?: string
  color?: 'blue' | 'purple'
}) {
  const colors = {
    blue:   'from-blue-50 to-transparent dark:from-blue-950/30 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-300',
    purple: 'from-purple-50 to-transparent dark:from-purple-950/30 border-purple-200 dark:border-purple-800 text-purple-800 dark:text-purple-300',
  }
  const iconColors = {
    blue:   'text-blue-600 dark:text-blue-400',
    purple: 'text-purple-600 dark:text-purple-400',
  }
  const badgeColors = {
    blue:   'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-700',
    purple: 'bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-400 border border-purple-200 dark:border-purple-700',
  }
  return (
    <div className={`flex items-center gap-3 px-5 py-3.5 rounded-xl border bg-gradient-to-r mb-4 ${colors[color]}`}>
      <Icon size={16} className={iconColors[color]} />
      <span className="text-sm font-bold">{title}</span>
      {badge && (
        <span className={`text-2xs px-2 py-0.5 rounded-full font-semibold ml-1 ${badgeColors[color]}`}>{badge}</span>
      )}
    </div>
  )
}

function CardHeader({ icon: Icon, title }: { icon: typeof Cpu; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4 pb-3 border-b border-clinical-border dark:border-slate-700">
      <Icon size={13} className="text-clinical-text-muted dark:text-slate-400" />
      <span className="text-xs font-bold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-widest">{title}</span>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-clinical-border dark:border-slate-700 last:border-0">
      <span className="text-sm text-clinical-text-secondary dark:text-slate-300">{label}</span>
      <span className="text-sm font-semibold tabular-nums font-mono text-clinical-text-primary dark:text-slate-100">{value}</span>
    </div>
  )
}

function StatHighlight({ value, label, sub }: { value: string; label: string; sub?: string }) {
  return (
    <div className="text-center py-3">
      <p className="text-2xl font-bold tabular-nums text-clinical-text-primary dark:text-slate-100">{value}</p>
      <p className="text-xs font-medium text-clinical-text-secondary dark:text-slate-300 mt-0.5">{label}</p>
      {sub && <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-0.5">{sub}</p>}
    </div>
  )
}

function MetricBadge({ value, target, label }: { value: number; target: number; label: string }) {
  const pass = value >= target
  return (
    <div className="flex flex-col items-center gap-1.5 py-3">
      <div className={`flex items-center gap-1 text-2xs font-bold px-2.5 py-0.5 rounded-full ${
        pass
          ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-400'
          : 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-400'
      }`}>
        {pass ? <CheckCircle size={9} /> : <AlertTriangle size={9} />}
        {pass ? 'PASS' : 'FAIL'}
      </div>
      <p className="text-xl font-bold tabular-nums text-clinical-text-primary dark:text-slate-100">
        {(value * 100).toFixed(1)}%
      </p>
      <p className="text-xs font-medium text-clinical-text-secondary dark:text-slate-300">{label}</p>
      <p className="text-2xs text-clinical-text-muted dark:text-slate-500">target ≥{(target * 100).toFixed(0)}%</p>
    </div>
  )
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export default function ModelInfoPage() {
  const { isDark }      = useStore()
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [lstm, setLstm] = useState<LstmMeta | null>(null)
  const [err, setErr]   = useState<string | null>(null)

  useEffect(() => {
    fetchModelInfo().then(setInfo).catch(e => setErr(String(e)))
    fetchLstmInfo().then(d => d && setLstm(d as unknown as LstmMeta)).catch(() => {})
  }, [])

  if (err)   return <p className="text-sm text-clinical-critical dark:text-red-400 p-4">{err}</p>
  if (!info) return <p className="text-sm text-clinical-text-muted dark:text-slate-400 p-4">Loading model information…</p>

  const importanceData = Object.entries(info.global_importances)
    .sort((a, b) => b[1] - a[1])
    .map(([name, val]) => ({ name: fmtFeatureName(name), value: +(val * 100).toFixed(1) }))

  const barColors = isDark
    ? ['#3B82F6','#60A5FA','#93C5FD','#BFDBFE','#DBEAFE','#EFF6FF']
    : ['#2C5282','#4A5568','#718096','#A0AEC0','#CBD5E0','#E2E8F0']

  return (
    <div className="space-y-8 max-w-5xl animate-fade-in-up">

      {/* ── Page heading ──────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-base font-bold text-clinical-text-primary dark:text-slate-100">Model Information</h1>
          <p className="text-sm text-clinical-text-muted dark:text-slate-400 mt-0.5">
            Live model metadata · calibrated probabilities · feature importances
          </p>
        </div>
        <div className="flex gap-2">
          <span className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-700 rounded-lg font-medium">
            <Cpu size={12} />XGBoost v{info.version}
          </span>
          {lstm && (
            <span className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 border border-purple-200 dark:border-purple-700 rounded-lg font-medium">
              <Layers size={12} />LSTM v{lstm.version}
            </span>
          )}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════════
          XGBOOST SECTION
      ══════════════════════════════════════════════════════════════════════ */}
      <section aria-label="XGBoost model">
        <SectionHeader icon={Cpu} title="XGBoost Binary Classifier — Instant ICP Classification" badge={`v${info.version}`} color="blue" />

        {/* Row 1: key stat strip */}
        <div className={`${card} p-0 mb-4 overflow-hidden`}>
          <div className="grid grid-cols-3 divide-x divide-clinical-border dark:divide-slate-700">
            <StatHighlight value={(info.metrics.f1 * 100).toFixed(1) + '%'}   label="F1-Score"          sub="primary metric" />
            <StatHighlight value={(info.metrics.auc * 100).toFixed(1) + '%'}  label="AUC-ROC"           sub="discrimination" />
            <StatHighlight value={(info.metrics.balanced_accuracy * 100).toFixed(1) + '%'} label="Balanced Accuracy" sub="imbalanced-robust" />
          </div>
        </div>

        {/* Row 2: metrics + training data */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <section className={`${card} p-5`} aria-label="XGBoost performance metrics">
            <CardHeader icon={Zap} title="Performance Metrics" />
            <div className="space-y-0">
              {[
                { label: 'F1-Score',          value: info.metrics.f1 },
                { label: 'AUC-ROC',           value: info.metrics.auc },
                { label: 'Precision',         value: info.metrics.precision },
                { label: 'Recall',            value: info.metrics.recall },
                { label: 'Specificity',       value: info.metrics.specificity },
                { label: 'Balanced Accuracy', value: info.metrics.balanced_accuracy },
              ].map(m => <Row key={m.label} label={m.label} value={m.value.toFixed(4)} />)}
            </div>
            <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-3">
              Patient-stratified 20% hold-out · {info.training_data.charis_patients + info.training_data.mimic_patients} patients.
              {info.calibrated && ' Isotonic-calibrated.'}
            </p>
          </section>

          <section className={`${card} p-5`} aria-label="XGBoost training data">
            <CardHeader icon={Database} title="Training Data" />
            <div className="space-y-0">
              {[
                { label: 'Datasets',          value: 'CHARIS + MIMIC-III' },
                { label: 'Total Patients',    value: String(info.training_data.charis_patients + info.training_data.mimic_patients) },
                { label: 'CHARIS (TBI ICU)',  value: `${info.training_data.charis_patients} patients` },
                { label: 'MIMIC (Gen. ICU)',  value: `${info.training_data.mimic_patients} patients` },
                { label: 'Total Windows',     value: info.training_data.total_windows.toLocaleString() },
                { label: 'Window Duration',   value: '10 seconds' },
                { label: 'ICP Threshold',     value: '15 mmHg' },
              ].map(r => <Row key={r.label} label={r.label} value={r.value} />)}
            </div>
          </section>
        </div>

        {/* Row 3: feature importance (full width) */}
        <div className={`${card} p-5 mb-4`}>
          <CardHeader icon={TrendingUp} title="Global Feature Importance (Gain)" />
          <ResponsiveContainer width="100%" height={190}>
            <BarChart data={importanceData} layout="vertical" margin={{ left: 0, right: 24, top: 0, bottom: 0 }}>
              <XAxis type="number" tickFormatter={v => `${v}%`}
                tick={{ fontSize: 10, fill: isDark ? '#718096' : '#718096' }} tickLine={false} axisLine={false} />
              <YAxis dataKey="name" type="category" width={155}
                tick={{ fontSize: 10, fill: isDark ? '#A0AEC0' : '#4A5568' }} tickLine={false} axisLine={false} />
              <Tooltip formatter={(v: number) => [`${v}%`, 'Importance']}
                contentStyle={{
                  fontSize: 11, borderRadius: 8,
                  background: isDark ? '#2D3748' : '#fff',
                  border: isDark ? '1px solid #4A5568' : '1px solid #E2E8F0',
                  color: isDark ? '#E2E8F0' : '#1A202C',
                }} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {importanceData.map((_, i) => (
                  <Cell key={i} fill={barColors[Math.min(i, barColors.length - 1)]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-1">
            Gain importance: average loss improvement per split. Ablation removed head_angle + motion_artifact_flag (0% gain).
          </p>
        </div>

        {/* Row 4: hyperparameters + feature definitions */}
        <div className="grid grid-cols-2 gap-4">
          <section className={`${card} p-5`} aria-label="Hyperparameters">
            <CardHeader icon={Cpu} title="Hyperparameters" />
            <div className="space-y-0">
              {Object.entries(info.hyperparameters).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between py-1.5 border-b border-clinical-border dark:border-slate-700 last:border-0">
                  <span className="text-sm font-mono text-clinical-text-secondary dark:text-slate-300">{k}</span>
                  <span className="text-sm tabular-nums font-mono font-semibold text-clinical-text-primary dark:text-slate-100">{String(v)}</span>
                </div>
              ))}
            </div>
          </section>

          <section className={`${card} p-5`} aria-label="Feature definitions">
            <CardHeader icon={Database} title="Feature Definitions" />
            <div className="space-y-0">
              {info.features.map(f => {
                const [lo, hi] = info.feature_ranges[f]
                const unit = info.feature_units[f]
                return (
                  <div key={f} className="flex items-center justify-between py-1.5 border-b border-clinical-border dark:border-slate-700 last:border-0">
                    <span className="text-sm text-clinical-text-secondary dark:text-slate-300">{fmtFeatureName(f)}</span>
                    <span className="text-xs tabular-nums font-mono text-clinical-text-muted dark:text-slate-400">
                      [{lo}–{hi}]{unit ? ` ${unit}` : ''}
                    </span>
                  </div>
                )
              })}
            </div>
          </section>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════════
          LSTM SECTION
      ══════════════════════════════════════════════════════════════════════ */}
      {lstm ? (
        <section aria-label="LSTM model" className="animate-fade-in-up">
          <SectionHeader icon={Layers} title="Bidirectional LSTM — 15-Minute Ahead Forecasting" badge={`v${lstm.version}`} color="purple" />

          {/* Row 1: key metric badges (full width) */}
          <div className={`${card} p-5 mb-4`}>
            <CardHeader icon={Zap} title="Test-Set Performance — 15-min Ahead Forecast" />
            <div className="grid grid-cols-4 divide-x divide-clinical-border dark:divide-slate-700 mb-4">
              <MetricBadge value={lstm.metrics.auc}                target={0.90} label="AUC-ROC" />
              <MetricBadge value={lstm.metrics.f1}                 target={0.75} label="F1-Score" />
              <MetricBadge value={lstm.metrics.early_warning_rate} target={0.60} label="Early Warning Rate" />
              <MetricBadge value={lstm.metrics.balanced_accuracy}  target={0.85} label="Balanced Accuracy" />
            </div>
            <div className="h-px bg-clinical-border dark:bg-slate-700 mb-4" />
            <div className="grid grid-cols-3 divide-x divide-clinical-border dark:divide-slate-700">
              {[
                { label: 'Precision',   value: lstm.metrics.precision },
                { label: 'Recall',      value: lstm.metrics.recall },
                { label: 'Specificity', value: lstm.metrics.specificity },
              ].map(m => (
                <StatHighlight key={m.label}
                  value={(m.value * 100).toFixed(1) + '%'}
                  label={m.label}
                />
              ))}
            </div>
            <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-4">
              Evaluated on patient-stratified test set ({lstm.training_data.test_sequences.toLocaleString()} sequences).
              Early Warning Rate = recall on 15-min-ahead forecast —
              correctly identified {(lstm.metrics.early_warning_rate * 100).toFixed(0)}% of impending Abnormal ICP episodes.
            </p>
          </div>

          {/* Row 2: confusion matrix + architecture */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <section className={`${card} p-5`} aria-label="LSTM confusion matrix">
              <CardHeader icon={Cpu} title="Confusion Matrix (Test Set)" />
              <div className="space-y-2">
                {/* header row */}
                <div className="grid grid-cols-[1fr_1fr_1fr] gap-2 text-center">
                  <div />
                  <div className="text-2xs font-bold text-emerald-600 dark:text-emerald-400">Pred Normal</div>
                  <div className="text-2xs font-bold text-red-600 dark:text-red-400">Pred Abnormal</div>
                </div>
                {/* TN / FP */}
                <div className="grid grid-cols-[1fr_1fr_1fr] gap-2 items-center">
                  <div className="text-2xs text-clinical-text-muted dark:text-slate-400 text-right pr-2">True Normal</div>
                  <div className="bg-emerald-100 dark:bg-emerald-900/40 rounded-lg py-3 text-center">
                    <p className="text-base font-bold text-emerald-700 dark:text-emerald-400">{lstm.metrics.tn.toLocaleString()}</p>
                    <p className="text-2xs text-emerald-600/70 dark:text-emerald-500">TN</p>
                  </div>
                  <div className="bg-red-50 dark:bg-red-900/20 rounded-lg py-3 text-center">
                    <p className="text-base font-bold text-red-500 dark:text-red-400">{lstm.metrics.fp.toLocaleString()}</p>
                    <p className="text-2xs text-red-400/70 dark:text-red-500">FP</p>
                  </div>
                </div>
                {/* FN / TP */}
                <div className="grid grid-cols-[1fr_1fr_1fr] gap-2 items-center">
                  <div className="text-2xs text-clinical-text-muted dark:text-slate-400 text-right pr-2">True Abnormal</div>
                  <div className="bg-red-50 dark:bg-red-900/20 rounded-lg py-3 text-center">
                    <p className="text-base font-bold text-red-500 dark:text-red-400">{lstm.metrics.fn.toLocaleString()}</p>
                    <p className="text-2xs text-red-400/70 dark:text-red-500">FN</p>
                  </div>
                  <div className="bg-emerald-100 dark:bg-emerald-900/40 rounded-lg py-3 text-center">
                    <p className="text-base font-bold text-emerald-700 dark:text-emerald-400">{lstm.metrics.tp.toLocaleString()}</p>
                    <p className="text-2xs text-emerald-600/70 dark:text-emerald-500">TP</p>
                  </div>
                </div>
              </div>
              <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-3">
                Decision threshold: {lstm.threshold.toFixed(4)} (F1-optimised on validation set).
              </p>
            </section>

            <section className={`${card} p-5`} aria-label="LSTM architecture">
              <CardHeader icon={Layers} title="Architecture & Training" />
              {/* Layer stack */}
              <div className="space-y-1.5 mb-4">
                {[
                  { step: 'Input',     desc: `${lstm.seq_len} timesteps × ${lstm.n_features} features` },
                  { step: 'BiLSTM',    desc: `${lstm.architecture.bilstm_units} units, return_sequences=True` },
                  { step: 'Attention', desc: 'TimeDistributed Dense(1, tanh) → Softmax' },
                  { step: 'Dense',     desc: `${lstm.architecture.dense_units} units, ReLU + Dropout(${lstm.architecture.dropout})` },
                  { step: 'Output',    desc: 'Dense(1, sigmoid) → P(Abnormal)' },
                ].map(({ step, desc }) => (
                  <div key={step} className="flex items-start gap-3">
                    <span className="text-2xs font-mono font-bold text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-900/30 px-1.5 py-0.5 rounded w-16 text-center flex-shrink-0 mt-0.5">
                      {step}
                    </span>
                    <span className="text-xs font-mono text-clinical-text-primary dark:text-slate-200">{desc}</span>
                  </div>
                ))}
              </div>
              <div className="h-px bg-clinical-border dark:bg-slate-700 mb-3" />
              <div className="space-y-0">
                {[
                  { label: 'Optimizer',        value: lstm.architecture.optimizer },
                  { label: 'Learning Rate',    value: String(lstm.architecture.learning_rate) },
                  { label: 'Batch Size',       value: String(lstm.architecture.batch_size) },
                  { label: 'Uncertainty Est.', value: 'MC Dropout (20 passes)' },
                  { label: 'Forecast Horizon', value: `${lstm.horizon_minutes} min ahead` },
                ].map(r => <Row key={r.label} label={r.label} value={r.value} />)}
              </div>
            </section>
          </div>

          {/* Row 3: training data (full width) */}
          <div className={`${card} p-5 mb-4`}>
            <CardHeader icon={Database} title="Training Data (Sequences)" />
            <div className="grid grid-cols-3 gap-4 mb-4">
              {[
                { label: 'Train', value: lstm.training_data.train_sequences.toLocaleString(), sub: 'sequences' },
                { label: 'Val',   value: lstm.training_data.val_sequences.toLocaleString(),   sub: 'sequences' },
                { label: 'Test',  value: lstm.training_data.test_sequences.toLocaleString(),  sub: 'sequences' },
              ].map(s => (
                <div key={s.label} className="text-center py-2 bg-slate-50 dark:bg-slate-700/40 rounded-lg">
                  <p className="text-xl font-bold tabular-nums text-clinical-text-primary dark:text-slate-100">{s.value}</p>
                  <p className="text-xs text-clinical-text-muted dark:text-slate-400 mt-0.5">{s.label} {s.sub}</p>
                </div>
              ))}
            </div>
            <div className="space-y-0">
              {[
                { label: 'Total Sequences',  value: lstm.training_data.total_sequences.toLocaleString() },
                { label: 'History Window',   value: `${lstm.seq_len} windows = 5 min` },
                { label: 'Forecast Target',  value: `Window +${lstm.horizon} (${lstm.horizon_minutes} min ahead)` },
                { label: 'Window Duration',  value: '10 seconds / window' },
                { label: 'Split Strategy',   value: 'Patient-stratified GroupShuffleSplit (70/10/20)' },
                { label: 'Training Date',    value: lstm.training_date },
              ].map(r => <Row key={r.label} label={r.label} value={r.value} />)}
            </div>
          </div>

          {/* Row 4: side-by-side comparison (full width) */}
          <div className={`${card} p-5`}>
            <CardHeader icon={TrendingUp} title="Model Comparison" />
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-clinical-border dark:border-slate-700">
                    <th className="text-left py-2 pr-4 text-xs font-semibold text-clinical-text-muted dark:text-slate-400 uppercase tracking-wide w-40">Metric</th>
                    <th className="text-center py-2 px-6 text-xs font-semibold text-blue-700 dark:text-blue-400 uppercase tracking-wide">
                      <span className="flex items-center justify-center gap-1.5"><Cpu size={11} />XGBoost v{info.version}</span>
                    </th>
                    <th className="text-center py-2 pl-6 text-xs font-semibold text-purple-700 dark:text-purple-400 uppercase tracking-wide">
                      <span className="flex items-center justify-center gap-1.5"><Layers size={11} />LSTM v{lstm.version}</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: 'Task',          xgb: 'Instant classification',    lstmVal: '15-min ahead forecast' },
                    { label: 'AUC-ROC',       xgb: info.metrics.auc.toFixed(4), lstmVal: lstm.metrics.auc.toFixed(4),               num: true },
                    { label: 'F1-Score',      xgb: info.metrics.f1.toFixed(4),  lstmVal: lstm.metrics.f1.toFixed(4),                num: true },
                    { label: 'Precision',     xgb: info.metrics.precision.toFixed(4),        lstmVal: lstm.metrics.precision.toFixed(4),        num: true },
                    { label: 'Recall',        xgb: info.metrics.recall.toFixed(4),           lstmVal: lstm.metrics.recall.toFixed(4),           num: true },
                    { label: 'Specificity',   xgb: info.metrics.specificity.toFixed(4),      lstmVal: lstm.metrics.specificity.toFixed(4),      num: true },
                    { label: 'Balanced Acc.', xgb: info.metrics.balanced_accuracy.toFixed(4),lstmVal: lstm.metrics.balanced_accuracy.toFixed(4),num: true },
                    { label: 'Input',         xgb: '1 window × 6 features',    lstmVal: `${lstm.seq_len} windows × 6 features` },
                    { label: 'Model Size',    xgb: '45.9 KB (gzipped)',          lstmVal: '~536 KB' },
                    { label: 'Uncertainty',   xgb: 'Isotonic calibration',      lstmVal: 'MC Dropout (20 passes)' },
                  ].map(({ label, xgb, lstmVal, num }) => {
                    const xgbWins  = num ? parseFloat(xgb) > parseFloat(lstmVal) : false
                    const lstmWins = num ? parseFloat(lstmVal) > parseFloat(xgb) : false
                    return (
                      <tr key={label} className="border-b border-clinical-border dark:border-slate-700 last:border-0 hover:bg-slate-50 dark:hover:bg-slate-700/30 transition-colors">
                        <td className="py-2.5 pr-4 text-xs font-medium text-clinical-text-secondary dark:text-slate-300">{label}</td>
                        <td className={`py-2.5 px-6 text-center text-xs font-mono tabular-nums ${xgbWins ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-clinical-text-primary dark:text-slate-100'}`}>
                          {xgb}
                        </td>
                        <td className={`py-2.5 pl-6 text-center text-xs font-mono tabular-nums ${lstmWins ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-clinical-text-primary dark:text-slate-100'}`}>
                          {lstmVal}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-3">
              Bold green = better value. LSTM trades marginal metric advantage for 15-min early warning capability.
              Use XGBoost for real-time monitoring; LSTM for anticipatory alerts.
            </p>
          </div>
        </section>
      ) : (
        <div className={`${card} p-6 flex items-center gap-5`}>
          <div className="p-3 bg-slate-100 dark:bg-slate-700 rounded-xl flex-shrink-0">
            <Layers size={28} className="text-slate-400 dark:text-slate-500" />
          </div>
          <div>
            <p className="text-sm font-semibold text-clinical-text-primary dark:text-slate-200">LSTM Forecaster — Not Yet Trained</p>
            <p className="text-xs text-clinical-text-muted dark:text-slate-400 mt-1 leading-relaxed">
              Run <code className="font-mono bg-slate-100 dark:bg-slate-700 px-1.5 py-0.5 rounded text-xs">python src/models/lstm_forecaster.py</code> from the project root to train the LSTM model and unlock this section.
            </p>
          </div>
        </div>
      )}

      {/* ── Disclaimer ────────────────────────────────────────────────────── */}
      <div className="rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 p-4 flex items-start gap-3">
        <FlaskConical size={16} className="text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
        <div>
          <p className="text-xs font-bold text-amber-800 dark:text-amber-400 uppercase tracking-widest mb-1">Research Prototype — Clinical Disclaimer</p>
          <p className="text-xs text-amber-700 dark:text-amber-300 leading-relaxed">
            <strong>NOT FDA-approved</strong> and not intended for autonomous diagnostic use.
            All clinical decisions must be made by qualified medical professionals.
            Validate rigorously before any clinical deployment.
          </p>
        </div>
      </div>

    </div>
  )
}
