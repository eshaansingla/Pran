import { TrendingUp, Clock, Info } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, ReferenceLine,
  ReferenceArea, ResponsiveContainer, CartesianGrid,
} from 'recharts'

// Mock forecast data — illustrates the future interface layout
const MOCK_HISTORY = Array.from({ length: 20 }, (_, i) => ({
  t: i - 20, value: 0.8 + 0.3 * Math.sin(i * 0.6) + Math.random() * 0.1,
}))
const MOCK_FORECAST = Array.from({ length: 12 }, (_, i) => ({
  t: i, value: 1.0 + 0.4 * Math.sin((20 + i) * 0.6),
  upper: 1.3 + 0.4 * Math.sin((20 + i) * 0.6),
  lower: 0.7 + 0.4 * Math.sin((20 + i) * 0.6),
}))

const INTEGRATION_POINTS = [
  { n: 1, text: 'Replace mock chart with real Recharts component fed from /api/predict_forecast' },
  { n: 2, text: 'Enable upload button → accept CSV with ≥30 consecutive windows (sequence input)' },
  { n: 3, text: 'Display predicted ICP trajectory with shaded 95% confidence intervals' },
  { n: 4, text: 'Add "Forecast Horizon" slider (5 – 30 min) → pass as query param' },
  { n: 5, text: 'Add uncertainty decomposition panel (aleatoric vs. epistemic)' },
]

export default function LSTMPlaceholder() {
  return (
    <div className="max-w-3xl mx-auto space-y-6 py-2">
      {/* Header card */}
      <div className="bg-white border border-clinical-border rounded-lg p-6 shadow-sm">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-lg bg-gray-100 text-gray-400">
            <TrendingUp size={32} strokeWidth={1.5} />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-clinical-text-primary">
              ICP Trend Forecasting
            </h2>
            <p className="text-sm text-clinical-text-secondary mt-1 max-w-xl">
              LSTM-based prediction of ICP trends 15–30 minutes ahead.
              Requires sequential data (minimum 30 consecutive 10-second windows,
              i.e. ≥5 minutes of continuous monitoring).
            </p>
          </div>
        </div>

        {/* Status badges */}
        <div className="flex items-center gap-3 mt-4">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-amber-50 text-amber-800 border border-amber-200 rounded-full text-xs font-medium">
            <Clock size={11} />
            In Development
          </span>
          <span className="text-xs text-clinical-text-muted">
            Expected Release: v2.0 · Q3 2026
          </span>
        </div>
      </div>

      {/* Mock chart — grayed out */}
      <div className="bg-white border border-clinical-border rounded-lg p-5 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide">
            Forecast Preview (Mock Data)
          </h3>
          <span className="text-xs text-clinical-text-muted italic">
            Illustrative only — not real predictions
          </span>
        </div>

        <div className="opacity-30 pointer-events-none select-none" aria-hidden="true">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <ReferenceArea x1={0} x2={11} fill="#FEF2F2" fillOpacity={0.5} />
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
              <XAxis
                dataKey="t"
                type="number"
                allowDuplicatedCategory={false}
                tick={{ fontSize: 9 }}
                label={{ value: 'Minutes from now', position: 'insideBottomRight', offset: -4, fontSize: 9 }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 2.5]}
                ticks={[0, 1, 2]}
                tickFormatter={v => (['Normal','Elevated','Critical'] as string[])[v] ?? ''}
                tick={{ fontSize: 9 }}
                tickLine={false}
                axisLine={false}
                width={58}
              />
              <ReferenceLine x={0} stroke="#6B7280" strokeDasharray="4 3" label={{ value: 'Now', fontSize: 9, fill: '#6B7280' }} />
              <Line data={MOCK_HISTORY} dataKey="value" stroke="#2C5282" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line data={MOCK_FORECAST} dataKey="value" stroke="#DC2626" strokeWidth={2} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
              <Line data={MOCK_FORECAST} dataKey="upper" stroke="#DC2626" strokeWidth={1} strokeDasharray="2 2" dot={false} isAnimationActive={false} />
              <Line data={MOCK_FORECAST} dataKey="lower" stroke="#DC2626" strokeWidth={1} strokeDasharray="2 2" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Legend */}
        <div className="flex gap-5 mt-2 pl-12 opacity-40">
          <div className="flex items-center gap-1.5 text-xs text-clinical-text-muted">
            <span className="w-8 h-0.5 bg-[#2C5282] inline-block" />
            Historical ICP
          </div>
          <div className="flex items-center gap-1.5 text-xs text-clinical-text-muted">
            <span className="w-8 h-0.5 bg-[#DC2626] border-dashed inline-block border-t-2 border-[#DC2626]" style={{ backgroundImage: 'repeating-linear-gradient(90deg,#DC2626 0 5px,transparent 5px 8px)', backgroundColor: 'transparent', height: 2 }} />
            Forecast + 95% CI
          </div>
        </div>

        {/* Disabled button */}
        <div className="mt-4 flex items-center gap-3">
          <button
            disabled
            aria-disabled="true"
            className="px-5 py-2 text-sm font-medium bg-clinical-primary text-white rounded-lg opacity-30 cursor-not-allowed"
          >
            Upload Sequence Data
          </button>
          <span className="text-xs text-clinical-text-muted">
            Minimum 30 consecutive windows required
          </span>
        </div>
      </div>

      {/* Future integration roadmap */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-3">
          <Info size={14} className="text-clinical-primary" />
          <h3 className="text-sm font-semibold text-clinical-primary">
            v2.0 Integration Roadmap
          </h3>
        </div>
        <ol className="space-y-2">
          {INTEGRATION_POINTS.map(p => (
            <li key={p.n} className="flex gap-3 text-xs text-blue-700">
              <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-200 text-clinical-primary font-semibold flex items-center justify-center text-2xs">
                {p.n}
              </span>
              <span>{p.text}</span>
            </li>
          ))}
        </ol>
      </div>

      {/* API stub notice */}
      <div className="rounded-lg border border-clinical-border bg-white p-4 shadow-sm">
        <h3 className="text-xs font-semibold text-clinical-text-secondary uppercase tracking-wide mb-2">
          API Stub
        </h3>
        <pre className="text-xs font-mono text-clinical-text-muted bg-gray-50 rounded p-3 overflow-x-auto">
{`POST /api/predict_forecast
→ 501 Not Implemented

{
  "error": "LSTM forecasting not yet available",
  "status": "in_development",
  "expected_release": "v2.0 (Q3 2026)"
}`}
        </pre>
      </div>
    </div>
  )
}
