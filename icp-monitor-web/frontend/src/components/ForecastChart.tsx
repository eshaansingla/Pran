import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ReferenceLine,
  ReferenceArea, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import type { ForecastResult } from '../types'
import { useStore } from '../store/useStore'
import { probToICP } from '../utils/formatters'

interface Props {
  sequence:   number[][]
  result:     ForecastResult
  histProbs?: number[]     // per-window XGBoost P(abnormal) [0-1]
}

interface ChartPoint {
  tMin:          number
  label:         string
  histProb?:     number    // 0-100 %
  forecastProb?: number    // 0-100 %
  ciLower?:      number    // 0-100 %
  ciUpper?:      number    // 0-100 %
  histICP?:      number    // mmHg (right axis)
  forecastICP?:  number    // mmHg (right axis)
}

// ─────────────────────────────────────────────────────────────────────────────
// Build chart data
// Historical : XGBoost P(abnormal) × 100  — raw, no ICP conversion
// Forecast   : LSTM forecast_probabilities × 100 — raw, no ICP conversion
// ─────────────────────────────────────────────────────────────────────────────
const HIST_DISPLAY = 30

function pct(p: number) { return +Math.max(0, Math.min(100, p * 100)).toFixed(1) }

function buildChartData(
  sequence: number[][],
  result:   ForecastResult,
  histProbs?: number[],
): ChartPoint[] {
  const horizonMin = result.horizon_minutes
  const displaySeq = sequence.slice(-HIST_DISPLAY)
  const dispLen    = displaySeq.length

  // Raw per-window XGBoost probability (0-100%)
  // When histProbs unavailable (e.g. reload from history), use forecast's
  // first probability as a consistent baseline instead of a misleading 10%.
  const fallbackProb = pct(result.forecast_probabilities?.[0] ?? result.probability)
  const rawProbs = displaySeq.map((_, i) => {
    if (histProbs && histProbs.length === sequence.length) {
      return pct(histProbs[sequence.length - dispLen + i])
    }
    return fallbackProb
  })

  // 3-point rolling smooth
  const smooth = rawProbs.map((_, i) => {
    const sl = rawProbs.slice(Math.max(0, i - 1), Math.min(rawProbs.length, i + 2))
    return +(sl.reduce((a, b) => a + b, 0) / sl.length).toFixed(1)
  })

  // Historical points
  const XGB_THR = 0.545  // XGBoost threshold for probToICP anchor
  const points: ChartPoint[] = []
  for (let i = 0; i < dispLen; i++) {
    const tMin      = -(dispLen - 1 - i) * 10 / 60
    const isTickMin = Math.round(tMin) === tMin
    points.push({
      tMin:     +tMin.toFixed(3),
      label:    i === dispLen - 1 ? 'Now' : isTickMin ? `${Math.round(tMin)}m` : '',
      histProb: smooth[i],
      histICP:  +probToICP(smooth[i] / 100, XGB_THR).toFixed(1),
    })
  }

  // Anchor: bridge history → forecast at t=0
  // Use the LSTM's first prediction blended with history for a smooth visual
  // transition.  This prevents the jarring cliff-drop when XGBoost and LSTM
  // disagree on the current state.
  const lastProb  = smooth[dispLen - 1]
  const fProbs = result.forecast_probabilities || Array(horizonMin).fill(result.probability)
  const ciLo   = result.forecast_ci_lower      || Array(horizonMin).fill(result.ci_lower)
  const ciHi   = result.forecast_ci_upper      || Array(horizonMin).fill(result.ci_upper)

  // Bridge point: blend of last history and first forecast
  const firstFP    = pct(fProbs[0])
  const bridgeProb = lastProb * 0.7 + firstFP * 0.3  // smooth transition
  const lastPt     = points[points.length - 1]
  lastPt.forecastProb = bridgeProb
  lastPt.forecastICP  = +probToICP(bridgeProb / 100, XGB_THR).toFixed(1)
  lastPt.ciLower      = bridgeProb
  lastPt.ciUpper      = bridgeProb

  // Forecast trajectory with momentum-decayed anchor blending.
  // First few minutes blend the last historical probability with the LSTM
  // output so the line transitions smoothly.  By minute 5 the blend is 0
  // and the chart shows pure LSTM output.  This is visually correct because
  // physiological parameters don't change instantaneously — a patient at
  // 80% P(Abnormal) at t=0 cannot realistically be 5% at t=+1 min.
  for (let m = 1; m <= horizonMin; m++) {
    const idx = m - 1
    const rawFP = pct(fProbs[idx])

    // Exponential blend: 0.5 at m=1, decaying to ~0 by m=5
    const blend = Math.max(0, Math.pow(0.5, m))
    const fp    = +(blend * lastProb + (1 - blend) * rawFP).toFixed(1)

    const lo  = pct(Math.max(0.001, ciLo[idx]))
    const hi  = pct(Math.min(0.999, ciHi[idx]))
    points.push({
      tMin:        m,
      label:       m === horizonMin ? `+${m}m` : m % 5 === 0 ? `+${m}m` : '',
      forecastProb: fp,
      forecastICP:  +probToICP(fp / 100, XGB_THR).toFixed(1),
      ciLower:     Math.min(fp, +(blend * lastProb + (1 - blend) * lo).toFixed(1)),
      ciUpper:     Math.max(fp, +(blend * lastProb + (1 - blend) * hi).toFixed(1)),
    })
  }

  return points
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

function ForecastTooltip({ active, payload, isDark, result }: {
  active?:  boolean
  payload?: Array<{ value: number; payload: ChartPoint }>
  isDark?:  boolean
  result:   ForecastResult
}) {
  if (!active || !payload?.length) return null
  const pt  = payload[0].payload
  const bg  = isDark ? '#1E293B' : '#fff'
  const br  = isDark ? '1px solid #334155' : '1px solid #E2E8F0'
  const tx  = isDark ? '#E2E8F0' : '#1A202C'
  const mu  = isDark ? '#94A3B8' : '#6B7280'
  const val = pt.forecastProb ?? pt.histProb
  const thr = +(result.threshold * 100).toFixed(1)
  const isAbn = val !== undefined && val >= thr

  return (
    <div style={{ background: bg, border: br, borderRadius: 8, padding: '8px 12px', fontSize: 11, color: tx, minWidth: 160 }}>
      <p style={{ fontWeight: 700, marginBottom: 4 }}>
        {pt.tMin <= 0 ? `History  ${pt.label}` : pt.label || `t +${pt.tMin.toFixed(0)}m`}
      </p>
      {val !== undefined && (
        <p style={{ color: isAbn ? '#DC2626' : '#059669', fontWeight: 600, marginBottom: 2 }}>
          P(Abnormal):&nbsp;{val.toFixed(1)}%
          <span style={{ color: mu, fontWeight: 400, marginLeft: 6, fontSize: 10 }}>
            {isAbn ? 'above threshold' : 'below threshold'}
          </span>
        </p>
      )}
      {pt.tMin <= 0 && (
        <p style={{ color: mu, fontSize: 10 }}>XGBoost — historical window</p>
      )}
      {pt.tMin > 0 && pt.ciLower !== undefined && pt.ciUpper !== undefined && pt.ciLower !== pt.ciUpper && (
        <p style={{ color: mu, fontSize: 10, marginTop: 2 }}>
          95% CI: {pt.ciLower.toFixed(1)}%–{pt.ciUpper.toFixed(1)}%
        </p>
      )}
      {pt.tMin > 0 && (
        <p style={{ color: mu, fontSize: 10, marginTop: 1 }}>
          Peak: {(result.probability * 100).toFixed(1)}% ({result.class_name}) · thr {thr}%
        </p>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ForecastChart({ sequence, result, histProbs }: Props) {
  const { isDark } = useStore()
  const data       = buildChartData(sequence, result, histProbs)
  const isAbn      = result.class === 1
  const thrPct     = +(result.threshold * 100).toFixed(1)

  // Only show historical amber line when there is meaningful variation (range > 5 pp).
  // A flat line at 2% for 5 minutes adds no clinical information and looks broken.
  const showHistLine = (() => {
    const vals = histProbs ?? []
    if (vals.length < 2) return false
    const mn = Math.min(...vals) * 100
    const mx = Math.max(...vals) * 100
    return mx - mn > 5
  })()

  const histColor  = isDark ? '#F59E0B' : '#D97706'
  const foreColor  = isAbn
    ? (isDark ? '#EF4444' : '#DC2626')
    : (isDark ? '#10B981' : '#059669')
  const gridColor  = isDark ? '#1E293B' : '#E2E8F0'
  const tickColor  = isDark ? '#64748B' : '#94A3B8'
  const bgColor    = isDark ? '#0F172A' : '#fff'
  const hMin       = result.horizon_minutes || 15
  const xTicks     = [-5, -4, -3, -2, -1, 0]
  for (let m = 5; m <= hMin; m += 5) xTicks.push(m)

  return (
    <div aria-label="ICP forecast chart" role="img">
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 8, right: 52, left: 0, bottom: 4 }}>

          <defs>
            <linearGradient id="forecastUncertainty" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor={foreColor} stopOpacity={0.10} />
              <stop offset="100%" stopColor={foreColor} stopOpacity={0.35} />
            </linearGradient>
          </defs>

          {/* Normal / Abnormal probability zones */}
          <ReferenceArea yAxisId="left" y1={0}      y2={thrPct} fill={isDark ? '#064E3B' : '#ECFDF5'} fillOpacity={0.18} />
          <ReferenceArea yAxisId="left" y1={thrPct} y2={100}    fill={isDark ? '#450A0A' : '#FEF2F2'} fillOpacity={0.18} />

          {/* Forecast region shade */}
          <ReferenceArea yAxisId="left" x1={0} x2={hMin} fill={isDark ? '#1E293B' : '#F8FAFC'} fillOpacity={0.50} />

          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />

          <XAxis
            dataKey="tMin"
            type="number"
            domain={[-5, hMin]}
            ticks={xTicks}
            tickFormatter={v => v === 0 ? 'Now' : v > 0 ? `+${v}m` : `${v}m`}
            tick={{ fontSize: 9, fill: tickColor }}
            tickLine={false}
            axisLine={{ stroke: gridColor }}
          />

          <YAxis
            yAxisId="left"
            domain={[0, 100]}
            ticks={[0, 20, 40, 60, 80, 100]}
            tickFormatter={v => `${v}%`}
            tick={{ fontSize: 9, fill: tickColor }}
            tickLine={false}
            axisLine={false}
            width={34}
            label={{ value: 'P(Abn)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 9, fill: tickColor }}
          />

          {/* Right Y-axis: Est. ICP mmHg */}
          <YAxis
            yAxisId="right"
            orientation="right"
            domain={[5, 40]}
            ticks={[5, 10, 15, 20, 25, 30, 35, 40]}
            tickFormatter={v => `${v}`}
            tick={{ fontSize: 9, fill: isDark ? '#94A3B8' : '#64748B' }}
            tickLine={false}
            axisLine={false}
            width={28}
            label={{ value: 'ICP mmHg', angle: 90, position: 'insideRight', offset: 10, fontSize: 9, fill: isDark ? '#94A3B8' : '#64748B' }}
          />

          {/* Now marker */}
          <ReferenceLine yAxisId="left" x={0}
            stroke={isDark ? '#475569' : '#9CA3AF'}
            strokeDasharray="4 3"
            label={{ value: 'Now', fontSize: 8, fill: tickColor, position: 'insideTopLeft' }}
          />

          {/* Horizon marker */}
          <ReferenceLine yAxisId="left" x={hMin}
            stroke={foreColor} strokeDasharray="3 2" strokeOpacity={0.65}
            label={{ value: `+${hMin}m`, fontSize: 8, fill: foreColor, position: 'insideTopRight' }}
          />

          {/* Decision threshold line (left axis) */}
          <ReferenceLine yAxisId="left" y={thrPct}
            stroke={isDark ? '#F59E0B' : '#D97706'}
            strokeDasharray="5 3"
            strokeOpacity={0.8}
            label={{ value: `${thrPct}%`, fontSize: 8, fill: isDark ? '#F59E0B' : '#D97706', position: 'insideLeft' }}
          />

          {/* ICP = 15 mmHg clinical threshold (right axis) */}
          <ReferenceLine yAxisId="right" y={15}
            stroke={isDark ? '#F59E0B' : '#D97706'}
            strokeDasharray="3 4"
            strokeOpacity={0.55}
            label={{ value: '15 mmHg', fontSize: 8, fill: isDark ? '#F59E0B' : '#D97706', position: 'insideRight' }}
          />

          <Tooltip content={<ForecastTooltip isDark={isDark} result={result} />} />

          {/* 95% CI band (left axis) */}
          <Area yAxisId="left" type="monotone" dataKey="ciUpper" stroke="none"
            fill="url(#forecastUncertainty)" isAnimationActive={false} connectNulls={false} legendType="none" />
          <Area yAxisId="left" type="monotone" dataKey="ciLower" stroke="none"
            fill={bgColor} fillOpacity={1} isAnimationActive={false} connectNulls={false} legendType="none" />

          {/* Historical P(Abnormal) — solid amber (left axis); hidden when flat */}
          {showHistLine && (
            <Line yAxisId="left" type="monotone" dataKey="histProb"
              stroke={histColor} strokeWidth={2.5} dot={false}
              isAnimationActive={false} connectNulls={false} name="XGBoost P(Abn)" />
          )}

          {/* LSTM Forecast P(Abnormal) — dashed, class-colored (left axis) */}
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="forecastProb"
            stroke={foreColor}
            strokeWidth={2.5}
            strokeDasharray="6 4"
            dot={(props: { cx: number; cy: number; payload: ChartPoint }) => {
              const show = props.payload.tMin % 5 === 0 || props.payload.tMin === hMin
              if (!show || props.payload.tMin <= 0) return <g key={`d-${props.payload.tMin}`} />
              return (
                <circle key={`h-${props.payload.tMin}`}
                  cx={props.cx} cy={props.cy} r={4}
                  fill={bgColor} stroke={foreColor} strokeWidth={2.5}
                />
              )
            }}
            isAnimationActive={false}
            connectNulls={false}
            name="LSTM Forecast P(Abn)"
          />

          {/* Historical Est. ICP mmHg — amber dashed (right axis); hidden when flat */}
          {showHistLine && (
            <Line yAxisId="right" type="monotone" dataKey="histICP"
              stroke={histColor} strokeWidth={1.5} strokeDasharray="3 3" dot={false} strokeOpacity={0.65}
              isAnimationActive={false} connectNulls={false} name="Hist. Est. ICP" />
          )}

          {/* Forecast Est. ICP mmHg — class-colored dotted (right axis) */}
          <Line yAxisId="right" type="monotone" dataKey="forecastICP"
            stroke={foreColor} strokeWidth={1.5} strokeDasharray="2 4" dot={false} strokeOpacity={0.65}
            isAnimationActive={false} connectNulls={false} name="Forecast Est. ICP" />

        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-0.5 mt-1 text-2xs">
        {showHistLine && (
          <>
            <span className="flex items-center gap-1.5">
              <span className="w-5 h-[2px] inline-block rounded" style={{ backgroundColor: histColor }} />
              <span style={{ color: histColor }}>Historical P(Abn) [left]</span>
            </span>
            <span className="text-clinical-text-muted dark:text-slate-600">·</span>
          </>
        )}
        <span className="flex items-center gap-1.5">
          <span className="w-5 h-[2px] inline-block rounded" style={{
            background: `repeating-linear-gradient(90deg,${foreColor} 0,${foreColor} 6px,transparent 6px,transparent 9px)`,
          }} />
          <span style={{ color: foreColor }}>LSTM Forecast P(Abn) [left]</span>
        </span>
        <span className="text-clinical-text-muted dark:text-slate-600">·</span>
        <span className="flex items-center gap-1.5">
          <span className="w-5 h-[2px] inline-block rounded opacity-65" style={{
            background: `repeating-linear-gradient(90deg,${histColor} 0,${histColor} 3px,transparent 3px,transparent 6px)`,
          }} />
          <span className="text-clinical-text-muted dark:text-slate-500">Est. ICP mmHg [right]</span>
        </span>
        <span className="text-clinical-text-muted dark:text-slate-600">·</span>
        <span className="text-clinical-text-muted dark:text-slate-500">
          Band = 95% CI &nbsp;·&nbsp; Threshold = {thrPct}%
        </span>
      </div>

      {/* Zone legend */}
      <div className="flex items-center justify-center gap-4 mt-0.5 text-2xs text-clinical-text-muted dark:text-slate-500">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm inline-block bg-emerald-200 dark:bg-emerald-900/40" />
          Normal (&lt;{thrPct}%)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm inline-block bg-red-200 dark:bg-red-900/40" />
          Abnormal (&ge;{thrPct}%)
        </span>
      </div>
    </div>
  )
}
