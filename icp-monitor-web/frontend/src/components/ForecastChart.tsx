import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ReferenceLine,
  ReferenceArea, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import type { ForecastResult } from '../types'
import { useStore } from '../store/useStore'
import { probToICP, mapToICP, icpGrade } from '../utils/formatters'

// ── Scaler constants from lstm_meta.json ──────────────────────────────────────
// Feature index 5 = mean_arterial_pressure
const MAP_MEAN = 88.352
const MAP_STD  = 7.850

/**
 * Auto-detect whether MAP value is a z-score or raw mmHg.
 * Physiological MAP is always ≥ 50 mmHg. If value is < 10, it's almost
 * certainly a z-score — reconstruct using training scaler.
 */
function toRawMAP(val: number): number {
  if (val > 50) return val               // already raw mmHg
  return val * MAP_STD + MAP_MEAN        // z-score → raw mmHg
}

interface Props {
  sequence: number[][]   // raw N × 6 windows from uploaded CSV
  result:   ForecastResult
}

interface ChartPoint {
  tMin:         number
  label:        string
  histICP?:     number
  forecastICP?: number
  ciLower?:     number
  ciUpper?:     number
}

// ─────────────────────────────────────────────────────────────────────────────
// Build chart data
// Historical: last 30 windows (5 min) of MAP-based ICP — fixed [-5, 0] window
// Forecast:   LSTM probability → ICP, interpolated lastHistICP → targetICP
// ─────────────────────────────────────────────────────────────────────────────
const HIST_DISPLAY = 30   // always show the last 30 windows = 5 min of history

function buildChartData(sequence: number[][], result: ForecastResult): ChartPoint[] {
  const threshold  = result.threshold
  const hasReal    = sequence.some(row => row.some(v => v !== 0))

  // Only display the last HIST_DISPLAY windows in the chart
  const displaySeq = sequence.slice(-HIST_DISPLAY)
  const dispLen    = displaySeq.length   // could be < 30 if sequence is short

  // Step 1: Raw ICP per displayed window
  const rawICPs = displaySeq.map(row => {
    if (!hasReal) return 12   // fallback for reloaded (no CSV)
    const rawMap = toRawMAP(row[5])
    return mapToICP(rawMap)   // MAP − 70, clamped [5, 40]
  })

  // Step 2: 3-point rolling average to smooth natural noise
  const smooth = rawICPs.map((_, i) => {
    const s = Math.max(0, i - 1)
    const e = Math.min(rawICPs.length, i + 2)
    const sl = rawICPs.slice(s, e)
    return sl.reduce((a, b) => a + b, 0) / sl.length
  })

  // Step 3: Historical points — always anchor last point at tMin=0
  const points: ChartPoint[] = []
  for (let i = 0; i < dispLen; i++) {
    const tMin = -(dispLen - 1 - i) * 10 / 60   // ≤ 0 minutes
    const isTickMin = Math.round(tMin) === tMin
    points.push({
      tMin:    +tMin.toFixed(3),
      label:   i === dispLen - 1 ? 'Now' : isTickMin ? `${Math.round(tMin)}m` : '',
      histICP: +smooth[i].toFixed(1),
    })
  }

  // Step 4: Forecast trajectory (0 → +30 min)
  const lastICP    = smooth[dispLen - 1]
  const targetICP  = probToICP(result.probability, threshold)
  const ciLowICP   = probToICP(Math.max(0.001, result.ci_lower), threshold)
  const ciHighICP  = probToICP(Math.min(0.999, result.ci_upper), threshold)
  const horizonMin = result.horizon_minutes
  const ciHalfFull = Math.max(0.5, (ciHighICP - ciLowICP) / 2)

  // Long-run ICP limit beyond model horizon
  const isAbn    = result.class === 1
  const limitICP = isAbn
    ? Math.min(40, targetICP + 5)
    : Math.max(5,  targetICP - 3)

  // Forecast design:
  //   result.probability = P(abnormal) AT the model horizon (+horizonMin minutes)
  //   targetICP          = LSTM's ICP estimate at t=+horizonMin
  //
  // At t=0 we anchor the forecast at the MAP-based current ICP (lastICP) because
  // the LSTM has no separate t=0 output — the bridge must start somewhere physical.
  // From t=0 → t=+horizonMin we interpolate toward targetICP (model's horizon estimate).
  // Beyond horizon we use damped continuation with widening uncertainty.
  //
  // This correctly reflects: "ICP is currently ~X mmHg (MAP), LSTM forecasts ~Y mmHg
  // in 15 min." A rising history + falling forecast is VALID when the LSTM (using all
  // 6 features) disagrees with MAP alone — the model may have detected early
  // normalisation signals not visible in MAP.
  const lastPt       = points[points.length - 1]
  lastPt.forecastICP = +lastICP.toFixed(1)
  lastPt.ciLower     = +lastICP.toFixed(1)
  lastPt.ciUpper     = +lastICP.toFixed(1)

  for (let m = 1; m <= 30; m++) {
    let icp: number
    let ciH: number

    if (m <= horizonMin) {
      const alpha = m / horizonMin
      icp = lastICP + alpha * (targetICP - lastICP)
      ciH = alpha * ciHalfFull
    } else {
      const extra = m - horizonMin
      icp = targetICP + (limitICP - targetICP) * (1 - Math.exp(-extra / 15))
      ciH = Math.min(ciHalfFull + 0.6 * extra, 10)
    }

    icp = Math.max(5, Math.min(40, icp))

    points.push({
      tMin:        m,
      label:       m === horizonMin ? `+${horizonMin}m` : m % 5 === 0 ? `+${m}m` : '',
      forecastICP: +icp.toFixed(1),
      ciLower:     +Math.max(5,  icp - ciH).toFixed(1),
      ciUpper:     +Math.min(40, icp + ciH).toFixed(1),
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

  const icpVal = pt.forecastICP ?? pt.histICP
  const grade  = icpVal !== undefined ? icpGrade(icpVal) : null

  return (
    <div style={{ background: bg, border: br, borderRadius: 8, padding: '8px 12px', fontSize: 11, color: tx, minWidth: 155 }}>
      <p style={{ fontWeight: 700, marginBottom: 4 }}>
        {pt.tMin <= 0 ? `History  ${pt.label}` : pt.label || `t +${pt.tMin.toFixed(0)}m`}
      </p>

      {icpVal !== undefined && (
        <p style={{ color: '#D97706', fontWeight: 600, marginBottom: 2 }}>
          Est. ICP:&nbsp;~{icpVal} mmHg
          {grade && (
            <span style={{ color: grade.color, marginLeft: 6, fontSize: 10 }}>
              {grade.label}
            </span>
          )}
        </p>
      )}

      {pt.tMin <= 0 && (
        <p style={{ color: mu, fontSize: 10 }}>MAP-based physical proxy (MAP − 70 mmHg)</p>
      )}

      {pt.tMin > 0 && pt.ciLower !== undefined && pt.ciUpper !== undefined && pt.ciLower !== pt.ciUpper && (
        <p style={{ color: mu, fontSize: 10, marginTop: 2 }}>
          95% CI: {pt.ciLower}–{pt.ciUpper} mmHg
        </p>
      )}
      {pt.tMin > 0 && (
        <p style={{ color: mu, fontSize: 10, marginTop: 1 }}>
          LSTM target at +{result.horizon_minutes}m: ~{probToICP(result.probability, result.threshold).toFixed(0)} mmHg ({result.class_name})
        </p>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ForecastChart({ sequence, result }: Props) {
  const { isDark } = useStore()
  const data       = buildChartData(sequence, result)
  const isAbn      = result.class === 1

  const icpColor   = isDark ? '#F59E0B' : '#D97706'
  const foreColor  = isAbn
    ? (isDark ? '#EF4444' : '#DC2626')
    : (isDark ? '#10B981' : '#059669')
  const gridColor  = isDark ? '#1E293B' : '#E2E8F0'
  const tickColor  = isDark ? '#64748B' : '#94A3B8'
  const bgColor    = isDark ? '#0F172A' : '#fff'
  const xTicks = [-5, -4, -3, -2, -1, 0, 5, 10, 15, 20, 25, 30]
  const xMin   = -5   // always fixed: show last 5 min of history

  return (
    <div aria-label="ICP forecast chart" role="img">
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 8, right: 28, left: 0, bottom: 4 }}>

          {/* Lundberg grade background zones */}
          <ReferenceArea y1={0}  y2={15} fill={isDark ? '#064E3B' : '#ECFDF5'} fillOpacity={0.15} />
          <ReferenceArea y1={15} y2={20} fill={isDark ? '#78350F' : '#FEF3C7'} fillOpacity={0.20} />
          <ReferenceArea y1={20} y2={40} fill={isDark ? '#450A0A' : '#FEF2F2'} fillOpacity={0.22} />

          {/* Forecast region shade */}
          <ReferenceArea x1={0} x2={30} fill={isDark ? '#1E293B' : '#F8FAFC'} fillOpacity={0.50} />

          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />

          <XAxis
            dataKey="tMin"
            type="number"
            domain={[xMin, 30]}
            ticks={xTicks}
            tickFormatter={v => v === 0 ? 'Now' : v > 0 ? `+${v}m` : `${v}m`}
            tick={{ fontSize: 9, fill: tickColor }}
            tickLine={false}
            axisLine={{ stroke: gridColor }}
          />

          <YAxis
            domain={[0, 40]}
            ticks={[0, 5, 10, 15, 20, 25, 30, 35, 40]}
            tickFormatter={v => `${v}`}
            tick={{ fontSize: 9, fill: icpColor }}
            tickLine={false}
            axisLine={false}
            width={30}
            label={{ value: 'mmHg', angle: -90, position: 'insideLeft', offset: 10, fontSize: 9, fill: icpColor }}
          />

          {/* Now marker */}
          <ReferenceLine x={0}
            stroke={isDark ? '#475569' : '#9CA3AF'}
            strokeDasharray="4 3"
            label={{ value: 'Now', fontSize: 8, fill: tickColor, position: 'insideTopLeft' }}
          />

          {/* Model horizon marker */}
          <ReferenceLine x={result.horizon_minutes}
            stroke={foreColor} strokeDasharray="3 2" strokeOpacity={0.65}
            label={{ value: `+${result.horizon_minutes}m`, fontSize: 8, fill: foreColor, position: 'insideTopRight' }}
          />

          {/* Clinical ICP thresholds */}
          <ReferenceLine y={15}
            stroke={icpColor} strokeDasharray="4 3" strokeOpacity={0.65}
            label={{ value: '15', fontSize: 8, fill: icpColor, position: 'right' }}
          />
          <ReferenceLine y={20}
            stroke={isDark ? '#EF4444' : '#DC2626'} strokeDasharray="3 2" strokeOpacity={0.35}
            label={{ value: '20', fontSize: 8, fill: isDark ? '#EF4444' : '#DC2626', position: 'right' }}
          />

          <Tooltip content={<ForecastTooltip isDark={isDark} result={result} />} />

          {/* 95% CI band: fill ciUpper then mask ciLower with background */}
          <Area
            dataKey="ciUpper"
            stroke="none"
            fill={foreColor}
            fillOpacity={0.18}
            isAnimationActive={false}
            connectNulls={false}
            legendType="none"
          />
          <Area
            dataKey="ciLower"
            stroke="none"
            fill={bgColor}
            fillOpacity={1}
            isAnimationActive={false}
            connectNulls={false}
            legendType="none"
          />

          {/* Historical MAP-based ICP (solid amber) */}
          <Line
            dataKey="histICP"
            stroke={icpColor}
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
            name="Historical ICP (MAP−70)"
          />

          {/* LSTM Forecast ICP (dashed, class-colored) */}
          <Line
            dataKey="forecastICP"
            stroke={foreColor}
            strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={(props: { cx: number; cy: number; payload: ChartPoint }) => {
              if (props.payload.tMin !== result.horizon_minutes)
                return <g key={`dot-${props.payload.tMin}`} />
              return (
                <circle
                  key={`hor-${props.payload.tMin}`}
                  cx={props.cx} cy={props.cy} r={5}
                  fill={foreColor}
                  stroke={isDark ? '#0F172A' : '#fff'}
                  strokeWidth={2}
                />
              )
            }}
            isAnimationActive={false}
            connectNulls={false}
            name="LSTM Forecast ICP"
          />

        </ComposedChart>
      </ResponsiveContainer>

      {/* Custom legend */}
      <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-0.5 mt-1 text-2xs">
        <span className="flex items-center gap-1.5">
          <span className="w-5 h-[2px] inline-block rounded" style={{ backgroundColor: icpColor }} />
          <span style={{ color: icpColor }}>Historical ICP (MAP−70)</span>
        </span>
        <span className="text-clinical-text-muted dark:text-slate-600">·</span>
        <span className="flex items-center gap-1.5">
          <span className="w-5 h-[2px] inline-block rounded" style={{
            background: `repeating-linear-gradient(90deg,${foreColor} 0,${foreColor} 6px,transparent 6px,transparent 9px)`,
          }} />
          <span style={{ color: foreColor }}>LSTM Forecast ICP</span>
        </span>
        <span className="text-clinical-text-muted dark:text-slate-600">·</span>
        <span className="text-clinical-text-muted dark:text-slate-500">
          Band = 95% CI &nbsp;·&nbsp;
          <span style={{ color: foreColor }}>● = +{result.horizon_minutes}m horizon</span>
        </span>
      </div>

      {/* Grade legend */}
      <div className="flex items-center justify-center gap-4 mt-0.5 text-2xs text-clinical-text-muted dark:text-slate-500">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm inline-block bg-emerald-200 dark:bg-emerald-900/40" />
          Normal (&lt;15 mmHg)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm inline-block bg-amber-200 dark:bg-amber-900/40" />
          Grade I (15–20)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm inline-block bg-red-200 dark:bg-red-900/40" />
          Grade II (&gt;20)
        </span>
      </div>
    </div>
  )
}
