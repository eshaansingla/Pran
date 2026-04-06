import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ReferenceLine,
  ReferenceArea, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import type { ForecastResult } from '../types'
import { useStore } from '../store/useStore'
import { probToICP } from '../utils/formatters'

interface Props {
  sequence:   number[][]   // raw feature rows (30 × 6)
  result:     ForecastResult
}

interface ChartPoint {
  t: number       // relative time in seconds (-290 … 0 = history, +900 = forecast)
  label: string
  histProb?: number
  forecastProb?: number
  ciBand?: [number, number]
  ciLower?: number
  ciUpper?: number
  histICP?: number      // estimated ICP mmHg from histProb
  forecastICP?: number  // estimated ICP mmHg from forecastProb
  ciLowerICP?: number
  ciUpperICP?: number
}

// Approximate per-window abnormal probability from raw features using
// a simple heuristic (MAP z-score).  This gives a rough historical trace
// without calling the XGBoost API on every window.
function estimateHistorical(sequence: number[][]): number[] {
  if (sequence.length === 0) return []
  const n = sequence.length
  const f = 6

  // Compute per-feature mean and std over the sequence
  const mean = Array(f).fill(0) as number[]
  const std  = Array(f).fill(1) as number[]
  for (let j = 0; j < f; j++) {
    const vals = sequence.map(row => row[j])
    mean[j] = vals.reduce((s, v) => s + v, 0) / n
    const variance = vals.reduce((s, v) => s + (v - mean[j]) ** 2, 0) / n
    std[j]  = Math.sqrt(variance) || 1
  }

  // Soft sigmoid applied to mean absolute z-score — gives 0–1 "abnormality proxy"
  return sequence.map(row => {
    const z = row.reduce((s, v, j) => s + Math.abs((v - mean[j]) / std[j]), 0) / f
    return 1 / (1 + Math.exp(-2.5 * (z - 1.5)))
  })
}

function buildChartData(sequence: number[][], result: ForecastResult): ChartPoint[] {
  const histProbs = estimateHistorical(sequence)
  const seqLen    = sequence.length
  const points: ChartPoint[] = []

  // Historical: t = -(seqLen-1)*10 … 0 seconds
  histProbs.forEach((p, i) => {
    const tSec = -(seqLen - 1 - i) * 10
    points.push({
      t:        tSec,
      label:    tSec === 0 ? 'Now' : `${tSec}s`,
      histProb: +p.toFixed(3),
      histICP:  +probToICP(p).toFixed(1),
    })
  })

  // Bridge: at t=0 also start the forecast line (0 CI width = no uncertainty yet)
  const lastHistProb = histProbs[seqLen - 1]
  const fSec = result.horizon_minutes * 60
  const bridgePt = points[points.length - 1]
  bridgePt.forecastProb = +lastHistProb.toFixed(3)
  bridgePt.forecastICP  = +probToICP(lastHistProb).toFixed(1)
  bridgePt.ciLower      = +lastHistProb.toFixed(3)
  bridgePt.ciUpper      = +lastHistProb.toFixed(3)
  bridgePt.ciLowerICP   = +probToICP(lastHistProb).toFixed(1)
  bridgePt.ciUpperICP   = +probToICP(lastHistProb).toFixed(1)

  // Interpolated forecast points: t = fSec/N … fSec
  // CI band widens linearly from 0 at t=0 to full width at t=fSec
  const N = 9
  const targetProb = result.probability
  const fullHalf   = (result.ci_upper - result.ci_lower) / 2
  for (let i = 1; i <= N; i++) {
    const alpha = i / N
    const tSec  = Math.round(alpha * fSec)
    const prob  = lastHistProb + alpha * (targetProb - lastHistProb)
    const half  = alpha * fullHalf
    const cLo   = +Math.max(0, prob - half).toFixed(3)
    const cHi   = +Math.min(1, prob + half).toFixed(3)
    points.push({
      t:            tSec,
      label:        i === N ? `+${result.horizon_minutes} min` : `+${(tSec / 60).toFixed(1)}m`,
      forecastProb: +prob.toFixed(3),
      forecastICP:  +probToICP(prob).toFixed(1),
      ciLower:      cLo,
      ciUpper:      cHi,
      ciLowerICP:   +probToICP(cLo).toFixed(1),
      ciUpperICP:   +probToICP(cHi).toFixed(1),
    })
  }

  return points
}

function ForecastTooltip({ active, payload, isDark }: {
  active?: boolean
  payload?: Array<{ name: string; value: number; payload: ChartPoint }>
  isDark?: boolean
}) {
  if (!active || !payload?.length) return null
  const pt = payload[0].payload
  const bg = isDark ? '#2D3748' : '#fff'
  const br = isDark ? '1px solid #4A5568' : '1px solid #E2E8F0'
  const tx = isDark ? '#E2E8F0' : '#1A202C'
  const mu = isDark ? '#A0AEC0' : '#718096'

  return (
    <div style={{ background: bg, border: br, borderRadius: 6, padding: '8px 12px', fontSize: 11, color: tx }}>
      <p style={{ fontWeight: 600, marginBottom: 4 }}>{pt.label}</p>
      {pt.histProb !== undefined && (
        <>
          <p style={{ color: mu }}>Activity proxy: {(pt.histProb * 100).toFixed(1)}%</p>
          {pt.histICP !== undefined && (
            <p style={{ color: '#D97706', fontWeight: 600 }}>Est. ICP: ~{pt.histICP.toFixed(0)} mmHg</p>
          )}
        </>
      )}
      {pt.forecastProb !== undefined && pt.histProb === undefined && (
        <>
          <p style={{ color: pt.forecastProb >= 0.5 ? '#DC2626' : '#059669', fontWeight: 600 }}>
            Projected: {(pt.forecastProb * 100).toFixed(1)}%
          </p>
          {pt.forecastICP !== undefined && (
            <p style={{ color: '#D97706', fontWeight: 600 }}>Est. ICP: ~{pt.forecastICP.toFixed(0)} mmHg</p>
          )}
          {pt.ciLower !== undefined && pt.ciUpper !== undefined && pt.ciLower !== pt.ciUpper && (
            <p style={{ color: mu }}>
              95% CI: [{(pt.ciLower * 100).toFixed(1)}%, {(pt.ciUpper * 100).toFixed(1)}%]
              &nbsp;(~{pt.ciLowerICP?.toFixed(0)}–{pt.ciUpperICP?.toFixed(0)} mmHg)
            </p>
          )}
        </>
      )}
    </div>
  )
}

export default function ForecastChart({ sequence, result }: Props) {
  const { isDark }  = useStore()
  const data        = buildChartData(sequence, result)

  const tickColor   = isDark ? '#718096' : '#718096'
  const gridColor   = isDark ? '#2D3748' : '#E2E8F0'
  const normalFill  = isDark ? '#064E3B20' : '#ECFDF540'
  const abnFill     = isDark ? '#450A0A20' : '#FEF2F240'
  const histColor   = isDark ? '#60A5FA'   : '#2C5282'
  const icpColor    = isDark ? '#F59E0B'   : '#D97706'
  const foreColor   = result.class === 1
    ? (isDark ? '#EF4444' : '#DC2626')
    : (isDark ? '#10B981' : '#059669')

  // X-axis ticks: every 60s for history, then the forecast tick
  const xTicks = data
    .filter(d => d.t % 60 === 0 || d.forecastProb !== undefined)
    .map(d => d.t)

  return (
    <div aria-label="ICP forecast chart" role="img">
      <ResponsiveContainer width="100%" height={250}>
        <ComposedChart data={data} margin={{ top: 8, right: 46, left: 0, bottom: 4 }}>
          {/* Background zones */}
          <ReferenceArea yAxisId="prob" y1={0}   y2={0.5} fill={normalFill} fillOpacity={1} />
          <ReferenceArea yAxisId="prob" y1={0.5} y2={1.0} fill={abnFill}    fillOpacity={1} />

          {/* Forecast window shade */}
          <ReferenceArea yAxisId="prob" x1={0} x2={result.horizon_minutes * 60}
            fill={isDark ? '#2D3748' : '#F7FAFC'} fillOpacity={0.6} />

          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />

          <XAxis
            dataKey="t"
            type="number"
            domain={[data[0].t, data[data.length - 1].t]}
            ticks={xTicks}
            tickFormatter={v => {
              if (v === 0) return 'Now'
              if (v > 0)   return `+${v / 60}m`
              return `${v}s`
            }}
            tick={{ fontSize: 9, fill: tickColor }}
            tickLine={false}
            axisLine={{ stroke: gridColor }}
          />
          <YAxis
            yAxisId="prob"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 9, fill: tickColor }}
            tickLine={false}
            axisLine={false}
            width={36}
          />

          {/* Right axis: estimated ICP mmHg */}
          <YAxis
            yAxisId="icp"
            orientation="right"
            domain={[0, 40]}
            ticks={[0, 5, 10, 15, 20, 25, 30, 35, 40]}
            tickFormatter={v => `${v}`}
            tick={{ fontSize: 9, fill: icpColor }}
            tickLine={false}
            axisLine={false}
            width={36}
            label={{ value: 'mmHg', angle: 90, position: 'insideRight', offset: 12, fontSize: 9, fill: icpColor }}
          />

          <ReferenceLine x={0} yAxisId="prob"
            stroke={isDark ? '#718096' : '#6B7280'}
            strokeDasharray="4 3"
            label={{ value: 'Now', fontSize: 9, fill: tickColor, position: 'insideTopLeft' }} />
          <ReferenceLine yAxisId="prob" y={0.5}
            stroke={isDark ? '#EF4444' : '#DC2626'}
            strokeDasharray="4 3" strokeOpacity={0.5} />
          <ReferenceLine yAxisId="icp" y={15}
            stroke={isDark ? '#F59E0B' : '#D97706'}
            strokeDasharray="3 2" strokeOpacity={0.35}
            label={{ value: '15mmHg', fontSize: 8, fill: icpColor, position: 'right' }} />

          <Tooltip content={<ForecastTooltip isDark={isDark} />} />

          {/* Historical probability trace */}
          <Line
            yAxisId="prob"
            dataKey="histProb"
            stroke={histColor}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
            name="Historical prob."
          />

          {/* Historical ICP trace */}
          <Line
            yAxisId="icp"
            dataKey="histICP"
            stroke={icpColor}
            strokeWidth={1.5}
            strokeDasharray="4 2"
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
            name="Est. ICP (mmHg)"
          />

          {/* CI area — rendered only where forecastProb exists */}
          <Area
            yAxisId="prob"
            dataKey="ciUpper"
            stroke="none"
            fill={foreColor}
            fillOpacity={0.15}
            isAnimationActive={false}
            connectNulls={false}
            legendType="none"
          />
          <Area
            yAxisId="prob"
            dataKey="ciLower"
            stroke="none"
            fill={isDark ? '#1A202C' : '#fff'}
            fillOpacity={1}
            isAnimationActive={false}
            connectNulls={false}
            legendType="none"
          />

          {/* Forecast probability line — continuous from Now to +horizon */}
          <Line
            yAxisId="prob"
            dataKey="forecastProb"
            stroke={foreColor}
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={(props: { cx: number; cy: number; payload: ChartPoint }) => {
              if (props.payload.t !== result.horizon_minutes * 60) return <g key={props.payload.t} />
              return (
                <circle
                  key={props.payload.t}
                  cx={props.cx}
                  cy={props.cy}
                  r={7}
                  fill={foreColor}
                  stroke={isDark ? '#1A202C' : '#fff'}
                  strokeWidth={2}
                />
              )
            }}
            isAnimationActive={false}
            connectNulls={false}
            name="Forecast prob."
          />

          {/* Forecast ICP trace */}
          <Line
            yAxisId="icp"
            dataKey="forecastICP"
            stroke={icpColor}
            strokeWidth={1.5}
            strokeDasharray="5 3"
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
            legendType="none"
          />

          <Legend
            wrapperStyle={{ fontSize: 11, paddingTop: 6 }}
            iconType="line"
          />
        </ComposedChart>
      </ResponsiveContainer>

      <p className="text-2xs text-clinical-text-muted dark:text-slate-500 mt-1 text-center">
        Solid = probability · <span style={{ color: '#D97706' }}>amber dashed = estimated ICP (mmHg)</span> ·
        forecast dashed to +{result.horizon_minutes} min with widening 95% CI.
        ICP estimate uses logistic transform anchored at 15 mmHg threshold.
      </p>
    </div>
  )
}
