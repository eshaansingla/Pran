import {
  ComposedChart, Line, XAxis, YAxis, Tooltip, ReferenceLine,
  ReferenceArea, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import type { TrendPoint } from '../types'
import { useStore } from '../store/useStore'
import { probToICP } from '../utils/formatters'

interface Props {
  data: TrendPoint[]
  threshold?: number   // model decision threshold (default 0.545 for XGBoost)
}

interface ChartPoint extends TrendPoint {
  icp: number  // estimated ICP in mmHg
}

function CustomDot(props: { cx?: number; cy?: number; payload?: ChartPoint; isDark?: boolean }) {
  const { cx, cy, payload, isDark } = props
  if (!payload || cx === undefined || cy === undefined) return null
  const color = payload.class === 0
    ? (isDark ? '#10B981' : '#059669')
    : (isDark ? '#EF4444' : '#DC2626')
  return <circle cx={cx} cy={cy} r={4} fill={color} stroke={isDark ? '#1A202C' : '#fff'} strokeWidth={1.5} />
}

function CustomTooltip({ active, payload, isDark }: {
  active?: boolean
  payload?: Array<{ payload: ChartPoint }>
  isDark?: boolean
}) {
  if (!active || !payload?.[0]) return null
  const p     = payload[0].payload
  const label = p.class === 0 ? 'Normal' : 'Abnormal'
  const color = p.class === 0 ? (isDark ? '#10B981' : '#059669') : (isDark ? '#EF4444' : '#DC2626')
  return (
    <div className={`border rounded shadow-sm px-3 py-2 text-xs space-y-1 ${
      isDark ? 'bg-slate-800 border-slate-600' : 'bg-white border-clinical-border'
    }`}>
      <p className="font-semibold" style={{ color }}>{label}</p>
      <p className={isDark ? 'text-slate-400' : 'text-clinical-text-secondary'}>Window #{p.windowId}</p>
      <p className={isDark ? 'text-slate-500' : 'text-clinical-text-muted'}>{p.timestamp}</p>
      <p className={`tabular-nums ${isDark ? 'text-slate-200' : 'text-clinical-text-primary'}`}>
        Confidence: {(p.confidence * 100).toFixed(1)}%
      </p>
      <p className={`tabular-nums font-semibold ${isDark ? 'text-amber-400' : 'text-amber-600'}`}>
        Est. ICP: ~{p.icp.toFixed(0)} mmHg
      </p>
    </div>
  )
}

const Y_TICKS  = [0, 1]
const Y_LABELS: Record<number, string> = { 0: 'Normal', 1: 'Abnormal' }

export default function TrendChart({ data, threshold = 0.545 }: Props) {
  const { isDark } = useStore()

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-clinical-text-muted dark:text-slate-400 text-sm">
        No prediction data yet
      </div>
    )
  }

  const visible: ChartPoint[] = data.slice(-180).map(pt => ({
    ...pt,
    // P(abnormal): confidence when class=1, else (1 − confidence)
    icp: probToICP(pt.class === 1 ? pt.confidence : 1 - pt.confidence, threshold),
  }))

  const tickColor  = isDark ? '#718096' : '#718096'
  const gridColor  = isDark ? '#2D3748' : '#E2E8F0'
  const normalFill = isDark ? '#064E3B' : '#ECFDF5'
  const abnFill    = isDark ? '#450A0A' : '#FEF2F2'
  const lineColor  = isDark ? '#3B82F6' : '#2C5282'
  const icpColor   = isDark ? '#F59E0B' : '#D97706'

  return (
    <div aria-label="ICP classification trend chart" role="img">
      <ResponsiveContainer width="100%" height={230}>
        <ComposedChart data={visible} margin={{ top: 8, right: 44, left: 0, bottom: 4 }}>
          <ReferenceArea yAxisId="class" y1={-0.2} y2={0.5} fill={normalFill} fillOpacity={0.6} />
          <ReferenceArea yAxisId="class" y1={0.5}  y2={1.2} fill={abnFill}    fillOpacity={0.6} />

          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />

          <XAxis
            dataKey="windowId"
            tick={{ fontSize: 10, fill: tickColor }}
            label={{ value: 'Window', position: 'insideBottomRight', offset: -4, fontSize: 10, fill: tickColor }}
            tickLine={false}
            axisLine={{ stroke: gridColor }}
          />

          {/* Left axis: classification */}
          <YAxis
            yAxisId="class"
            domain={[-0.2, 1.2]}
            ticks={Y_TICKS}
            tickFormatter={v => Y_LABELS[v] ?? ''}
            tick={{ fontSize: 10, fill: tickColor }}
            tickLine={false}
            axisLine={false}
            width={58}
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
            label={{ value: 'mmHg', angle: 90, position: 'insideRight', offset: 10, fontSize: 9, fill: icpColor }}
          />

          <ReferenceLine yAxisId="class" y={0.5}
            stroke={isDark ? '#EF4444' : '#DC2626'} strokeDasharray="4 3" strokeOpacity={0.5} />
          <ReferenceLine yAxisId="icp" y={15}
            stroke={isDark ? '#F59E0B' : '#D97706'} strokeDasharray="3 2" strokeOpacity={0.4}
            label={{ value: '15', fontSize: 8, fill: icpColor, position: 'right' }} />

          <Tooltip content={<CustomTooltip isDark={isDark} />} />

          {/* Classification step line */}
          <Line
            yAxisId="class"
            type="stepAfter"
            dataKey="class"
            stroke={lineColor}
            strokeWidth={2}
            dot={<CustomDot isDark={isDark} />}
            activeDot={{ r: 6, strokeWidth: 2, stroke: isDark ? '#1A202C' : '#fff' }}
            isAnimationActive={false}
            name="Classification"
          />

          {/* Estimated ICP line */}
          <Line
            yAxisId="icp"
            dataKey="icp"
            stroke={icpColor}
            strokeWidth={1.5}
            strokeDasharray="4 2"
            dot={false}
            isAnimationActive={false}
            name="Est. ICP (mmHg)"
          />

          <Legend
            wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
            iconType="line"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
