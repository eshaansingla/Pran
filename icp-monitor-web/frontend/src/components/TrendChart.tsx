import {
  ComposedChart, Line, XAxis, YAxis, Tooltip, ReferenceLine,
  ReferenceArea, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import type { TrendPoint } from '../types'
import { useStore } from '../store/useStore'

interface Props {
  data: TrendPoint[]
  threshold?: number   // model decision threshold (default 0.545 for XGBoost)
}

type ChartPoint = TrendPoint & { probPct: number }

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
        P(Abnormal): {p.probPct.toFixed(1)}%
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
    probPct: +(Math.max(0, Math.min(100, (pt.class === 1 ? pt.confidence : 1 - pt.confidence) * 100)).toFixed(1)),
  }))

  const thrPct     = +(threshold * 100).toFixed(1)
  const tickColor  = isDark ? '#718096' : '#718096'
  const gridColor  = isDark ? '#2D3748' : '#E2E8F0'
  const normalFill = isDark ? '#064E3B' : '#ECFDF5'
  const abnFill    = isDark ? '#450A0A' : '#FEF2F2'
  const lineColor  = isDark ? '#3B82F6' : '#2C5282'
  const thrColor   = isDark ? '#F59E0B' : '#D97706'

  return (
    <div aria-label="ICP classification trend chart" role="img">
      <ResponsiveContainer width="100%" height={230}>
        <ComposedChart data={visible} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <ReferenceArea y1={0}       y2={thrPct} fill={normalFill} fillOpacity={0.5} />
          <ReferenceArea y1={thrPct}  y2={100}    fill={abnFill}    fillOpacity={0.5} />

          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />

          <XAxis
            dataKey="windowId"
            tick={{ fontSize: 10, fill: tickColor }}
            label={{ value: 'Window', position: 'insideBottomRight', offset: -4, fontSize: 10, fill: tickColor }}
            tickLine={false}
            axisLine={{ stroke: gridColor }}
          />

          <YAxis
            domain={[0, 100]}
            ticks={[0, 20, 40, 60, 80, 100]}
            tickFormatter={v => `${v}%`}
            tick={{ fontSize: 9, fill: tickColor }}
            tickLine={false}
            axisLine={false}
            width={38}
            label={{ value: 'P(Abn)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 9, fill: tickColor }}
          />

          <ReferenceLine y={thrPct}
            stroke={thrColor} strokeDasharray="4 3" strokeOpacity={0.8}
            label={{ value: `${thrPct}%`, fontSize: 8, fill: thrColor, position: 'right' }} />

          <Tooltip content={<CustomTooltip isDark={isDark} />} />

          <Line
            type="monotone"
            dataKey="probPct"
            stroke={lineColor}
            strokeWidth={2}
            dot={<CustomDot isDark={isDark} />}
            activeDot={{ r: 6, strokeWidth: 2, stroke: isDark ? '#1A202C' : '#fff' }}
            isAnimationActive={false}
            name="P(Abnormal) %"
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
