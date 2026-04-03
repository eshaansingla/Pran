import {
  LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine,
  ReferenceArea, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import type { TrendPoint } from '../types'
import { CLASS_COLORS } from '../utils/formatters'

interface Props {
  data: TrendPoint[]
}

function CustomDot(props: {
  cx?: number; cy?: number; payload?: TrendPoint
}) {
  const { cx, cy, payload } = props
  if (!payload || cx === undefined || cy === undefined) return null
  const color = CLASS_COLORS[payload.class]
  return <circle cx={cx} cy={cy} r={4} fill={color} stroke="#fff" strokeWidth={1.5} />
}

function CustomTooltip({ active, payload }: {
  active?: boolean
  payload?: Array<{ payload: TrendPoint }>
}) {
  if (!active || !payload?.[0]) return null
  const p = payload[0].payload
  const label = ['Normal', 'Elevated', 'Critical'][p.class]
  const color = CLASS_COLORS[p.class]
  return (
    <div className="bg-white border border-clinical-border rounded shadow-sm px-3 py-2 text-xs space-y-1">
      <p className="font-semibold" style={{ color }}>
        {label}
      </p>
      <p className="text-clinical-text-secondary">Window #{p.windowId}</p>
      <p className="text-clinical-text-muted">{p.timestamp}</p>
      <p className="tabular-nums text-clinical-text-primary">
        Confidence: {(p.confidence * 100).toFixed(1)}%
      </p>
    </div>
  )
}

const Y_TICKS = [0, 1, 2]
const Y_LABELS: Record<number, string> = { 0: 'Normal', 1: 'Elevated', 2: 'Critical' }

export default function TrendChart({ data }: Props) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-clinical-text-muted text-sm">
        No prediction data yet
      </div>
    )
  }

  // Show at most last 180 windows
  const visible = data.slice(-180)

  return (
    <div
      aria-label="ICP classification trend chart"
      role="img"
    >
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={visible} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
          {/* Background zones */}
          <ReferenceArea y1={-0.5} y2={0.5} fill="#ECFDF5" fillOpacity={0.6} />
          <ReferenceArea y1={0.5}  y2={1.5} fill="#FFFBEB" fillOpacity={0.6} />
          <ReferenceArea y1={1.5}  y2={2.5} fill="#FEF2F2" fillOpacity={0.6} />

          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />

          <XAxis
            dataKey="windowId"
            tick={{ fontSize: 10, fill: '#718096' }}
            label={{ value: 'Window', position: 'insideBottomRight', offset: -4, fontSize: 10, fill: '#718096' }}
            tickLine={false}
            axisLine={{ stroke: '#E2E8F0' }}
          />
          <YAxis
            domain={[-0.2, 2.2]}
            ticks={Y_TICKS}
            tickFormatter={v => Y_LABELS[v] ?? ''}
            tick={{ fontSize: 10, fill: '#718096' }}
            tickLine={false}
            axisLine={false}
            width={58}
          />

          {/* Boundary lines */}
          <ReferenceLine y={0.5} stroke="#059669" strokeDasharray="4 3" strokeOpacity={0.5} />
          <ReferenceLine y={1.5} stroke="#D97706" strokeDasharray="4 3" strokeOpacity={0.5} />

          <Tooltip content={<CustomTooltip />} />

          <Line
            type="stepAfter"
            dataKey="class"
            stroke="#2C5282"
            strokeWidth={2}
            dot={<CustomDot />}
            activeDot={{ r: 6, strokeWidth: 2, stroke: '#fff' }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
