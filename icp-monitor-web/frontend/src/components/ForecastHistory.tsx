import { useState } from 'react'
import { Clock, Trash2, RotateCcw, History, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { useStore, type StoredForecast } from '../store/useStore'
import type { ForecastResult } from '../types'
import { probToICP } from '../utils/formatters'

interface Props {
  onLoad: (result: ForecastResult, fileName: string, seqLen: number) => void
}

function fmtDate(iso: string) {
  return new Date(iso).toLocaleString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

function ForecastRow({ f, onLoad, onDelete }: {
  f: StoredForecast
  onLoad: () => void
  onDelete: () => void
}) {
  const isAbn = f.class === 1
  const estICP = probToICP(f.probability, f.result.threshold)

  return (
    <div className="flex items-center gap-3 px-4 py-2.5 hover:bg-slate-50 dark:hover:bg-slate-700/40 transition-colors">
      {isAbn
        ? <AlertTriangle size={13} className="text-red-500 dark:text-red-400 flex-shrink-0" />
        : <CheckCircle2  size={13} className="text-emerald-500 dark:text-emerald-400 flex-shrink-0" />
      }

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-xs font-medium text-clinical-text-primary dark:text-slate-200 truncate">
            {fmtDate(f.date)}
          </p>
          <span className={`text-2xs font-semibold px-1.5 py-0.5 rounded ${
            isAbn
              ? 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-400'
              : 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-400'
          }`}>
            {isAbn ? 'Abnormal' : 'Normal'}
          </span>
        </div>
        <p className="text-2xs text-clinical-text-muted dark:text-slate-400 truncate">
          {f.fileName} · {f.seqLen} windows ({f.durationMin} min) ·{' '}
          <span className="tabular-nums font-mono">{(f.probability * 100).toFixed(1)}%</span>
          {' '}· ~<span className="tabular-nums font-mono">{estICP.toFixed(0)} mmHg</span>
          {' '}· +{f.horizon_minutes} min ahead · {f.confidence_label} confidence
        </p>
      </div>

      <div className="flex items-center gap-1.5 flex-shrink-0">
        <button
          onClick={onLoad}
          aria-label="Reload forecast"
          title="Reload forecast"
          className="p-1 rounded text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-colors"
        >
          <RotateCcw size={13} />
        </button>
        <button
          onClick={onDelete}
          aria-label="Delete forecast"
          title="Delete forecast"
          className="p-1 rounded text-clinical-text-muted dark:text-slate-500 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
        >
          <Trash2 size={13} />
        </button>
      </div>
    </div>
  )
}

export default function ForecastHistory({ onLoad }: Props) {
  const { forecasts, removeForecast } = useStore()
  const [open, setOpen] = useState(false)

  if (forecasts.length === 0) return null

  const abnCount = forecasts.filter(f => f.class === 1).length

  return (
    <div className="bg-white dark:bg-slate-800 border border-clinical-border dark:border-slate-600 rounded-lg shadow-sm overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <History size={14} className="text-purple-600 dark:text-purple-400" />
          <span className="text-xs font-semibold text-clinical-text-secondary dark:text-slate-300 uppercase tracking-wide">
            Forecast History
          </span>
          <span className="text-2xs bg-slate-100 dark:bg-slate-700 text-clinical-text-muted dark:text-slate-400 px-1.5 py-0.5 rounded font-mono">
            {forecasts.length}
          </span>
          {abnCount > 0 && (
            <span className="text-2xs bg-red-100 dark:bg-red-900/40 text-red-600 dark:text-red-400 px-1.5 py-0.5 rounded font-semibold">
              {abnCount} abnormal
            </span>
          )}
        </div>
        <span className="text-2xs text-clinical-text-muted dark:text-slate-500">
          {open ? 'hide' : 'show'}
        </span>
      </button>

      {open && (
        <div className="divide-y divide-clinical-border dark:divide-slate-700 max-h-72 overflow-y-auto">
          {forecasts.map(f => (
            <ForecastRow
              key={f.id}
              f={f}
              onLoad={() => onLoad(f.result, f.fileName, f.seqLen)}
              onDelete={() => removeForecast(f.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
