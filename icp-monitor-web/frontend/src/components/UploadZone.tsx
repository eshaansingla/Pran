import React, { useCallback, useRef, useState } from 'react'
import { UploadCloud, X, AlertCircle, Download, FileText, CheckCircle2 } from 'lucide-react'
import { fetchExampleCsv } from '../utils/api'
import { downloadBlob } from '../utils/formatters'

interface Props {
  onFile: (file: File) => void
  loading: boolean
  errors: string[]
  onClear: () => void
  hasData: boolean
}

export default function UploadZone({ onFile, loading, errors, onClear, hasData }: Props) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((f: File) => {
    if (!f.name.toLowerCase().endsWith('.csv')) {
      alert('Only .csv files are accepted.')
      return
    }
    if (f.size > 10 * 1024 * 1024) {
      alert('File exceeds 10 MB limit.')
      return
    }
    onFile(f)
  }, [onFile])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [handleFile])

  const downloadExample = async () => {
    const csv = await fetchExampleCsv()
    downloadBlob(new Blob([csv], { type: 'text/csv' }), 'icp_example.csv')
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="section-heading">Data Upload</h2>
        <button
          onClick={downloadExample}
          className="flex items-center gap-1.5 text-xs text-clinical-primary dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
        >
          <Download size={12} />
          Example CSV
        </button>
      </div>

      {!hasData ? (
        <div
          role="button"
          tabIndex={0}
          id="file-input-trigger"
          aria-label="Upload zone — drag and drop or click to select CSV"
          onClick={() => !loading && inputRef.current?.click()}
          onKeyDown={e => e.key === 'Enter' && !loading && inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`
            relative border-2 border-dashed rounded-xl px-6 py-9
            flex flex-col items-center justify-center gap-3
            transition-all duration-200 select-none
            ${loading
              ? 'cursor-wait opacity-75 border-clinical-primary dark:border-blue-500 bg-blue-50/50 dark:bg-blue-900/10'
              : dragging
                ? 'cursor-copy border-clinical-primary dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20 scale-[1.01]'
                : 'cursor-pointer border-clinical-border dark:border-slate-600 bg-white dark:bg-slate-800 hover:border-clinical-primary dark:hover:border-blue-400 hover:bg-blue-50/40 dark:hover:bg-blue-900/10'}
          `}
        >
          {loading ? (
            <div className="relative">
              <div className="w-10 h-10 border-2 border-clinical-primary/30 dark:border-blue-400/30 rounded-full" />
              <div className="absolute inset-0 w-10 h-10 border-2 border-clinical-primary dark:border-blue-400 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <div className={`p-3 rounded-full transition-colors duration-200 ${dragging ? 'bg-blue-100 dark:bg-blue-900/40' : 'bg-slate-100 dark:bg-slate-700'}`}>
              <UploadCloud
                size={24}
                className={dragging ? 'text-clinical-primary dark:text-blue-400' : 'text-clinical-text-muted dark:text-slate-400'}
                strokeWidth={1.5}
              />
            </div>
          )}
          <div className="text-center">
            <p className="text-sm font-semibold text-clinical-text-primary dark:text-slate-200">
              {loading ? 'Processing…' : dragging ? 'Drop to upload' : 'Drop CSV or click to browse'}
            </p>
            <p className="text-xs text-clinical-text-muted dark:text-slate-400 mt-1">
              6 feature columns · max 10 MB · .csv only
            </p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="sr-only"
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
        </div>
      ) : (
        <div className="flex items-center justify-between px-4 py-3 bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-xl animate-fade-in-up">
          <div className="flex items-center gap-2 text-sm">
            <CheckCircle2 size={15} className="text-emerald-600 dark:text-emerald-400" />
            <FileText size={14} className="text-emerald-600 dark:text-emerald-400" />
            <span className="font-medium text-emerald-800 dark:text-emerald-200">File loaded & analysed</span>
          </div>
          <button
            onClick={onClear}
            className="flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
            aria-label="Clear uploaded data"
          >
            <X size={13} />
            Clear
          </button>
        </div>
      )}

      {errors.length > 0 && (
        <div className="rounded-xl border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 p-3 space-y-1 animate-fade-in-up">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-clinical-critical dark:text-red-400">
            <AlertCircle size={12} />
            {errors.length} validation issue{errors.length > 1 ? 's' : ''}
          </div>
          <ul className="space-y-0.5 pl-4">
            {errors.slice(0, 6).map((e, i) => (
              <li key={i} className="text-xs text-red-700 dark:text-red-300 list-disc">{e}</li>
            ))}
            {errors.length > 6 && (
              <li className="text-xs text-red-500 dark:text-red-400 list-none">+{errors.length - 6} more…</li>
            )}
          </ul>
        </div>
      )}

      <p className="text-2xs text-clinical-text-muted dark:text-slate-500 leading-relaxed">
        Columns: cardiac_amplitude · cardiac_frequency · respiratory_amplitude · slow_wave_power · cardiac_power · mean_arterial_pressure
      </p>
    </div>
  )
}
