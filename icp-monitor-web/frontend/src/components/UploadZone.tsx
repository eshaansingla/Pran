import React, { useCallback, useRef, useState } from 'react'
import { Upload, X, AlertCircle, Download, FileText } from 'lucide-react'
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
        <h2 className="text-sm font-semibold text-clinical-text-secondary uppercase tracking-wide">
          Data Upload
        </h2>
        <button
          onClick={downloadExample}
          className="flex items-center gap-1.5 text-xs text-clinical-primary hover:underline"
          aria-label="Download example CSV"
        >
          <Download size={13} />
          Example CSV
        </button>
      </div>

      {!hasData ? (
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload zone — drag and drop or click to select CSV"
          onClick={() => inputRef.current?.click()}
          onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`
            relative border-2 border-dashed rounded-lg px-6 py-10
            flex flex-col items-center justify-center gap-3 cursor-pointer
            transition-colors duration-150 select-none
            ${dragging
              ? 'border-clinical-primary bg-blue-50'
              : 'border-clinical-border bg-white hover:border-clinical-primary hover:bg-blue-50/30'}
          `}
        >
          <Upload
            size={32}
            className={dragging ? 'text-clinical-primary' : 'text-clinical-text-muted'}
            strokeWidth={1.5}
          />
          <div className="text-center">
            <p className="text-sm font-medium text-clinical-text-primary">
              {loading ? 'Processing…' : 'Drop CSV or click to browse'}
            </p>
            <p className="text-xs text-clinical-text-muted mt-1">
              8 feature columns · max 10 MB · .csv only
            </p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="sr-only"
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
            aria-hidden="true"
          />
        </div>
      ) : (
        <div className="flex items-center justify-between px-4 py-3 bg-white border border-clinical-border rounded-lg">
          <div className="flex items-center gap-2 text-sm text-clinical-text-primary">
            <FileText size={16} className="text-clinical-primary" />
            <span>File loaded</span>
          </div>
          <button
            onClick={onClear}
            className="flex items-center gap-1 text-xs text-clinical-text-muted hover:text-clinical-critical"
            aria-label="Clear uploaded data"
          >
            <X size={14} />
            Clear
          </button>
        </div>
      )}

      {errors.length > 0 && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 space-y-1">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-clinical-critical">
            <AlertCircle size={13} />
            {errors.length} validation issue{errors.length > 1 ? 's' : ''}
          </div>
          <ul className="space-y-0.5 pl-4">
            {errors.slice(0, 8).map((e, i) => (
              <li key={i} className="text-xs text-red-700 list-disc">{e}</li>
            ))}
            {errors.length > 8 && (
              <li className="text-xs text-red-500 list-none">
                +{errors.length - 8} more issues…
              </li>
            )}
          </ul>
        </div>
      )}

      <p className="text-2xs text-clinical-text-muted leading-relaxed">
        Expected columns: cardiac_amplitude, cardiac_frequency, respiratory_amplitude,
        slow_wave_power, cardiac_power, mean_arterial_pressure, head_angle, motion_artifact_flag
      </p>
    </div>
  )
}
