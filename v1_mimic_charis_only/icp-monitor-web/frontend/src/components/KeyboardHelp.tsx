import { useEffect } from 'react'
import { X, Keyboard } from 'lucide-react'

const SHORTCUTS = [
  { keys: ['Ctrl', 'U'],  action: 'Upload CSV file' },
  { keys: ['Ctrl', 'E'],  action: 'Export PDF report' },
  { keys: ['Ctrl', 'D'],  action: 'Toggle dark mode' },
  { keys: ['Ctrl', 'H'],  action: 'Show keyboard shortcuts' },
  { keys: ['Ctrl', '1'],  action: 'Go to ICP Classification tab' },
  { keys: ['Ctrl', '2'],  action: 'Go to ICP Forecasting tab' },
  { keys: ['Ctrl', '3'],  action: 'Go to Model Info tab' },
  { keys: ['←', '→'],    action: 'Previous / Next window in modal' },
  { keys: ['Esc'],        action: 'Close modal / overlay' },
]

interface Props {
  onClose: () => void
}

export default function KeyboardHelp({ onClose }: Props) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose() }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard shortcuts"
    >
      <div
        className="absolute inset-0 bg-black/50 dark:bg-black/70"
        onClick={onClose}
        aria-hidden="true"
      />
      <div className="relative bg-white dark:bg-slate-800 rounded-lg shadow-xl w-full max-w-sm border border-clinical-border dark:border-slate-600">
        <div className="flex items-center justify-between px-5 py-4 border-b border-clinical-border dark:border-slate-700">
          <div className="flex items-center gap-2">
            <Keyboard size={16} className="text-clinical-primary dark:text-blue-400" />
            <h2 className="text-sm font-semibold text-clinical-text-primary dark:text-slate-100">
              Keyboard Shortcuts
            </h2>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="p-1.5 rounded text-clinical-text-muted dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        <div className="px-5 py-4 space-y-2">
          {SHORTCUTS.map(({ keys, action }, i) => (
            <div key={i} className="flex items-center justify-between">
              <span className="text-xs text-clinical-text-secondary dark:text-slate-300">{action}</span>
              <div className="flex items-center gap-1">
                {keys.map((k, j) => (
                  <span key={j}>
                    <kbd className="inline-flex items-center px-1.5 py-0.5 text-2xs font-mono bg-slate-100 dark:bg-slate-700 text-clinical-text-primary dark:text-slate-200 rounded border border-clinical-border dark:border-slate-600">
                      {k}
                    </kbd>
                    {j < keys.length - 1 && (
                      <span className="text-2xs text-clinical-text-muted dark:text-slate-500 mx-0.5">+</span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="px-5 pb-4">
          <p className="text-2xs text-clinical-text-muted dark:text-slate-500 text-center">
            Press Esc to close
          </p>
        </div>
      </div>
    </div>
  )
}
