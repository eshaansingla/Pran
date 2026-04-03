import type { ICPClass } from '../types'

export const CLASS_LABELS = ['Normal', 'Elevated', 'Critical'] as const
export const CLASS_COLORS: Record<ICPClass, string> = {
  0: '#059669',
  1: '#D97706',
  2: '#DC2626',
}
export const CLASS_BG: Record<ICPClass, string> = {
  0: '#D1FAE5',
  1: '#FEF3C7',
  2: '#FEE2E2',
}
export const CLASS_TEXT: Record<ICPClass, string> = {
  0: '#065F46',
  1: '#92400E',
  2: '#991B1B',
}
export const CLASS_BORDER: Record<ICPClass, string> = {
  0: '#A7F3D0',
  1: '#FDE68A',
  2: '#FECACA',
}

export function fmtTimestamp(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString('en-GB', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  }).replace(',', '')
}

export function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`
}

export function fmtFeatureName(raw: string): string {
  return raw
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function isoNow(): string {
  return new Date().toISOString()
}

export function sessionId(): string {
  const d = new Date()
  return `SES-${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}-${String(d.getHours()).padStart(2,'0')}${String(d.getMinutes()).padStart(2,'0')}`
}
