import type { ICPClass } from '../types'

export const CLASS_LABELS = ['Normal', 'Abnormal'] as const

export const CLASS_COLORS: Record<ICPClass, string> = {
  0: '#059669',
  1: '#DC2626',
}
export const CLASS_BG: Record<ICPClass, string> = {
  0: '#D1FAE5',
  1: '#FEE2E2',
}
export const CLASS_TEXT: Record<ICPClass, string> = {
  0: '#065F46',
  1: '#991B1B',
}
export const CLASS_BORDER: Record<ICPClass, string> = {
  0: '#A7F3D0',
  1: '#FECACA',
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

/**
 * P(abnormal) → estimated ICP (mmHg).
 * Literature: Czosnyka & Pickard (Brain 2004).
 * Anchored: P(threshold) ↦ 15 mmHg (clinical intracranial hypertension threshold).
 * Scale = 3 mmHg / logit unit. Clamped to [5, 40] mmHg.
 *
 * Examples (LSTM threshold = 0.285):
 *   P=0.285 → 15 mmHg   (threshold = clinical boundary)
 *   P=0.10  → ~11 mmHg  (clearly normal)
 *   P=0.70  → ~20 mmHg  (mildly elevated)
 *   P=0.90  → ~24 mmHg  (moderately elevated)
 */
export function probToICP(p: number, threshold = 0.5): number {
  const pc = Math.max(0.001, Math.min(0.999, p))
  const tc = Math.max(0.001, Math.min(0.999, threshold))
  const logitShift = Math.log(pc / (1 - pc)) - Math.log(tc / (1 - tc))
  return Math.max(5, Math.min(40, 15 + 3 * logitShift))
}

/**
 * MAP (mmHg) → estimated ICP (mmHg).
 * Based on CPP = MAP − ICP; assumes normal CPP ≈ 70 mmHg (Rosner & Daughton 1990).
 * ICP = MAP − 70, clamped to [5, 40] mmHg.
 */
export function mapToICP(map: number): number {
  return Math.max(5, Math.min(40, map - 70))
}

/**
 * ICP (mmHg) → P(abnormal). Exact inverse of probToICP.
 * Useful for converting MAP-based ICP estimates back to probability space.
 */
export function icpToProb(icp: number, threshold = 0.5): number {
  const tc = Math.max(0.001, Math.min(0.999, threshold))
  const logitT = Math.log(tc / (1 - tc))
  const logit = (Math.max(5, Math.min(40, icp)) - 15) / 3 + logitT
  return Math.max(0.001, Math.min(0.999, 1 / (1 + Math.exp(-logit))))
}

/** Lundberg clinical ICP grade. */
export function icpGrade(icp: number): { label: string; color: string; darkColor: string } {
  if (icp < 15)  return { label: 'Normal',   color: '#059669', darkColor: '#10B981' }
  if (icp < 20)  return { label: 'Grade I',  color: '#D97706', darkColor: '#F59E0B' }
  if (icp < 40)  return { label: 'Grade II', color: '#DC2626', darkColor: '#EF4444' }
  return           { label: 'Grade III', color: '#7C2D12', darkColor: '#FCA5A5' }
}
