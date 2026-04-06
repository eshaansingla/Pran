import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { BatchResult, ForecastResult, WindowPrediction } from '../types'

export interface StoredSession {
  id: string
  date: string          // ISO string
  windowCount: number
  abnormalPct: number
  durationMin: number
  normal: number
  abnormal: number
  predictions: WindowPrediction[]  // capped at 200 for storage
}

export interface StoredForecast {
  id: string
  date: string           // ISO string
  fileName: string
  seqLen: number
  durationMin: number    // history window duration
  class: 0 | 1
  probability: number
  confidence_label: string
  horizon_minutes: number
  result: ForecastResult
}

interface AppStore {
  // ── Theme ──────────────────────────────────────────────────
  isDark: boolean
  setDark: (v: boolean) => void
  toggleDark: () => void

  // ── XGBoost session history ────────────────────────────────
  sessions: StoredSession[]
  addSession: (result: BatchResult) => void
  removeSession: (id: string) => void

  // ── LSTM forecast history ──────────────────────────────────
  forecasts: StoredForecast[]
  addForecast: (result: ForecastResult, fileName: string, seqLen: number) => void
  removeForecast: (id: string) => void

  // ── Flagged windows ────────────────────────────────────────
  flagged: number[]
  toggleFlag: (windowId: number) => void
  isFlagged: (windowId: number) => boolean
}

function systemDark(): boolean {
  try { return window.matchMedia('(prefers-color-scheme: dark)').matches } catch { return false }
}

export const useStore = create<AppStore>()(
  persist(
    (set, get) => ({
      // Theme
      isDark: systemDark(),
      setDark: (v) => {
        set({ isDark: v })
        document.documentElement.classList.toggle('dark', v)
      },
      toggleDark: () => {
        const next = !get().isDark
        set({ isDark: next })
        document.documentElement.classList.toggle('dark', next)
      },

      // XGBoost sessions
      sessions: [],
      addSession: (result) => {
        const s: StoredSession = {
          id: `ses-${Date.now()}`,
          date: new Date().toISOString(),
          windowCount: result.predictions.length,
          abnormalPct: result.summary.abnormal_pct,
          durationMin: +(result.predictions.length * 10 / 60).toFixed(1),
          normal: result.summary.normal,
          abnormal: result.summary.abnormal,
          predictions: result.predictions.slice(0, 200),
        }
        set(state => ({ sessions: [s, ...state.sessions].slice(0, 10) }))
      },
      removeSession: (id) =>
        set(state => ({ sessions: state.sessions.filter(s => s.id !== id) })),

      // LSTM forecasts
      forecasts: [],
      addForecast: (result, fileName, seqLen) => {
        const f: StoredForecast = {
          id: `fct-${Date.now()}`,
          date: new Date().toISOString(),
          fileName,
          seqLen,
          durationMin: +(seqLen * 10 / 60).toFixed(1),
          class: result.class,
          probability: result.probability,
          confidence_label: result.confidence_label,
          horizon_minutes: result.horizon_minutes,
          result,
        }
        set(state => ({ forecasts: [f, ...state.forecasts].slice(0, 10) }))
      },
      removeForecast: (id) =>
        set(state => ({ forecasts: state.forecasts.filter(f => f.id !== id) })),

      // Flagged windows
      flagged: [],
      toggleFlag: (windowId) =>
        set(state => ({
          flagged: state.flagged.includes(windowId)
            ? state.flagged.filter(id => id !== windowId)
            : [...state.flagged, windowId],
        })),
      isFlagged: (windowId) => get().flagged.includes(windowId),
    }),
    {
      name: 'icp-monitor-store',
      partialize: (state) => ({
        isDark:    state.isDark,
        sessions:  state.sessions,
        forecasts: state.forecasts,
        flagged:   state.flagged,
      }),
    }
  )
)
