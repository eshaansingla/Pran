import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { BatchResult, WindowPrediction } from '../types'

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

interface AppStore {
  // ── Theme ──────────────────────────────────────────────────
  isDark: boolean
  setDark: (v: boolean) => void
  toggleDark: () => void

  // ── Session history ────────────────────────────────────────
  sessions: StoredSession[]
  addSession: (result: BatchResult) => void
  removeSession: (id: string) => void

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
      // Theme — initialize from system, override by localStorage via persist
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
        isDark:   state.isDark,
        sessions: state.sessions,
        flagged:  state.flagged,
      }),
    }
  )
)
