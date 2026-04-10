export type ICPClass = 0 | 1
export type ICPClassName = 'Normal' | 'Abnormal'

export interface FeatureContribution {
  name: string
  value: number
  unit: string
  status: 'HIGH' | 'NORMAL' | 'LOW'
  shap: number
  impact_pct: number
}

export interface SinglePrediction {
  class: ICPClass
  class_name: ICPClassName
  probability: number
  probabilities: [number, number]
  confidence: number
  timestamp: string
  top_features: FeatureContribution[]
}

export interface WindowPrediction {
  window_id: number
  class: ICPClass
  class_name: ICPClassName
  probability: number
  probabilities: [number, number]
  confidence: number
}

export interface BatchSummary {
  total: number
  normal: number
  abnormal: number
  normal_pct: number
  abnormal_pct: number
}

export interface BatchResult {
  predictions: WindowPrediction[]
  parse_warnings: string[]
  summary: BatchSummary
  timestamp: string
  feature_names: string[]
}

export interface ModelMetrics {
  f1: number
  auc: number
  precision: number
  recall: number
  specificity: number
  balanced_accuracy: number
}

export interface ModelInfo {
  version: string
  model_type: string
  classifier: string
  threshold_mmhg: number
  calibrated?: boolean
  metrics: ModelMetrics
  training_date: string
  training_data: {
    charis_patients: number
    mimic_patients: number
    total_windows: number
  }
  features: string[]
  feature_units: Record<string, string>
  feature_ranges: Record<string, [number, number]>
  classes: string[]
  hyperparameters: Record<string, number | string>
  global_importances: Record<string, number>
  previous_model_note: string
}

export interface TrendPoint {
  windowId: number
  timestamp: string
  class: ICPClass
  confidence: number
  label: string
}

// ── LSTM Forecasting ──────────────────────────────────────────────────────────

export interface AttentionHighlight {
  timestep: number      // negative = N windows before now (e.g. -5 = 50 sec ago)
  importance: number    // 0–1
}

export interface FeatureHighlight {
  name: string
  importance: number    // 0–1
}

export interface ForecastResult {
  class: ICPClass
  class_name: ICPClassName
  probability: number
  probabilities: [number, number]
  confidence_label: 'High' | 'Medium' | 'Low'
  ci_lower: number
  ci_upper: number
  std: number
  horizon_minutes: number
  interpretation: string
  attention_weights: number[]     // length = seq_len (30), sum ≈ 1
  attention_highlights: AttentionHighlight[]
  feature_highlights: FeatureHighlight[]
  model_version: string
  seq_len: number
  threshold: number
  timestamp: string
  forecast_probabilities?: number[]
  forecast_ci_lower?: number[]
  forecast_ci_upper?: number[]
}

export type ActiveTab = 'dashboard' | 'forecasting' | 'model'
