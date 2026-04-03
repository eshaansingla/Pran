export type ICPClass = 0 | 1 | 2
export type ICPClassName = 'Normal' | 'Elevated' | 'Critical'

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
  probabilities: [number, number, number]
  confidence: number
  timestamp: string
  top_features: FeatureContribution[]
}

export interface WindowPrediction {
  window_id: number
  class: ICPClass
  class_name: ICPClassName
  probabilities: [number, number, number]
  confidence: number
}

export interface BatchSummary {
  total: number
  normal: number
  elevated: number
  critical: number
  normal_pct: number
  elevated_pct: number
  critical_pct: number
}

export interface BatchResult {
  predictions: WindowPrediction[]
  parse_warnings: string[]
  summary: BatchSummary
  timestamp: string
  feature_names: string[]
}

export interface ModelMetrics {
  macro_f1: number
  weighted_f1: number
  balanced_accuracy: number
  auc_normal: number
  auc_elevated: number
  auc_critical: number
}

export interface ModelInfo {
  version: string
  model_type: string
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
}

export interface TrendPoint {
  windowId: number
  timestamp: string
  class: ICPClass
  confidence: number
  label: string
}

export type ActiveTab = 'dashboard' | 'forecasting' | 'model'
