import type { BatchResult, ModelInfo, SinglePrediction } from '../types'

const BASE = '/api'

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const msg =
      body?.detail?.validation_errors?.join('\n') ||
      body?.detail ||
      body?.error ||
      `HTTP ${res.status}`
    throw new Error(String(msg))
  }
  return res.json() as Promise<T>
}

export async function predictSingle(features: number[]): Promise<SinglePrediction> {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  })
  return handleResponse<SinglePrediction>(res)
}

export async function predictBatch(file: File): Promise<BatchResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/predict_batch`, { method: 'POST', body: form })
  return handleResponse<BatchResult>(res)
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${BASE}/model_info`)
  return handleResponse<ModelInfo>(res)
}

export async function fetchExampleCsv(): Promise<string> {
  const res = await fetch(`${BASE}/example_csv`)
  const data = await handleResponse<{ csv: string }>(res)
  return data.csv
}
