import LSTMPlaceholder from '../components/LSTMPlaceholder'

export default function Forecasting() {
  return (
    <div>
      <div className="mb-5">
        <h1 className="text-base font-semibold text-clinical-text-primary">ICP Trend Forecasting</h1>
        <p className="text-sm text-clinical-text-muted mt-0.5">
          LSTM-based predictive modelling — planned for v2.0
        </p>
      </div>
      <LSTMPlaceholder />
    </div>
  )
}
