/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        clinical: {
          normal:     '#059669',
          elevated:   '#D97706',
          critical:   '#DC2626',
          primary:    '#2C5282',
          secondary:  '#4A5568',
          background: '#F7FAFC',
          panel:      '#FFFFFF',
          border:     '#E2E8F0',
          text: {
            primary:   '#1A202C',
            secondary: '#4A5568',
            muted:     '#718096',
          },
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': '0.65rem',
      },
    },
  },
  plugins: [],
}
