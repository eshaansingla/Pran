/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        clinical: {
          normal:     '#059669',
          abnormal:   '#DC2626',
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
        dark: {
          background: '#1A202C',
          panel:      '#2D3748',
          border:     '#4A5568',
          normal:     '#10B981',
          abnormal:   '#EF4444',
          primary:    '#3B82F6',
          text: {
            primary:   '#E2E8F0',
            secondary: '#A0AEC0',
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
      transitionDuration: {
        '200': '200ms',
      },
    },
  },
  plugins: [],
}
