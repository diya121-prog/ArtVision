/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#faf8f5',
          100: '#f4ede1',
          200: '#e8dcbf',
          300: '#d8c295',
          400: '#c5a369',
          500: '#b48a47',
          600: '#9b7137',
          700: '#7c552c',
          800: '#674627',
          900: '#563a23',
          950: '#301e11',
        },
        atelier: {
          paper: '#faf8f5',
          sand: '#f3efe6',
          canvas: '#ece5d8',
          ink: '#1c1917',
          bronze: '#a86524',
          ochre: '#c68a35',
          verdigris: '#2d6a4f',
          crimson: '#8b1e2b',
        }
      }
    },
  },
  plugins: [],
}
