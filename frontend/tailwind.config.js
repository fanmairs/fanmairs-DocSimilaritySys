/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{vue,js}"],
  theme: {
    extend: {
      colors: {
        "paper-50": "#f7f4ec",
        "paper-100": "#efe9dd",
        "ink-900": "#1d2a25",
        "mint-500": "#3f8f79",
        "mint-600": "#2f6f5e",
        "amber-500": "#e3883b",
        "amber-600": "#be6924"
      },
      fontFamily: {
        display: ['"Manrope"', '"Noto Sans SC"', "sans-serif"],
        body: ['"Source Han Sans SC"', '"Noto Sans SC"', "sans-serif"],
        prose: ['"Noto Serif SC"', '"Source Han Serif SC"', "serif"]
      },
      boxShadow: {
        soft: "0 14px 40px -22px rgba(31, 56, 48, 0.42)"
      },
      keyframes: {
        "rise-in": {
          "0%": { opacity: 0, transform: "translateY(10px)" },
          "100%": { opacity: 1, transform: "translateY(0)" }
        },
        "float-orb": {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" }
        }
      },
      animation: {
        "rise-in": "rise-in 0.5s ease-out forwards",
        "float-orb": "float-orb 5s ease-in-out infinite"
      }
    }
  },
  plugins: []
};
