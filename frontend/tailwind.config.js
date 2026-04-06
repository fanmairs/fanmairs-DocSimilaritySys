/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{vue,js}"],
  theme: {
    extend: {
      colors: {
        "paper-50": "#f7f6f1",
        "paper-100": "#efede3",
        "ink-900": "#1b2622",
        "night-950": "#0e1614",
        "night-900": "#182420",
        "mint-500": "#3f8f79",
        "mint-600": "#2f6f5e",
        "amber-500": "#e3883b",
        "amber-600": "#be6924"
      },
      fontFamily: {
        display: ['"Manrope"', '"Noto Sans SC"', "sans-serif"],
        body: ['"Noto Sans SC"', "sans-serif"],
        prose: ['"Noto Sans SC"', "sans-serif"]
      },
      boxShadow: {
        soft: "0 20px 52px -30px rgba(21, 41, 35, 0.55)",
        panel: "0 18px 38px -24px rgba(16, 34, 30, 0.42)"
      },
      keyframes: {
        "rise-in": {
          "0%": { opacity: 0, transform: "translateY(10px)" },
          "100%": { opacity: 1, transform: "translateY(0)" }
        },
        "float-orb": {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" }
        },
        "ambient-float": {
          "0%, 100%": { transform: "translate3d(0, 0, 0)" },
          "50%": { transform: "translate3d(0, -8px, 0)" }
        },
        "soft-pulse": {
          "0%, 100%": { opacity: 0.72 },
          "50%": { opacity: 1 }
        }
      },
      animation: {
        "rise-in": "rise-in 0.5s ease-out forwards",
        "float-orb": "float-orb 5s ease-in-out infinite",
        "ambient-float": "ambient-float 6s ease-in-out infinite",
        "soft-pulse": "soft-pulse 2.4s ease-in-out infinite"
      }
    }
  },
  plugins: []
};
