import React, { createContext, useContext, useMemo, useState } from "react";

export type ThemeMode = "emergence" | "cyberspace" | "synthwave";
const THEME_ORDER: ThemeMode[] = ["emergence", "cyberspace", "synthwave"];

type ThemeContextValue = {
  mode: ThemeMode;
  themes: ThemeMode[];
  toggleMode: () => void;
  setMode: (mode: ThemeMode) => void;
};

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>("emergence");

  const value = useMemo(
    () => ({
      mode,
      themes: THEME_ORDER,
      setMode,
      toggleMode: () =>
        setMode((prev) => {
          const idx = THEME_ORDER.indexOf(prev);
          return THEME_ORDER[(idx + 1) % THEME_ORDER.length];
        }),
    }),
    [mode],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return ctx;
}