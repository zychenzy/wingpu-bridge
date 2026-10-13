export function fmtMb(value: number | null | undefined): string {
  if (value == null || Number.isNaN(Number(value))) return "—"
  return Number(value).toLocaleString("en-US")
}

export function fmtDur(seconds: number | null | undefined): string {
  if (seconds == null) return "—"
  const s = Math.max(0, Math.round(Number(seconds)))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${String(r).padStart(2, "0")}s`
  return `${r}s`
}

export function clockText(iso: string | null | undefined): string {
  if (!iso) {
    return new Date().toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
  }
  const match = String(iso).match(/T(\d{2}:\d{2}:\d{2})/)
  return match ? match[1] : String(iso)
}

export function shortTime(iso: string | null | undefined): string {
  const match = String(iso || "").match(/T(\d{2}:\d{2}:\d{2})/)
  return match ? match[1] : "—"
}

export function readTheme(): "light" | "dark" {
  const query = new URLSearchParams(location.search).get("theme")
  const stored = localStorage.getItem("gpu-dashboard-theme")
  const theme = query || stored || "dark"
  return theme === "light" ? "light" : "dark"
}

export function applyTheme(theme: "light" | "dark", persist: boolean): void {
  const root = document.documentElement
  root.dataset.theme = theme
  root.classList.toggle("dark", theme === "dark")
  root.style.colorScheme = theme
  if (persist) localStorage.setItem("gpu-dashboard-theme", theme)
}
