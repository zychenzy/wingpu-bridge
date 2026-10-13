import type { ReactNode } from "react"

import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

type Tone = "ok" | "warn" | "bad" | "idle"

const toneClass: Record<Tone, string> = {
  ok: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  warn: "bg-amber-500/10 text-amber-800 dark:text-amber-400",
  bad: "bg-destructive/10 text-destructive",
  idle: "bg-muted text-muted-foreground",
}

const dotClass: Record<Tone, string> = {
  ok: "bg-emerald-600 dark:bg-emerald-400",
  warn: "bg-amber-600 dark:bg-amber-400",
  bad: "bg-destructive",
  idle: "bg-muted-foreground/50",
}

export function StatusBadge({ children, tone }: { children: ReactNode; tone: Tone }) {
  return (
    <Badge variant="secondary" className={cn("font-normal", toneClass[tone])}>
      <span className={cn("size-1.5 rounded-full", dotClass[tone])} aria-hidden="true" />
      {children}
    </Badge>
  )
}

export function serviceTone(state: string | undefined): Tone {
  if (state === "loaded" || state === "running") return "ok"
  if (state === "stopped" || state === "idle") return "idle"
  return "idle"
}
