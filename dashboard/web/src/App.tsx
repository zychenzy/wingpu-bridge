import { useCallback, useEffect, useRef, useState } from "react"
import { toast } from "sonner"

import { AppSidebar } from "@/components/app-sidebar"
import { EventsPanel } from "@/components/events-panel"
import { LogPanel } from "@/components/log-panel"
import { Overview } from "@/components/overview"
import { Toaster } from "@/components/ui/sonner"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Workloads } from "@/components/workloads"
import { ApiError, getLogs, getStatus, runAction } from "@/lib/api"
import { applyTheme, clockText, readTheme } from "@/lib/format"
import type { LlamaForm, Status } from "@/lib/types"
import { Badge } from "@/components/ui/badge"

const emptyLlama: LlamaForm = { model: "", lane: "", cacheK: "", cacheV: "", dirty: false }

export default function App() {
  const [data, setData] = useState<Status | null>(null)
  const [theme, setTheme] = useState<"light" | "dark">(readTheme)
  const [section, setSection] = useState("overview")
  const [busy, setBusy] = useState(false)
  const [evictOpen, setEvictOpen] = useState(false)
  const [logSource, setLogSource] = useState("llama")
  const [logText, setLogText] = useState("Loading…")
  const [logEpoch, setLogEpoch] = useState(0)
  const [llama, setLlama] = useState<LlamaForm>(emptyLlama)
  const [engine, setEngine] = useState({ value: "whisper", dirty: false })
  const [pdf, setPdf] = useState("")
  const busyRef = useRef(false)
  const evictRef = useRef(false)

  useEffect(() => {
    busyRef.current = busy
  }, [busy])
  useEffect(() => {
    evictRef.current = evictOpen
  }, [evictOpen])

  const refresh = useCallback(async () => {
    const next = await getStatus()
    setData(next)
  }, [])

  useEffect(() => {
    let stop = false
    const tick = () => {
      if (busyRef.current || evictRef.current) return
      getStatus()
        .then((next) => {
          if (!stop) setData(next)
        })
        .catch((error: unknown) => {
          if (!stop && !busyRef.current) {
            toast.error(error instanceof Error ? error.message : "status failed")
          }
        })
    }
    tick()
    const id = window.setInterval(tick, 5000)
    return () => {
      stop = true
      window.clearInterval(id)
    }
  }, [])

  useEffect(() => {
    if (!data) return
    setLlama((prev) =>
      prev.dirty
        ? prev
        : {
            model: data.gateway.selected_model,
            lane: data.gateway.selected_runtime,
            cacheK: data.gateway.cache_k,
            cacheV: data.gateway.cache_v,
            dirty: false,
          },
    )
    setEngine((prev) =>
      prev.dirty ? prev : { value: data.services.asr?.engine || "whisper", dirty: false },
    )
  }, [data])

  useEffect(() => {
    let cancel = false
    getLogs(logSource)
      .then((text) => {
        if (!cancel) setLogText(text || "(empty)")
      })
      .catch((error: unknown) => {
        if (!cancel) setLogText(error instanceof Error ? error.message : "log failed")
      })
    return () => {
      cancel = true
    }
  }, [logSource, logEpoch])

  async function act(name: string, body?: Record<string, unknown>) {
    setBusy(true)
    try {
      const message = await runAction(name, body)
      toast.success(message || `${name} ok`)
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : "action failed")
    } finally {
      setBusy(false)
      setEvictOpen(false)
      setLogEpoch((value) => value + 1)
      try {
        await refresh()
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "status failed")
      }
    }
  }

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark"
    applyTheme(next, true)
    setTheme(next)
  }

  function showLog(source: string) {
    setLogSource(source)
    setSection("logs")
    document.getElementById("logs")?.scrollIntoView({ behavior: "smooth", block: "start" })
  }

  return (
    <TooltipProvider>
      <SidebarProvider>
        <AppSidebar
          section={section}
          onNavigate={setSection}
          host={data?.host || "win-gpu"}
          distro={data?.distro || "Ubuntu"}
        />
        <SidebarInset>
          <header className="sticky top-0 z-20 flex h-12 items-center gap-2 border-b bg-background px-3">
            <SidebarTrigger />
            <Separator orientation="vertical" className="h-4" />
            <h1 className="text-sm font-medium">GPU Bridge</h1>
            <div className="ml-auto flex items-center gap-3">
              {data?.mock ? <Badge variant="outline">Example data</Badge> : null}
              <span className="hidden text-sm text-muted-foreground sm:inline">{data?.host || "win-gpu"}</span>
              <time className="font-mono text-xs tabular-nums text-muted-foreground" dateTime={data?.sampled_at || ""}>
                {clockText(data?.sampled_at)}
              </time>
              <Button variant="outline" size="sm" type="button" onClick={toggleTheme}>
                {theme === "dark" ? "Light" : "Dark"}
              </Button>
            </div>
          </header>
          <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-8 p-4 md:p-6">
            {data?.mock ? (
              <p className="text-sm text-muted-foreground">
                Example data. This page is not talking to the GPU host.
              </p>
            ) : null}
            {data ? (
              <>
                <section id="overview" className="grid gap-3">
                  <h2 className="text-sm font-medium">Overview</h2>
                  <Overview
                    data={data}
                    busy={busy}
                    onEvict={() => setEvictOpen(true)}
                    onInstall={() => void act("broker-install")}
                    onWatchdog={() => void act(data.watchdog.running ? "watchdog-stop" : "watchdog-start")}
                  />
                </section>
                <section id="workloads">
                  <Workloads
                    data={data}
                    busy={busy}
                    llama={llama}
                    onLlama={setLlama}
                    engine={engine.value}
                    onEngine={(value) => setEngine({ value, dirty: true })}
                    pdf={pdf}
                    onPdf={setPdf}
                    onAction={(name, body) => {
                      void act(name, body).then(() => {
                        if (name === "llama-apply") setLlama((prev) => ({ ...prev, dirty: false }))
                        if (name.startsWith("asr-")) setEngine((prev) => ({ ...prev, dirty: false }))
                      })
                    }}
                    onLog={showLog}
                  />
                </section>
                <section id="events">
                  <EventsPanel events={data.events || []} />
                </section>
                <section id="logs">
                  <LogPanel source={logSource} text={logText} onSource={setLogSource} />
                </section>
              </>
            ) : (
              <div className="grid gap-4">
                <Skeleton className="h-8 w-40" />
                <Skeleton className="h-40 w-full" />
                <Skeleton className="h-64 w-full" />
              </div>
            )}
          </div>
        </SidebarInset>
        <AlertDialog open={evictOpen} onOpenChange={setEvictOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Evict {data?.holder?.runtime_id || "the holder"}?</AlertDialogTitle>
              <AlertDialogDescription>
                The broker stops that process and waits for VRAM to drain.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction variant="destructive" onClick={() => void act("evict", { confirm: true })}>
                Evict now
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
        <Toaster theme={theme} />
      </SidebarProvider>
    </TooltipProvider>
  )
}
