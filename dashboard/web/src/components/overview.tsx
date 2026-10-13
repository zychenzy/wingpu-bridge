import { Bar, BarChart, XAxis, YAxis } from "recharts"

import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { fmtDur, fmtMb, shortTime } from "@/lib/format"
import type { Status } from "@/lib/types"
import { StatusBadge } from "@/components/status-badge"

export function Overview({
  data,
  busy,
  onEvict,
  onInstall,
  onWatchdog,
}: {
  data: Status
  busy: boolean
  onEvict: () => void
  onInstall: () => void
  onWatchdog: () => void
}) {
  const gpu = data.gpu
  const used = Number(gpu.mem_used_mb) || 0
  const total = Number(gpu.mem_total_mb) || 0
  const free = Math.max(0, total - used)
  const pct = total ? Math.max(0, Math.min(100, (used / total) * 100)) : 0
  const util = Math.max(0, Math.min(100, Number(gpu.util_pct) || 0))
  const warn = pct >= 92
  const holder = data.holder
  const broker = data.broker
  const wd = data.watchdog
  const gw = data.gateway

  const chartConfig = {
    used: {
      label: "Used",
      color: warn ? "var(--destructive)" : "var(--foreground)",
    },
    free: { label: "Free", color: "var(--muted)" },
  } satisfies ChartConfig

  const watchdogValue =
    wd.blocked_reason === "busy"
      ? "busy, holding"
      : wd.next_evict_in_s != null
        ? fmtDur(wd.next_evict_in_s)
        : "—"

  return (
    <div className="grid gap-4 lg:grid-cols-6">
      <Card className="shadow-none lg:col-span-3">
        <CardHeader>
          <CardDescription>GPU</CardDescription>
          <CardTitle className="text-base">{gpu.name || "NVIDIA GPU"}</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-medium tabular-nums tracking-tight ${warn ? "text-destructive" : ""}`}>
              {fmtMb(gpu.mem_used_mb)}
            </span>
            <span className="text-sm text-muted-foreground tabular-nums">/ {fmtMb(gpu.mem_total_mb)} MB</span>
            <span className="ml-auto text-sm text-muted-foreground tabular-nums">{Math.round(pct)}%</span>
          </div>
          <ChartContainer config={chartConfig} className="aspect-auto h-8 w-full">
            <BarChart
              data={[{ name: "VRAM", used, free }]}
              layout="vertical"
              margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
              barSize={8}
            >
              <XAxis type="number" hide domain={[0, total || 1]} />
              <YAxis type="category" dataKey="name" hide />
              <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
              <Bar dataKey="used" stackId="vram" fill="var(--color-used)" radius={[2, 0, 0, 2]} />
              <Bar dataKey="free" stackId="vram" fill="var(--color-free)" radius={[0, 2, 2, 0]} />
            </BarChart>
          </ChartContainer>
          {gpu.processes?.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Process</TableHead>
                  <TableHead>PID</TableHead>
                  <TableHead className="text-right">VRAM</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {gpu.processes.map((proc) => (
                  <TableRow key={proc.pid}>
                    <TableCell>{proc.name || "process"}</TableCell>
                    <TableCell className="font-mono tabular-nums">{proc.pid ?? "—"}</TableCell>
                    <TableCell className="text-right tabular-nums">{fmtMb(proc.vram_mb)} MB</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">No processes on the card.</p>
          )}
        </CardContent>
      </Card>

      <Card className="shadow-none lg:col-span-3">
        <CardHeader>
          <CardDescription>Utilization</CardDescription>
          <CardTitle className="text-2xl font-medium tabular-nums tracking-tight">{util}%</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-6">
          <div className="h-1 overflow-hidden rounded-full bg-muted">
            <div className="h-full bg-foreground" style={{ width: `${util}%` }} />
          </div>
          <dl className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <dt className="text-xs text-muted-foreground">Broker</dt>
              <dd className="mt-1 font-mono text-xs">
                {broker.reachable
                  ? `${broker.version} · protocol ${broker.protocol}`
                  : "Unreachable"}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Sample</dt>
              <dd className="mt-1 font-mono text-xs tabular-nums">{shortTime(data.sampled_at)}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <Card className="shadow-none lg:col-span-3">
        <CardHeader>
          <CardDescription>Lease</CardDescription>
          <CardTitle className="font-mono text-base">{holder ? holder.runtime_id : "No holder"}</CardTitle>
          <CardAction className="flex flex-wrap justify-end gap-1">
            {!broker.reachable ? (
              <StatusBadge tone="bad">Unreachable</StatusBadge>
            ) : !holder ? (
              <StatusBadge tone="idle">Free</StatusBadge>
            ) : (
              <>
                <StatusBadge tone={holder.alive ? "ok" : "bad"}>{holder.alive ? "Alive" : "Not alive"}</StatusBadge>
                <StatusBadge tone={holder.busy ? "warn" : "ok"}>{holder.busy ? "Busy" : "Not busy"}</StatusBadge>
              </>
            )}
          </CardAction>
        </CardHeader>
        <CardContent className="grid gap-4">
          <dl className="grid grid-cols-[5rem_1fr] gap-y-2 text-sm">
            <dt className="text-muted-foreground">Label</dt>
            <dd className="font-mono text-xs break-all">{holder?.label || "—"}</dd>
            <dt className="text-muted-foreground">Since</dt>
            <dd className="font-mono text-xs tabular-nums">{holder ? shortTime(holder.acquired_at) : "—"}</dd>
            <dt className="text-muted-foreground">Touch</dt>
            <dd className="font-mono text-xs tabular-nums">
              {holder?.last_active_at ? shortTime(holder.last_active_at) : "never"}
            </dd>
            <dt className="text-muted-foreground">State</dt>
            <dd>{holder?.state || (holder ? "active" : "—")}</dd>
          </dl>
          <div className="flex flex-wrap gap-2">
            <Button variant="destructive" size="sm" disabled={busy || !holder} onClick={onEvict}>
              Evict
            </Button>
            <Button variant="outline" size="sm" disabled={busy} onClick={onInstall}>
              Install broker
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="shadow-none lg:col-span-3">
        <CardHeader>
          <CardDescription>Idle offload</CardDescription>
          <CardTitle className="text-base">Watchdog and gateway</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-2">
          <div className="grid gap-1">
            <p className="text-xs text-muted-foreground">Watchdog</p>
            <p className="text-sm">{wd.running ? `pid ${wd.pid}` : "stopped"}</p>
            <p className="text-2xl font-medium tabular-nums tracking-tight">{watchdogValue}</p>
            <p className="text-xs text-muted-foreground">
              {wd.idle_timeout_s != null ? `limit ${fmtDur(wd.idle_timeout_s)}` : ""}
            </p>
          </div>
          <div className="grid gap-1">
            <p className="text-xs text-muted-foreground">Gateway</p>
            <p className="text-sm">{gw.up ? gw.idle_status || "up" : "down"}</p>
            <p className="text-2xl font-medium tabular-nums tracking-tight">
              {gw.seconds_until_offload != null ? fmtDur(gw.seconds_until_offload) : "—"}
            </p>
            <p className="text-xs text-muted-foreground">
              {gw.idle_timeout_seconds != null ? `limit ${fmtDur(gw.idle_timeout_seconds)}` : ""}
            </p>
          </div>
          <div className="sm:col-span-2">
            <Button variant="outline" size="sm" disabled={busy} onClick={onWatchdog}>
              {wd.running ? "Stop watchdog" : "Start watchdog"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
