import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { LlamaForm, ServiceCard, Status } from "@/lib/types"
import { StatusBadge, serviceTone } from "@/components/status-badge"

function Choice({
  label,
  value,
  options,
  onChange,
  disabled,
}: {
  label: string
  value: string
  options: string[]
  onChange: (value: string) => void
  disabled: boolean
}) {
  if (!value || options.length === 0) {
    return (
      <label className="grid gap-1">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-sm text-muted-foreground">—</span>
      </label>
    )
  }
  return (
    <label className="grid min-w-36 gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <Select value={value} onValueChange={onChange} disabled={disabled}>
        <SelectTrigger size="sm" className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  )
}

export function Workloads({
  data,
  busy,
  llama,
  onLlama,
  engine,
  onEngine,
  pdf,
  onPdf,
  onAction,
  onLog,
}: {
  data: Status
  busy: boolean
  llama: LlamaForm
  onLlama: (next: LlamaForm) => void
  engine: string
  onEngine: (value: string) => void
  pdf: string
  onPdf: (value: string) => void
  onAction: (name: string, body?: Record<string, unknown>) => void
  onLog: (source: string) => void
}) {
  const gw = data.gateway
  const catalog = data.catalog
  const services = data.services
  const llamaSvc = services.llama
  const models = (catalog.models || []).map((item) => item.name)

  function patchLlama(partial: Partial<LlamaForm>) {
    onLlama({ ...llama, ...partial, dirty: true })
  }

  return (
    <Card className="shadow-none">
      <CardHeader>
        <CardDescription>Workloads</CardDescription>
        <CardTitle className="text-base">Models and sibling bridges</CardTitle>
      </CardHeader>
      <CardContent className="px-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-6">Workload</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Configuration</TableHead>
              <TableHead className="pr-6">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow>
              <TableCell className="pl-6 align-top">
                <div className="font-medium">llama.cpp</div>
                <p className="mt-1 max-w-56 text-xs text-muted-foreground">{llamaSvc?.command || "wingpu"}</p>
              </TableCell>
              <TableCell className="align-top">
                <div className="flex flex-col items-start gap-1">
                  <StatusBadge tone={serviceTone(llamaSvc?.state)}>{llamaSvc?.state || "unknown"}</StatusBadge>
                  {llamaSvc?.holds_gpu ? <span className="text-xs text-muted-foreground">Holds GPU</span> : null}
                </div>
              </TableCell>
              <TableCell className="align-top">
                <p className="mb-2 text-xs text-muted-foreground">
                  Served model {gw.served_model || "qwen-local"} · context {gw.context_length || "—"} ·{" "}
                  {gw.active_requests ?? 0} active
                </p>
                <div className="flex flex-wrap gap-2">
                  <Choice
                    label="Model"
                    value={llama.model}
                    options={models}
                    disabled={busy}
                    onChange={(model) => patchLlama({ model })}
                  />
                  <Choice
                    label="Lane"
                    value={llama.lane}
                    options={catalog.lanes || []}
                    disabled={busy}
                    onChange={(lane) => patchLlama({ lane })}
                  />
                  <Choice
                    label="KV K"
                    value={llama.cacheK}
                    options={catalog.cache_types || []}
                    disabled={busy}
                    onChange={(cacheK) => patchLlama({ cacheK })}
                  />
                  <Choice
                    label="KV V"
                    value={llama.cacheV}
                    options={catalog.cache_types || []}
                    disabled={busy}
                    onChange={(cacheV) => patchLlama({ cacheV })}
                  />
                </div>
              </TableCell>
              <TableCell className="pr-6 align-top">
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    disabled={busy}
                    onClick={() =>
                      onAction("llama-apply", {
                        model: llama.model,
                        lane: llama.lane,
                        cache_k: llama.cacheK,
                        cache_v: llama.cacheV,
                      })
                    }
                  >
                    Apply and restart
                  </Button>
                  <Button size="sm" variant="outline" disabled={busy} onClick={() => onAction("llama-offload")}>
                    Offload
                  </Button>
                  <Button size="sm" variant="outline" disabled={busy} onClick={() => onAction("llama-stop")}>
                    Stop
                  </Button>
                  <Button size="sm" variant="outline" disabled={busy} onClick={() => onAction("gateway-restart")}>
                    Restart gateway
                  </Button>
                  <Button size="sm" variant="outline" disabled={busy} onClick={() => onLog("llama")}>
                    Logs
                  </Button>
                </div>
              </TableCell>
            </TableRow>
            {(["ocr", "locate", "asr", "mineru"] as const).map((key) => {
              const spec: ServiceCard | undefined = services[key]
              if (!spec) return null
              return (
                <TableRow key={key}>
                  <TableCell className="pl-6 align-top">
                    <div className="font-medium">{spec.title || key}</div>
                    <p className="mt-1 max-w-56 text-xs text-muted-foreground">{spec.command}</p>
                  </TableCell>
                  <TableCell className="align-top">
                    <div className="flex flex-col items-start gap-1">
                      <StatusBadge tone={serviceTone(spec.state)}>{spec.state || "unknown"}</StatusBadge>
                      {spec.holds_gpu ? <span className="text-xs text-muted-foreground">Holds GPU</span> : null}
                    </div>
                  </TableCell>
                  <TableCell className="align-top">
                    <p className="mb-2 max-w-md text-xs text-muted-foreground">{spec.blurb}</p>
                    {key === "asr" ? (
                      <Choice
                        label="Engine"
                        value={engine}
                        options={["whisper", "qwen3", "both"]}
                        disabled={busy}
                        onChange={onEngine}
                      />
                    ) : null}
                    {key === "mineru" ? (
                      <label className="grid max-w-sm gap-1">
                        <span className="text-xs text-muted-foreground">PDF</span>
                        <Input
                          value={pdf}
                          placeholder="/Users/you/scan.pdf"
                          onChange={(event) => onPdf(event.target.value)}
                          disabled={busy}
                        />
                      </label>
                    ) : null}
                  </TableCell>
                  <TableCell className="pr-6 align-top">
                    <div className="flex flex-wrap gap-2">
                      {key === "mineru" ? (
                        <>
                          <Button size="sm" disabled={busy} onClick={() => onAction("mineru-run", { pdf })}>
                            Run PDF
                          </Button>
                          <Button size="sm" variant="outline" disabled={busy} onClick={() => onAction("mineru-status")}>
                            Status
                          </Button>
                        </>
                      ) : (
                        <>
                          <Button
                            size="sm"
                            disabled={busy}
                            onClick={() => onAction(`${key}-start`, key === "asr" ? { engine } : {})}
                          >
                            Start
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={busy}
                            onClick={() => onAction(`${key}-stop`, key === "asr" ? { engine } : {})}
                          >
                            Stop
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={busy}
                            onClick={() => onAction(`${key}-restart`, key === "asr" ? { engine } : {})}
                          >
                            Restart
                          </Button>
                        </>
                      )}
                      <Button size="sm" variant="outline" disabled={busy} onClick={() => onLog(key)}>
                        Logs
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
