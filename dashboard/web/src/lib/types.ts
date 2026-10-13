export type GpuProcess = {
  pid: number
  name: string
  vram_mb: number
}

export type Holder = {
  runtime_id: string
  label: string
  alive: boolean
  busy: boolean
  pid: number
  acquired_at: string
  last_active_at: string
  vram_mb: number
  idle_age_s: number
  state: string
  shutdown_url: string | null
}

export type ServiceCard = {
  title: string
  command: string
  state: string
  runtime_id: string
  holds_gpu: boolean
  blurb: string
  engine?: string
}

export type BrokerEvent = {
  id: number
  ts: string
  event: string
  runtime_id: string
  detail: string | null
}

export type Status = {
  ok: boolean
  mock: boolean
  sampled_at: string
  host: string
  distro: string
  gpu: {
    name: string
    mem_used_mb: number
    mem_total_mb: number
    util_pct: number
    processes: GpuProcess[]
  }
  broker: {
    version: string
    protocol: number
    reachable: boolean
    dir: string
  }
  holder: Holder | null
  watchdog: {
    running: boolean
    pid: number | null
    idle_timeout_s: number
    poll_s: number
    next_evict_in_s: number | null
    blocked_reason: string | null
    idle_age_s: number | null
  }
  gateway: {
    up: boolean
    url: string
    served_model: string
    selected_model: string
    selected_runtime: string
    cache_k: string
    cache_v: string
    context_length: number
    active_requests: number
    idle_status: string
    seconds_until_offload: number | null
    idle_timeout_seconds: number
    idle_offload_enabled: boolean
  }
  catalog: {
    models: { name: string; context_length: number; notes?: string }[]
    lanes: string[]
    cache_types: string[]
  }
  services: Record<string, ServiceCard>
  events: BrokerEvent[]
}

export type Job = {
  id: string
  action: string
  status: "running" | "ok" | "error"
  output: string
  error: string
}

export type LlamaForm = {
  model: string
  lane: string
  cacheK: string
  cacheV: string
  dirty: boolean
}
