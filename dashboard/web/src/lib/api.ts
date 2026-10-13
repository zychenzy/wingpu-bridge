import type { Job, Status } from "@/lib/types"

export class ApiError extends Error {}

async function request(path: string, options?: RequestInit): Promise<Record<string, unknown>> {
  const response = await fetch(path, options)
  const payload = (await response.json().catch(() => ({ ok: false, error: "bad response" }))) as {
    ok?: boolean
    error?: string
  }
  if (!response.ok || payload.ok === false) {
    throw new ApiError(payload.error || `request failed (${response.status})`)
  }
  return payload
}

export function getStatus(): Promise<Status> {
  return request("/api/status") as unknown as Promise<Status>
}

export async function getLogs(source: string): Promise<string> {
  const payload = await request(`/api/logs/${source}?n=80`)
  return String(payload.text ?? "")
}

async function waitJob(id: string): Promise<string> {
  for (let i = 0; i < 60; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 1000))
    const payload = await request(`/api/jobs/${id}`)
    const job = payload.job as Job
    if (job.status === "ok") return job.output || "done"
    if (job.status === "error") throw new ApiError(job.error || "failed")
  }
  return "still running"
}

export async function runAction(name: string, body?: Record<string, unknown>): Promise<string> {
  const payload = await request(`/api/actions/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  })
  const job = payload.job as Job
  if (job.status === "running") return waitJob(job.id)
  if (job.status === "error") throw new ApiError(job.error || `${name} failed`)
  return job.output || `${name} ok`
}
