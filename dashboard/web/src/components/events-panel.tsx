import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { shortTime } from "@/lib/format"
import type { BrokerEvent } from "@/lib/types"

export function EventsPanel({ events }: { events: BrokerEvent[] }) {
  const rows = events.slice().reverse().slice(0, 8)
  return (
    <Card className="shadow-none">
      <CardHeader>
        <CardDescription>Broker events</CardDescription>
        <CardTitle className="text-base">Recent</CardTitle>
      </CardHeader>
      <CardContent className="px-0">
        {rows.length === 0 ? (
          <p className="px-6 text-sm text-muted-foreground">No events yet.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-6">Time</TableHead>
                <TableHead>Event</TableHead>
                <TableHead>Runtime</TableHead>
                <TableHead className="pr-6">Detail</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={row.id}>
                  <TableCell className="pl-6 font-mono text-xs tabular-nums">{shortTime(row.ts)}</TableCell>
                  <TableCell className="font-medium">{row.event || "event"}</TableCell>
                  <TableCell className="font-mono text-xs">{row.runtime_id || "—"}</TableCell>
                  <TableCell className="pr-6 text-muted-foreground">{row.detail || "—"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
