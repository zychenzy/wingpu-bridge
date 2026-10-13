import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

const SOURCES = ["llama", "gateway", "watchdog", "ocr", "locate", "asr", "mineru"]

export function LogPanel({
  source,
  text,
  onSource,
}: {
  source: string
  text: string
  onSource: (source: string) => void
}) {
  return (
    <Card className="shadow-none">
      <CardHeader>
        <CardDescription>Logs</CardDescription>
        <CardTitle className="text-base">Output</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3">
        <Tabs value={source} onValueChange={onSource}>
          <TabsList variant="line" className="h-auto w-full flex-wrap justify-start">
            {SOURCES.map((item) => (
              <TabsTrigger key={item} value={item} className="font-mono text-xs">
                {item}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <pre className="max-h-80 overflow-auto rounded-lg border bg-muted/40 p-4 font-mono text-xs leading-5 whitespace-pre-wrap">
          {text || "(empty)"}
        </pre>
      </CardContent>
    </Card>
  )
}
