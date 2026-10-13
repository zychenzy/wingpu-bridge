import { ActivityIcon, LayoutDashboardIcon, LayersIcon, ScrollTextIcon } from "lucide-react"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar"

const NAV = [
  { id: "overview", label: "Overview", icon: LayoutDashboardIcon },
  { id: "workloads", label: "Workloads", icon: LayersIcon },
  { id: "events", label: "Events", icon: ActivityIcon },
  { id: "logs", label: "Logs", icon: ScrollTextIcon },
] as const

export function AppSidebar({
  section,
  onNavigate,
  host,
  distro,
}: {
  section: string
  onNavigate: (id: string) => void
  host: string
  distro: string
}) {
  const { setOpenMobile } = useSidebar()

  return (
    <Sidebar collapsible="offcanvas">
      <SidebarHeader className="px-4 py-4">
        <div className="text-sm font-medium">GPU Bridge</div>
        <p className="text-xs text-muted-foreground">Loopback · 127.0.0.1:18081</p>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigate</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV.map((item) => (
                <SidebarMenuItem key={item.id}>
                  <SidebarMenuButton
                    asChild
                    isActive={section === item.id}
                  >
                    <a
                      href={`#${item.id}`}
                      onClick={() => {
                        onNavigate(item.id)
                        setOpenMobile(false)
                      }}
                    >
                      <item.icon />
                      <span>{item.label}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="px-4 py-3 text-xs text-muted-foreground">
        <p>{host || "win-gpu"}</p>
        <p>{distro || "Ubuntu"}</p>
      </SidebarFooter>
    </Sidebar>
  )
}
