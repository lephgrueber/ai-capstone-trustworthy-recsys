import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";

export function MovieDialog({ open, onOpenChange, title, description = "Additional details", closeLabel = "Close details", contentClassName = "", accessibleName, children }: { open: boolean; onOpenChange: (open: boolean) => void; title: string; description?: string; closeLabel?: string; contentClassName?: string; accessibleName?: string; children: ReactNode }) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm data-[state=open]:animate-in" />
        <DialogPrimitive.Content className={`dialog-panel fixed inset-y-0 right-0 z-50 w-full max-w-md overflow-y-auto border-l border-slate-700 bg-[#0b1220] p-6 shadow-2xl focus:outline-none sm:p-8 ${contentClassName}`}>
          <DialogPrimitive.Title className="dialog-title pr-10 text-2xl font-bold text-slate-50">{accessibleName ? <><span aria-hidden="true">{title}</span><span className="sr-only">{accessibleName}</span></> : title}</DialogPrimitive.Title>
          <DialogPrimitive.Description className="mt-2 text-sm text-slate-300">{description}</DialogPrimitive.Description>
          {children}
          <DialogPrimitive.Close aria-label={closeLabel} className="absolute right-5 top-5 rounded-full p-2 text-slate-300 hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"><X size={20} /></DialogPrimitive.Close>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
