"use client";

import { AlertCircle, CheckCircle2, X } from "lucide-react";

export type ToastState = {
  type: "success" | "error";
  message: string;
} | null;

type ToastProps = {
  toast: ToastState;
  onClose: () => void;
};

export function Toast({ toast, onClose }: ToastProps) {
  if (!toast) {
    return null;
  }

  const Icon = toast.type === "success" ? CheckCircle2 : AlertCircle;
  const colorClasses =
    toast.type === "success"
      ? "border-emerald-200 bg-emerald-50 text-emerald-900"
      : "border-rose-200 bg-rose-50 text-rose-900";

  return (
    <div
      role="status"
      className={`fixed right-4 top-20 z-50 flex w-[calc(100vw-2rem)] max-w-sm items-start gap-3 rounded-lg border p-4 shadow-lg ${colorClasses}`}
    >
      <Icon className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
      <p className="min-w-0 flex-1 text-sm font-medium leading-5">
        {toast.message}
      </p>
      <button
        type="button"
        onClick={onClose}
        className="rounded-md p-1 hover:bg-white/60"
        aria-label="Close notification"
      >
        <X className="h-4 w-4" aria-hidden="true" />
      </button>
    </div>
  );
}
