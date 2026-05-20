import type { Metadata } from "next";
import Link from "next/link";
import { BarChart3, UploadCloud } from "lucide-react";

import "./globals.css";

export const metadata: Metadata = {
  title: "CV Hackathon Leaderboard",
  description: "Leaderboard evaluation platform for computer vision submissions.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-950 antialiased">
        <header className="border-b border-slate-200 bg-white">
          <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
            <Link href="/" className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-950 text-white">
                <BarChart3 className="h-5 w-5" aria-hidden="true" />
              </span>
              <span className="text-base font-semibold tracking-normal text-slate-950">
                CV Leaderboard
              </span>
            </Link>

            <nav className="flex items-center gap-2">
              <Link
                href="/"
                className="rounded-md px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100 hover:text-slate-950"
              >
                Leaderboard
              </Link>
              <Link
                href="/submit"
                className="inline-flex items-center gap-2 rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800"
              >
                <UploadCloud className="h-4 w-4" aria-hidden="true" />
                Submit
              </Link>
            </nav>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
