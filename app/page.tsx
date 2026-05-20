"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Eye, Medal, RefreshCw, Trophy, UploadCloud } from "lucide-react";

import { StatusBadge } from "@/components/status-badge";
import { Toast, type ToastState } from "@/components/toast";
import { fetchLeaderboard, type LeaderboardEntry } from "@/lib/api";

type LeaderboardMode = "type" | "brand";
type SortBy = "eff" | "acc";

const rankStyles = [
  "border-yellow-300 bg-yellow-100 text-yellow-900",
  "border-slate-300 bg-slate-100 text-slate-800",
  "border-orange-300 bg-orange-100 text-orange-900",
];

export default function LeaderboardPage() {
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [leaderboardMode, setLeaderboardMode] = useState<LeaderboardMode>("type");
  const [sortBy, setSortBy] = useState<SortBy>("acc");
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [toast, setToast] = useState<ToastState>(null);

  async function loadLeaderboard(options?: { refreshing?: boolean }) {
    if (options?.refreshing) {
      setIsRefreshing(true);
    } else {
      setIsLoading(true);
    }

    try {
      const data = await fetchLeaderboard({ mode: leaderboardMode, sortBy });
      setEntries(data);
      if (options?.refreshing) {
        setToast({ type: "success", message: "Leaderboard refreshed." });
      }
    } catch (error) {
      setToast({
        type: "error",
        message: error instanceof Error ? error.message : "Unable to load leaderboard.",
      });
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }

  useEffect(() => {
    void loadLeaderboard();
  }, [leaderboardMode, sortBy]);

  const stats = useMemo(() => {
    const doneCount = entries.filter((entry) => entry.status === "done").length;
    const scoreKey = sortBy === "acc" ? "acc_score" : "eff_score";
    const bestScore = entries
      .map((entry) => entry[scoreKey])
      .filter((value): value is number => value !== null)
      .sort((a, b) => b - a)[0];

    return {
      total: entries.length,
      doneCount,
      bestScore,
    };
  }, [entries, sortBy]);

  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <Toast toast={toast} onClose={() => setToast(null)} />

      <div className="flex flex-col gap-5 border-b border-slate-200 pb-6 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600">
            <Trophy className="h-3.5 w-3.5 text-yellow-600" aria-hidden="true" />
            Hackathon Evaluation
          </div>
          <h1 className="text-3xl font-semibold tracking-normal text-slate-950 sm:text-4xl">
            Leaderboard
          </h1>
          <div className="mt-4 flex flex-wrap gap-3 text-sm text-slate-600">
            <span className="rounded-md border border-slate-200 bg-white px-3 py-2">
              {stats.total} {leaderboardMode} submissions
            </span>
            <span className="rounded-md border border-slate-200 bg-white px-3 py-2">
              {stats.doneCount} evaluated
            </span>
            <span className="rounded-md border border-slate-200 bg-white px-3 py-2">
              Best {sortBy.toUpperCase()}{" "}
              {sortBy === "acc"
                ? formatPercent(stats.bestScore)
                : formatEfficiency(stats.bestScore)}
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-2 sm:flex-row">
          <button
            type="button"
            onClick={() => loadLeaderboard({ refreshing: true })}
            disabled={isRefreshing}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-md border border-slate-300 bg-white px-4 text-sm font-semibold text-slate-800 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RefreshCw
              className={`h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`}
              aria-hidden="true"
            />
            Refresh
          </button>
          <Link
            href="/submit"
            className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-semibold text-white hover:bg-slate-800"
          >
            <UploadCloud className="h-4 w-4" aria-hidden="true" />
            Submit Model
          </Link>
        </div>
      </div>

      <section className="mt-6 flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-3 shadow-sm md:flex-row md:items-center md:justify-between">
        <div className="flex flex-wrap gap-2">
          <SegmentButton
            active={leaderboardMode === "type"}
            label="Type Leaderboard"
            onClick={() => setLeaderboardMode("type")}
          />
          <SegmentButton
            active={leaderboardMode === "brand"}
            label="Brand Leaderboard"
            onClick={() => setLeaderboardMode("brand")}
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold text-slate-500">Rank by</span>
          <SegmentButton
            active={sortBy === "eff"}
            label="EFF"
            onClick={() => setSortBy("eff")}
          />
          <SegmentButton
            active={sortBy === "acc"}
            label="ACC"
            onClick={() => setSortBy("acc")}
          />
        </div>
      </section>

      <section className="mt-6 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200">
            <thead className="bg-slate-100">
              <tr>
                <TableHead>Rank</TableHead>
                <TableHead>Team Name</TableHead>
                <TableHead>Mode</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>ACC</TableHead>
                <TableHead>EFF</TableHead>
                <TableHead>Details</TableHead>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {isLoading ? (
                <LoadingRows />
              ) : entries.length > 0 ? (
                entries.map((entry, index) => (
                  <tr
                    key={entry.submission_id}
                    className={index < 3 ? "bg-slate-50/70" : "hover:bg-slate-50"}
                  >
                    <td className="whitespace-nowrap px-4 py-4">
                      <RankBadge rank={index + 1} />
                    </td>
                    <td className="px-4 py-4">
                      <div className="font-semibold text-slate-950">
                        {entry.team_name}
                      </div>
                      <div className="mt-1 text-xs text-slate-500">
                        Submission #{entry.submission_id}
                      </div>
                      {entry.description ? (
                        <p className="mt-2 max-w-xl text-sm leading-5 text-slate-600">
                          {entry.description}
                        </p>
                      ) : null}
                    </td>
                    <td className="whitespace-nowrap px-4 py-4">
                      <span className="rounded-md border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold uppercase text-slate-700">
                        {entry.evaluation_mode}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-4">
                      <StatusBadge status={entry.status} />
                    </td>
                    <td className="whitespace-nowrap px-4 py-4 text-sm font-semibold text-slate-900">
                      {formatPercent(entry.acc_score)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-4 text-sm font-semibold text-slate-900">
                      {formatEfficiency(entry.eff_score)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-4">
                      <Link
                        href={`/submissions/${entry.submission_id}`}
                        className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-slate-300 bg-white px-3 text-sm font-semibold text-slate-800 hover:bg-slate-100"
                      >
                        <Eye className="h-4 w-4" aria-hidden="true" />
                        View
                      </Link>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="px-4 py-14 text-center">
                    <Medal className="mx-auto h-10 w-10 text-slate-300" aria-hidden="true" />
                    <p className="mt-3 text-sm font-medium text-slate-600">
                      No submissions yet.
                    </p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function SegmentButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`h-10 rounded-md border px-3 text-sm font-semibold ${
        active
          ? "border-slate-950 bg-slate-950 text-white"
          : "border-slate-300 bg-white text-slate-700 hover:bg-slate-100"
      }`}
    >
      {label}
    </button>
  );
}

function TableHead({ children }: { children: React.ReactNode }) {
  return (
    <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-normal text-slate-500">
      {children}
    </th>
  );
}

function RankBadge({ rank }: { rank: number }) {
  const topRankClass = rankStyles[rank - 1];

  return (
    <span
      className={`inline-flex h-9 min-w-9 items-center justify-center rounded-full border px-3 text-sm font-bold ${
        topRankClass ?? "border-slate-200 bg-white text-slate-700"
      }`}
    >
      {rank <= 3 ? <Medal className="mr-1 h-4 w-4" aria-hidden="true" /> : null}
      {rank}
    </span>
  );
}

function LoadingRows() {
  return Array.from({ length: 5 }).map((_, index) => (
    <tr key={index}>
      {Array.from({ length: 7 }).map((__, cellIndex) => (
        <td key={cellIndex} className="px-4 py-4">
          <div className="h-5 w-full max-w-36 animate-pulse rounded bg-slate-100" />
        </td>
      ))}
    </tr>
  ));
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(2)}%`;
}

function formatEfficiency(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(4);
}
