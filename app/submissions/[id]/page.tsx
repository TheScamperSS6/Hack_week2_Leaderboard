"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  AlertTriangle,
  ArrowLeft,
  ClipboardList,
  Film,
  RefreshCw,
  Video,
} from "lucide-react";

import { StatusBadge } from "@/components/status-badge";
import { Toast, type ToastState } from "@/components/toast";
import {
  API_BASE_URL,
  fetchSubmissionResults,
  generateSubmissionPreviews,
  type PreviewVideo,
  type QuestionResult,
  type SubmissionResults,
} from "@/lib/api";

export default function SubmissionResultsPage() {
  const params = useParams<{ id: string }>();
  const submissionId = Number(params.id);
  const [report, setReport] = useState<SubmissionResults | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isGeneratingPreviews, setIsGeneratingPreviews] = useState(false);
  const [toast, setToast] = useState<ToastState>(null);

  async function loadResults(options?: { refreshing?: boolean }) {
    if (!Number.isFinite(submissionId)) {
      setToast({ type: "error", message: "Invalid submission id." });
      setIsLoading(false);
      return;
    }

    if (options?.refreshing) {
      setIsRefreshing(true);
    } else {
      setIsLoading(true);
    }

    try {
      const data = await fetchSubmissionResults(submissionId);
      setReport(data);
      if (options?.refreshing) {
        setToast({ type: "success", message: "Results refreshed." });
      }
    } catch (error) {
      setToast({
        type: "error",
        message: error instanceof Error ? error.message : "Unable to load results.",
      });
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }

  useEffect(() => {
    void loadResults();
  }, [submissionId]);

  async function handleGeneratePreviews() {
    if (!Number.isFinite(submissionId)) {
      setToast({ type: "error", message: "Invalid submission id." });
      return;
    }

    setIsGeneratingPreviews(true);
    try {
      const data = await generateSubmissionPreviews(submissionId);
      setReport((current) =>
        current ? { ...current, preview_videos: data.preview_videos } : current,
      );
      setToast({ type: "success", message: "Preview videos generated." });
    } catch (error) {
      setToast({
        type: "error",
        message:
          error instanceof Error ? error.message : "Unable to generate preview videos.",
      });
    } finally {
      setIsGeneratingPreviews(false);
    }
  }

  const averageQuestionScore = useMemo(() => {
    if (!report?.question_results.length) {
      return null;
    }

    const total = report.question_results.reduce(
      (sum, question) => sum + question.acc_score,
      0,
    );
    return total / report.question_results.length;
  }, [report]);

  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <Toast toast={toast} onClose={() => setToast(null)} />

      <div className="flex flex-col gap-5 border-b border-slate-200 pb-6 md:flex-row md:items-end md:justify-between">
        <div>
          <Link
            href="/"
            className="mb-4 inline-flex items-center gap-2 text-sm font-semibold text-slate-600 hover:text-slate-950"
          >
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Back to leaderboard
          </Link>
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600">
            <ClipboardList className="h-3.5 w-3.5 text-slate-700" aria-hidden="true" />
            Submission Results
          </div>
          <h1 className="text-3xl font-semibold tracking-normal text-slate-950 sm:text-4xl">
            {report ? report.team_name : `Submission #${submissionId}`}
          </h1>
          {report?.description ? (
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              {report.description}
            </p>
          ) : null}
          {report ? (
            <div className="mt-4 flex flex-wrap gap-3 text-sm text-slate-600">
              <Metric label="Submission" value={`#${report.submission_id}`} />
              <Metric label="Mode" value={report.evaluation_mode.toUpperCase()} />
              <span className="inline-flex items-center rounded-md border border-slate-200 bg-white px-3 py-2">
                <span className="mr-2 text-slate-500">Status</span>
                <StatusBadge status={report.status} />
              </span>
              <Metric label="Metadata" value={report.metadata_count.toLocaleString()} />
            </div>
          ) : null}
        </div>

        <button
          type="button"
          onClick={() => loadResults({ refreshing: true })}
          disabled={isRefreshing || isLoading}
          className="inline-flex h-11 items-center justify-center gap-2 rounded-md border border-slate-300 bg-white px-4 text-sm font-semibold text-slate-800 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <RefreshCw
            className={`h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`}
            aria-hidden="true"
          />
          Refresh
        </button>
      </div>

      {isLoading ? (
        <LoadingState />
      ) : report ? (
        <>
          <section className="mt-6 grid gap-3 md:grid-cols-3">
            <ScoreCard label="Total ACC" value={formatPercent(report.acc_score)} />
            <ScoreCard label="Question Avg" value={formatPercent(averageQuestionScore)} />
            <ScoreCard label="EFF" value={formatEfficiency(report.eff_score)} />
          </section>

          {report.error_message ? (
            <FailurePanel errorMessage={report.error_message} />
          ) : null}

          <PreviewSection
            previews={report.preview_videos}
            isGenerating={isGeneratingPreviews}
            onGenerate={handleGeneratePreviews}
          />

          <section className="mt-6 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 px-4 py-4">
              <h2 className="text-base font-semibold text-slate-950">
                Question Comparison
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Prediction is calculated from saved metadata using the same scoring logic as the leaderboard.
              </p>
            </div>

            <div className="divide-y divide-slate-100">
              {report.question_results.map((question) => (
                <QuestionPanel key={question.question_id} question={question} />
              ))}
            </div>
          </section>
        </>
      ) : (
        <section className="mt-6 rounded-lg border border-slate-200 bg-white px-4 py-14 text-center shadow-sm">
          <p className="text-sm font-medium text-slate-600">No result data found.</p>
        </section>
      )}
    </main>
  );
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <span className="rounded-md border border-slate-200 bg-white px-3 py-2">
      <span className="text-slate-500">{label}</span>{" "}
      <span className="font-semibold text-slate-900">{value}</span>
    </span>
  );
}

function ScoreCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-normal text-slate-500">
        {label}
      </div>
      <div className="mt-2 text-2xl font-semibold text-slate-950">{value}</div>
    </div>
  );
}

function FailurePanel({ errorMessage }: { errorMessage: string }) {
  return (
    <section className="mt-6 rounded-lg border border-rose-200 bg-rose-50 p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-white text-rose-700 shadow-sm">
          <AlertTriangle className="h-5 w-5" aria-hidden="true" />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-rose-950">
            Evaluation failed
          </h2>
          <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap rounded-md border border-rose-200 bg-white p-3 text-xs leading-5 text-rose-950">
            {errorMessage}
          </pre>
        </div>
      </div>
    </section>
  );
}

function PreviewSection({
  previews,
  isGenerating,
  onGenerate,
}: {
  previews: PreviewVideo[];
  isGenerating: boolean;
  onGenerate: () => void;
}) {
  const availablePreviews = previews.filter((preview) => preview.exists);
  const missingCount = previews.length - availablePreviews.length;

  return (
    <section className="mt-6">
      <div className="flex flex-col gap-3 border-b border-slate-200 pb-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="inline-flex items-center gap-2 text-sm font-semibold text-slate-950">
            <Film className="h-4 w-4 text-slate-600" aria-hidden="true" />
            Detection Preview
          </div>
          <p className="mt-1 text-sm text-slate-500">
            Rendered from saved detection metadata with ROI, track ID, and predicted label overlays.
          </p>
        </div>
        <button
          type="button"
          onClick={onGenerate}
          disabled={isGenerating}
          className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <RefreshCw
            className={`h-4 w-4 ${isGenerating ? "animate-spin" : ""}`}
            aria-hidden="true"
          />
          {isGenerating ? "Generating..." : "Generate Preview"}
        </button>
      </div>

      {availablePreviews.length > 0 ? (
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          {availablePreviews.map((preview) => (
            <div
              key={preview.cctv_id}
              className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm"
            >
              <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
                <div className="inline-flex items-center gap-2 text-sm font-semibold text-slate-950">
                  <Video className="h-4 w-4 text-slate-500" aria-hidden="true" />
                  {preview.cctv_id}
                </div>
                <a
                  href={assetUrl(preview.video_url)}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs font-semibold text-slate-500 hover:text-slate-950"
                >
                  Open
                </a>
              </div>
              <video
                className="aspect-video w-full bg-slate-950"
                controls
                preload="metadata"
                src={assetUrl(preview.video_url)}
              />
            </div>
          ))}
        </div>
      ) : (
        <div className="mt-4 rounded-lg border border-dashed border-slate-300 bg-white px-4 py-8 text-center">
          <Film className="mx-auto h-9 w-9 text-slate-300" aria-hidden="true" />
          <p className="mt-3 text-sm font-medium text-slate-600">
            No preview videos generated yet.
          </p>
        </div>
      )}

      {missingCount > 0 && availablePreviews.length > 0 ? (
        <p className="mt-3 text-sm text-slate-500">
          {missingCount} CCTV preview{missingCount > 1 ? "s are" : " is"} not generated yet.
        </p>
      ) : null}
    </section>
  );
}

function QuestionPanel({ question }: { question: QuestionResult }) {
  return (
    <article className="grid gap-4 px-4 py-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_160px]">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-md bg-slate-950 px-2.5 py-1 text-xs font-semibold text-white">
            {question.question_id}
          </span>
          <span className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-700">
            {question.cctv_id}
          </span>
          <span className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-700">
            {question.time_range}
          </span>
        </div>
        <p className="mt-3 text-sm leading-6 text-slate-700">{question.query}</p>
        <div className="mt-3 text-xs font-semibold uppercase tracking-normal text-slate-500">
          Group by {question.group_by.join(", ")}
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-2">
        <ResultBox title="Prediction" values={question.prediction} tone="prediction" />
        <ResultBox title="Answer" values={question.ground_truth} tone="answer" />
      </div>

      <div className="flex items-start justify-start lg:justify-end">
        <div className="w-full rounded-lg border border-slate-200 bg-slate-50 p-4 lg:w-36">
          <div className="text-xs font-semibold uppercase tracking-normal text-slate-500">
            Jaccard ACC
          </div>
          <div className="mt-2 text-xl font-semibold text-slate-950">
            {formatPercent(question.acc_score)}
          </div>
        </div>
      </div>
    </article>
  );
}

function ResultBox({
  title,
  values,
  tone,
}: {
  title: string;
  values: Record<string, number>;
  tone: "prediction" | "answer";
}) {
  const entries = Object.entries(values).sort(([a], [b]) => a.localeCompare(b));
  const toneClass =
    tone === "prediction"
      ? "border-sky-200 bg-sky-50 text-sky-950"
      : "border-emerald-200 bg-emerald-50 text-emerald-950";

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="text-xs font-semibold uppercase tracking-normal text-slate-500">
        {title}
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {entries.length > 0 ? (
          entries.map(([key, value]) => (
            <span
              key={key}
              className={`inline-flex items-center gap-2 rounded-md border px-2.5 py-1 text-sm font-semibold ${toneClass}`}
            >
              <span className="max-w-36 truncate">{key}</span>
              <span>{value}</span>
            </span>
          ))
        ) : (
          <span className="text-sm font-medium text-slate-400">No count</span>
        )}
      </div>
    </div>
  );
}

function assetUrl(path: string) {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${API_BASE_URL}${path}`;
}

function LoadingState() {
  return (
    <section className="mt-6 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="h-8 max-w-64 animate-pulse rounded bg-slate-100" />
      <div className="mt-6 space-y-4">
        {Array.from({ length: 5 }).map((_, index) => (
          <div key={index} className="h-28 animate-pulse rounded-lg bg-slate-100" />
        ))}
      </div>
    </section>
  );
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
