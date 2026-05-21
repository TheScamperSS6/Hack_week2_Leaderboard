"use client";

import { FormEvent, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Cpu,
  FileJson,
  Gauge,
  Loader2,
  Send,
  UploadCloud,
} from "lucide-react";

import { Toast, type ToastState } from "@/components/toast";
import { submitModel } from "@/lib/api";

type EvaluationMode = "brand" | "type";

type FileFieldProps = {
  label: string;
  accept: string;
  icon: React.ReactNode;
  required?: boolean;
  onChange: (file: File | null) => void;
};

export default function SubmitPage() {
  const [evaluationMode, setEvaluationMode] = useState<EvaluationMode>("type");
  const [teamName, setTeamName] = useState("");
  const [description, setDescription] = useState("");
  const [yoloGflops, setYoloGflops] = useState("");
  const [classGflops, setClassGflops] = useState("");
  const [yoloFile, setYoloFile] = useState<File | null>(null);
  const [classifierFile, setClassifierFile] = useState<File | null>(null);
  const [classifierExternalDataFiles, setClassifierExternalDataFiles] = useState<File[]>([]);
  const [labelsFile, setLabelsFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toast, setToast] = useState<ToastState>(null);
  const [fileInputKey, setFileInputKey] = useState(0);

  const canSubmit = useMemo(
    () =>
      teamName.trim() &&
      yoloFile &&
      (evaluationMode === "type" ||
        (classifierFile && labelsFile)) &&
      !isSubmitting,
    [
      classifierFile,
      evaluationMode,
      isSubmitting,
      labelsFile,
      teamName,
      yoloFile,
    ],
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!yoloFile) {
      setToast({ type: "error", message: "Please attach yolo.onnx." });
      return;
    }

    if (evaluationMode === "brand" && (!classifierFile || !labelsFile)) {
      setToast({
        type: "error",
        message: "Brand mode requires classifier.onnx and labels.json.",
      });
      return;
    }

    setIsSubmitting(true);
    try {
      const submission = await submitModel({
        evaluationMode,
        teamName,
        description,
        yoloGflops,
        classGflops: evaluationMode === "brand" ? classGflops : "",
        yoloFile,
        classifierFile,
        classifierExternalDataFiles:
          evaluationMode === "brand" ? classifierExternalDataFiles : [],
        labelsFile,
      });

      setToast({
        type: "success",
        message: submission.evaluation_task_id
          ? `${submission.team_name ?? teamName} submission #${submission.id} queued. YOLO ${submission.yolo_gflops.toFixed(4)} GFLOPs.`
          : `${submission.team_name ?? teamName} submission #${submission.id} created. YOLO ${submission.yolo_gflops.toFixed(4)} GFLOPs.`,
      });
      setTeamName("");
      setDescription("");
      setYoloGflops("");
      setClassGflops("");
      setYoloFile(null);
      setClassifierFile(null);
      setClassifierExternalDataFiles([]);
      setLabelsFile(null);
      setFileInputKey((value) => value + 1);
    } catch (error) {
      setToast({
        type: "error",
        message: error instanceof Error ? error.message : "Submission failed.",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
      <Toast toast={toast} onClose={() => setToast(null)} />

      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-sm font-semibold text-slate-600 hover:text-slate-950"
          >
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Leaderboard
          </Link>
          <h1 className="mt-3 text-3xl font-semibold tracking-normal text-slate-950 sm:text-4xl">
            Submit Model
          </h1>
        </div>
      </div>

      <form
        onSubmit={handleSubmit}
        className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm sm:p-6"
      >
        <div className="mb-6 grid gap-3 sm:grid-cols-2">
          <ModeButton
            active={evaluationMode === "type"}
            title="Type Mode"
            description="YOLO only, score by car/truck/bus/motorcycle."
            onClick={() => {
              setEvaluationMode("type");
              setClassGflops("");
              setClassifierFile(null);
              setClassifierExternalDataFiles([]);
              setLabelsFile(null);
              setFileInputKey((value) => value + 1);
            }}
          />
          <ModeButton
            active={evaluationMode === "brand"}
            title="Brand Mode"
            description="YOLO + classifier, score by type/brand."
            onClick={() => setEvaluationMode("brand")}
          />
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <Field label="Team Name">
            <input
              value={teamName}
              onChange={(event) => setTeamName(event.target.value)}
              type="text"
              maxLength={255}
              required
              className="h-11 w-full rounded-md border border-slate-300 px-3 text-sm outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-200"
              placeholder="Team A"
            />
          </Field>

          <Field label="YOLO GFLOPs (optional)">
            <div className="relative">
              <Gauge
                className="pointer-events-none absolute left-3 top-3 h-5 w-5 text-slate-400"
                aria-hidden="true"
              />
              <input
                value={yoloGflops}
                onChange={(event) => setYoloGflops(event.target.value)}
                type="number"
                min="0"
                step="0.0001"
                className="h-11 w-full rounded-md border border-slate-300 pl-10 pr-3 text-sm outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-200"
                placeholder="Auto"
              />
            </div>
          </Field>

          {evaluationMode === "brand" ? (
            <Field label="Classifier GFLOPs (optional)">
              <div className="relative">
                <Cpu
                  className="pointer-events-none absolute left-3 top-3 h-5 w-5 text-slate-400"
                  aria-hidden="true"
                />
                <input
                  value={classGflops}
                  onChange={(event) => setClassGflops(event.target.value)}
                  type="number"
                  min="0"
                  step="0.0001"
                  className="h-11 w-full rounded-md border border-slate-300 pl-10 pr-3 text-sm outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-200"
                  placeholder="Auto"
                />
              </div>
            </Field>
          ) : (
            <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-600">
              Classifier is not used in Type Mode.
            </div>
          )}
        </div>

        <div className="mt-4">
          <Field label="Model Description (optional)">
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              maxLength={1500}
              rows={4}
              className="w-full resize-y rounded-md border border-slate-300 px-3 py-3 text-sm outline-none focus:border-slate-900 focus:ring-2 focus:ring-slate-200"
              placeholder="Model name, backbone, training data, augmentation, tracking or post-processing techniques..."
            />
          </Field>
        </div>

        <div key={fileInputKey} className="mt-6 grid gap-4 md:grid-cols-3">
          <FileField
            label="yolo.onnx"
            accept=".onnx"
            icon={<UploadCloud className="h-5 w-5" aria-hidden="true" />}
            onChange={setYoloFile}
          />
          {evaluationMode === "brand" ? (
            <>
              <FileField
                label="classifier.onnx"
                accept=".onnx"
                icon={<UploadCloud className="h-5 w-5" aria-hidden="true" />}
                onChange={setClassifierFile}
              />
              <MultiFileField
                label="classifier external data (optional)"
                accept=".data,.bin,application/octet-stream"
                icon={<UploadCloud className="h-5 w-5" aria-hidden="true" />}
                onChange={setClassifierExternalDataFiles}
              />
              <FileField
                label="labels.json"
                accept=".json,application/json"
                icon={<FileJson className="h-5 w-5" aria-hidden="true" />}
                onChange={setLabelsFile}
              />
            </>
          ) : (
            <FileField
              label="labels.json (optional)"
              accept=".json,application/json"
              icon={<FileJson className="h-5 w-5" aria-hidden="true" />}
              required={false}
              onChange={setLabelsFile}
            />
          )}
        </div>

        <div className="mt-6 flex flex-col-reverse gap-3 border-t border-slate-200 pt-5 sm:flex-row sm:items-center sm:justify-end">
          <Link
            href="/"
            className="inline-flex h-11 items-center justify-center rounded-md border border-slate-300 bg-white px-4 text-sm font-semibold text-slate-700 hover:bg-slate-100"
          >
            Cancel
          </Link>
          <button
            type="submit"
            disabled={!canSubmit}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-slate-950 px-5 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isSubmitting ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            ) : (
              <Send className="h-4 w-4" aria-hidden="true" />
            )}
            Submit Model
          </button>
        </div>
      </form>
    </main>
  );
}

function ModeButton({
  active,
  title,
  description,
  onClick,
}: {
  active: boolean;
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border p-4 text-left transition ${
        active
          ? "border-slate-950 bg-slate-950 text-white"
          : "border-slate-200 bg-white text-slate-800 hover:bg-slate-50"
      }`}
    >
      <span className="block text-sm font-semibold">{title}</span>
      <span className={`mt-1 block text-xs ${active ? "text-slate-200" : "text-slate-500"}`}>
        {description}
      </span>
    </button>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-semibold text-slate-700">{label}</span>
      {children}
    </label>
  );
}

function FileField({ label, accept, icon, required = true, onChange }: FileFieldProps) {
  const [fileName, setFileName] = useState("");

  return (
    <label className="flex min-h-36 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-5 text-center hover:border-slate-500 hover:bg-white">
      <span className="flex h-10 w-10 items-center justify-center rounded-md bg-white text-slate-700 shadow-sm">
        {icon}
      </span>
      <span className="mt-3 text-sm font-semibold text-slate-800">{label}</span>
      <span className="mt-1 max-w-full truncate text-xs text-slate-500">
        {fileName || "Choose file"}
      </span>
      <input
        type="file"
        accept={accept}
        required={required}
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0] ?? null;
          setFileName(file?.name ?? "");
          onChange(file);
        }}
      />
    </label>
  );
}

function MultiFileField({
  label,
  accept,
  icon,
  onChange,
}: {
  label: string;
  accept: string;
  icon: React.ReactNode;
  onChange: (files: File[]) => void;
}) {
  const [fileNames, setFileNames] = useState("");

  return (
    <label className="flex min-h-36 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-5 text-center hover:border-slate-500 hover:bg-white">
      <span className="flex h-10 w-10 items-center justify-center rounded-md bg-white text-slate-700 shadow-sm">
        {icon}
      </span>
      <span className="mt-3 text-sm font-semibold text-slate-800">{label}</span>
      <span className="mt-1 max-w-full truncate text-xs text-slate-500">
        {fileNames || "Choose file(s)"}
      </span>
      <input
        type="file"
        accept={accept}
        multiple
        className="sr-only"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          setFileNames(files.map((file) => file.name).join(", "));
          onChange(files);
        }}
      />
    </label>
  );
}
