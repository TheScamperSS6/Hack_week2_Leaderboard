export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api";

const UPLOAD_API_BASE_URL =
  process.env.NEXT_PUBLIC_UPLOAD_API_BASE_URL?.trim() ?? "";

const BACKEND_PORT =
  process.env.NEXT_PUBLIC_BACKEND_PORT?.trim() || "8001";

export type SubmissionStatus =
  | "pending"
  | "processing"
  | "done"
  | "failed"
  | "cancelled";

export type LeaderboardEntry = {
  submission_id: number;
  user_id: number;
  team_name: string;
  status: SubmissionStatus;
  evaluation_mode: "brand" | "type";
  description: string | null;
  error_message: string | null;
  evaluation_task_id: string | null;
  acc_score: number | null;
  eff_score: number | null;
  yolo_gflops: number;
  class_gflops: number;
  created_at: string;
};

export type SubmitModelInput = {
  evaluationMode: "brand" | "type";
  teamName: string;
  description: string;
  yoloGflops: string;
  classGflops: string;
  yoloFile: File;
  classifierFile: File | null;
  classifierExternalDataFiles: File[];
  labelsFile: File | null;
};

export type SubmissionCreated = {
  id: number;
  user_id: number;
  team_name?: string | null;
  status: SubmissionStatus;
  evaluation_mode: "brand" | "type";
  description: string | null;
  error_message: string | null;
  evaluation_task_id: string | null;
  yolo_model_path: string;
  class_model_path: string | null;
  labels_json_path: string | null;
  yolo_gflops: number;
  class_gflops: number;
};

export type QuestionResult = {
  question_id: string;
  cctv_id: string;
  time_range: string;
  query: string;
  group_by: string[];
  prediction: Record<string, number>;
  ground_truth: Record<string, number>;
  acc_score: number;
};

export type PreviewVideo = {
  cctv_id: string;
  video_url: string;
  file_path: string;
  exists: boolean;
};

export type SubmissionResults = {
  submission_id: number;
  user_id: number;
  team_name: string;
  status: SubmissionStatus;
  evaluation_mode: "brand" | "type";
  description: string | null;
  error_message: string | null;
  evaluation_task_id: string | null;
  acc_score: number | null;
  eff_score: number | null;
  metadata_count: number;
  questions_csv_path: string;
  answers_csv_path: string;
  question_results: QuestionResult[];
  preview_videos: PreviewVideo[];
};

export type PreviewGenerationResult = {
  submission_id: number;
  preview_videos: PreviewVideo[];
};

export async function fetchLeaderboard(
  options?: { mode?: "brand" | "type"; sortBy?: "acc" | "eff" },
  signal?: AbortSignal,
): Promise<LeaderboardEntry[]> {
  const params = new URLSearchParams({
    mode: options?.mode ?? "type",
    sort_by: options?.sortBy ?? "acc",
  });
  const response = await fetch(`${API_BASE_URL}/leaderboard?${params.toString()}`, {
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, "Unable to load leaderboard"));
  }

  return response.json();
}

export async function fetchSubmissionResults(
  submissionId: number,
  signal?: AbortSignal,
): Promise<SubmissionResults> {
  const response = await fetch(`${API_BASE_URL}/submissions/${submissionId}/results`, {
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, "Unable to load submission results"));
  }

  return response.json();
}

export async function generateSubmissionPreviews(
  submissionId: number,
): Promise<PreviewGenerationResult> {
  const response = await fetch(`${uploadApiBaseUrl()}/submissions/${submissionId}/previews`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, "Unable to generate preview videos"));
  }

  return response.json();
}

export async function cancelSubmission(submissionId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/submissions/${submissionId}/cancel`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, "Unable to cancel submission"));
  }
}

export async function submitModel(
  input: SubmitModelInput,
): Promise<SubmissionCreated> {
  const formData = new FormData();
  formData.append("evaluation_mode", input.evaluationMode);
  formData.append("team_name", input.teamName);
  if (input.description.trim()) {
    formData.append("description", input.description);
  }
  if (input.yoloGflops.trim()) {
    formData.append("yolo_gflops", input.yoloGflops);
  }
  if (input.classGflops.trim()) {
    formData.append("class_gflops", input.classGflops);
  }
  formData.append("yolo_model", input.yoloFile);
  if (input.classifierFile) {
    formData.append("class_model", input.classifierFile);
  }
  for (const file of input.classifierExternalDataFiles) {
    formData.append("class_external_data", file);
  }
  if (input.labelsFile) {
    formData.append("labels_json", input.labelsFile);
  }

  const response = await fetch(`${uploadApiBaseUrl()}/submit`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, "Submission failed"));
  }

  return response.json();
}

function uploadApiBaseUrl() {
  if (UPLOAD_API_BASE_URL) {
    return trimTrailingSlash(UPLOAD_API_BASE_URL);
  }

  if (API_BASE_URL.startsWith("http://") || API_BASE_URL.startsWith("https://")) {
    return trimTrailingSlash(API_BASE_URL);
  }

  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:${BACKEND_PORT}`;
  }

  return trimTrailingSlash(API_BASE_URL);
}

function trimTrailingSlash(value: string) {
  return value.replace(/\/+$/, "");
}

async function apiErrorMessage(response: Response, fallback: string) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    return fallback;
  }

  return fallback;
}
