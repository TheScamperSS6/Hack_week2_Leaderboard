import ast
import csv
import math
import re
from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import Select, case, func, literal, select
from sqlalchemy.orm import Session

from app.models import EvaluationMetadata, Submission


QUESTION_ID_COLUMN = "Question ID"
CCTV_ID_COLUMN = "CCTV ID"
TIME_RANGE_COLUMN = "Time Range"
QUERY_COLUMN = "Query"
ANSWER_COLUMN = "Answer"
FALLBACK_CAR_BRAND = "Toyota"


def parse_timestamp_to_seconds(value: str | int | float) -> float:
    if isinstance(value, int | float):
        return float(value)

    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")

    separator = ":" if ":" in text else "."
    parts = text.split(separator)
    if len(parts) == 1:
        return float(parts[0])

    if len(parts) == 2:
        if separator == "." and len(parts[1]) != 2:
            return float(text)
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"unsupported timestamp format: {value}")


def parse_time_range(value: str) -> tuple[float, float]:
    parts = re.split(r"\s*(?:-|\u2013|\u2014)\s*", str(value).strip(), maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"unsupported time range format: {value}")

    start_seconds = parse_timestamp_to_seconds(parts[0])
    end_seconds = parse_timestamp_to_seconds(parts[1])
    if end_seconds < start_seconds:
        raise ValueError(f"time range end is before start: {value}")
    return start_seconds, end_seconds


def group_fields_from_query(query: str) -> tuple[str, ...]:
    text = str(query).casefold()
    brand_word = "\u0e22\u0e35\u0e48\u0e2b\u0e49\u0e2d"

    if "type/brand" in text or ("type" in text and "brand" in text):
        return ("type_or_brand",)

    if brand_word in text or "brand" in text:
        return ("brand",)

    return ("vehicle_type",)


def normalize_answer_key(value: object) -> str:
    if value is None:
        return "null"

    if isinstance(value, tuple | list):
        parts = [normalize_answer_key(part) for part in value]
        return "_".join(part for part in parts if part)

    text = str(value).strip().strip("\"'")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",").replace(", ", ",")
    if "," in text:
        text = "_".join(part.strip().strip("\"'") for part in text.split(",") if part.strip())
    return text.casefold()


def make_group_key(values: Iterable[object]) -> str:
    return "_".join(normalize_answer_key(value) for value in values)


def parse_answer(answer: str) -> dict[str, int]:
    text = str(answer).strip()
    if not text or text in {"[]", "{}", "-"}:
        return {}

    parsed = _parse_answer_with_literal_eval(text)
    if parsed is not None:
        return parsed

    result: dict[str, int] = {}
    consumed_spans: list[tuple[int, int]] = []
    tuple_pattern = re.compile(
        r"\(\s*(?P<first>[^,():\[\]]+?)\s*,\s*(?P<second>[^():\[\]]+?)\s*\)\s*:\s*(?P<count>-?\d+)",
    )
    for match in tuple_pattern.finditer(text):
        key = make_group_key((match.group("first"), match.group("second")))
        result[key] = int(match.group("count"))
        consumed_spans.append(match.span())

    remainder = _remove_spans(text, consumed_spans)
    simple_pattern = re.compile(
        r"[\"']?(?P<key>[^\"':,\[\]\{\}\(\)]+?)[\"']?\s*:\s*(?P<count>-?\d+)",
    )
    for match in simple_pattern.finditer(remainder):
        key = normalize_answer_key(match.group("key"))
        if key:
            result[key] = int(match.group("count"))

    return result


def _parse_answer_with_literal_eval(text: str) -> dict[str, int] | None:
    candidates = [text]
    if text.startswith("[") and text.endswith("]"):
        candidates.append("{" + text[1:-1] + "}")

    for candidate in candidates:
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue

        if isinstance(value, dict):
            return {
                normalize_answer_key(key): int(count)
                for key, count in value.items()
            }
        if isinstance(value, list):
            merged: dict[str, int] = {}
            for item in value:
                if isinstance(item, dict):
                    merged.update(
                        {
                            normalize_answer_key(key): int(count)
                            for key, count in item.items()
                        }
                    )
            if merged:
                return merged

    return None


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text

    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            chars[index] = " "
    return "".join(chars)


def load_answers_csv(path: str | Path) -> dict[str, dict[str, int]]:
    answers: dict[str, dict[str, int]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            question_id = str(row[QUESTION_ID_COLUMN]).strip()
            answers[question_id] = parse_answer(csv_value(row, ANSWER_COLUMN))
    return answers


def csv_value(row: dict[str, str], column: str) -> str:
    value = row.get(column) or ""
    extra_values = row.get(None) or []
    if column == ANSWER_COLUMN and extra_values:
        value = ",".join([value, *extra_values])
    return value


def prediction_query(
    submission_id: int,
    cctv_id: str,
    start_seconds: float,
    end_seconds: float,
    group_fields: tuple[str, ...],
) -> Select:
    group_columns = [group_column_for_field(field) for field in group_fields]
    stmt = (
        select(
            *group_columns,
            func.count(func.distinct(EvaluationMetadata.track_id)).label("count"),
        )
        .where(EvaluationMetadata.submission_id == submission_id)
        .where(EvaluationMetadata.cctv_id == cctv_id)
        .where(EvaluationMetadata.timestamp >= start_seconds)
        .where(EvaluationMetadata.timestamp <= end_seconds)
    )

    for column in group_columns:
        stmt = stmt.where(column.is_not(None))

    return stmt.group_by(*group_columns)


def group_column_for_field(field: str):
    if field == "type_or_brand":
        return case(
            (
                EvaluationMetadata.vehicle_type == "car",
                func.coalesce(EvaluationMetadata.brand, literal(FALLBACK_CAR_BRAND)),
            ),
            else_=EvaluationMetadata.vehicle_type,
        )
    if field == "brand":
        return case(
            (
                EvaluationMetadata.vehicle_type == "car",
                func.coalesce(EvaluationMetadata.brand, literal(FALLBACK_CAR_BRAND)),
            ),
            else_=EvaluationMetadata.brand,
        )
    return getattr(EvaluationMetadata, field)


def prediction_dict_for_question(
    db: Session,
    submission_id: int,
    cctv_id: str,
    time_range: str,
    query: str,
) -> dict[str, int]:
    start_seconds, end_seconds = parse_time_range(time_range)
    group_fields = group_fields_from_query(query)
    rows = db.execute(
        prediction_query(
            submission_id=submission_id,
            cctv_id=cctv_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            group_fields=group_fields,
        )
    ).all()

    predictions: dict[str, int] = {}
    for row in rows:
        values = row[: len(group_fields)]
        key = make_group_key(values)
        predictions[key] = int(row[-1])
    return predictions


def question_results_from_csv(
    db: Session,
    submission_id: int,
    questions_csv_path: str | Path,
    answers_csv_path: str | Path,
) -> list[dict[str, object]]:
    answers = load_answers_csv(answers_csv_path)
    results: list[dict[str, object]] = []

    with Path(questions_csv_path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            question_id = str(row[QUESTION_ID_COLUMN]).strip()
            cctv_id = str(row[CCTV_ID_COLUMN]).strip()
            time_range = csv_value(row, TIME_RANGE_COLUMN)
            query = csv_value(row, QUERY_COLUMN)
            group_fields = group_fields_from_query(query)
            ground_truth = answers.get(question_id, {})
            prediction = prediction_dict_for_question(
                db=db,
                submission_id=submission_id,
                cctv_id=cctv_id,
                time_range=time_range,
                query=query,
            )

            results.append(
                {
                    "question_id": question_id,
                    "cctv_id": cctv_id,
                    "time_range": time_range,
                    "query": query,
                    "group_by": list(group_fields),
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "acc_score": question_accuracy(prediction, ground_truth),
                }
            )

    return results


def question_accuracy(prediction: dict[str, int], ground_truth: dict[str, int]) -> float:
    return jaccard_accuracy(prediction, ground_truth)


def jaccard_accuracy(prediction: dict[str, int], ground_truth: dict[str, int]) -> float:
    keys = set(prediction) | set(ground_truth)
    if not keys:
        return 100.0

    intersection = 0
    union = 0
    for key in keys:
        pred_count = max(0, prediction.get(key, 0))
        gt_count = max(0, ground_truth.get(key, 0))
        intersection += min(pred_count, gt_count)
        union += max(pred_count, gt_count)

    if union == 0:
        return 100.0
    return intersection / union * 100


def score_submission_from_csv(
    db: Session,
    submission: Submission,
    questions_csv_path: str | Path,
    answers_csv_path: str | Path,
) -> tuple[float, float]:
    answers = load_answers_csv(answers_csv_path)
    question_scores: list[float] = []

    with Path(questions_csv_path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            question_id = str(row[QUESTION_ID_COLUMN]).strip()
            ground_truth = answers.get(question_id, {})
            prediction = prediction_dict_for_question(
                db=db,
                submission_id=submission.id,
                cctv_id=str(row[CCTV_ID_COLUMN]).strip(),
                time_range=csv_value(row, TIME_RANGE_COLUMN),
                query=csv_value(row, QUERY_COLUMN),
            )
            question_scores.append(question_accuracy(prediction, ground_truth))

    total_acc = sum(question_scores) / len(question_scores) if question_scores else 0.0
    eff_denominator = math.log10(submission.yolo_gflops + submission.class_gflops) + 1
    eff_score = total_acc / eff_denominator
    return total_acc, eff_score
