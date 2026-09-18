from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[\w'-]+", flags=re.UNICODE)
_WS = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class SparseNBModel:
    format_version: int
    feature_dim: int
    alpha: float
    text_key: str
    label_key: str
    class_documents: dict[str, int]
    class_feature_totals: dict[str, int]
    feature_counts: dict[str, dict[int, int]]

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(sorted(self.class_documents))

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        if not self.class_documents:
            raise RuntimeError("Model has no classes.")
        scores = self.score(text)
        winner = max(scores, key=scores.__getitem__)
        return winner, scores

    def score(self, text: str) -> dict[str, float]:
        if not self.class_documents:
            raise RuntimeError("Model has no classes.")
        features = _hashed_features(text, self.feature_dim)
        total_documents = sum(self.class_documents.values())
        class_count = len(self.class_documents)
        scores: dict[str, float] = {}
        for label in self.classes:
            documents = self.class_documents[label]
            prior_denominator = total_documents + self.alpha * class_count
            prior = math.log((documents + self.alpha) / prior_denominator)
            denominator = (
                self.class_feature_totals[label] + self.alpha * self.feature_dim
            )
            counts = self.feature_counts[label]
            score = prior
            for feature_id, frequency in features.items():
                numerator = counts.get(feature_id, 0) + self.alpha
                likelihood = math.log(numerator / denominator)
                score += frequency * likelihood
            scores[label] = score
        return scores

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "text_key": self.text_key,
            "label_key": self.label_key,
            "class_documents": dict(sorted(self.class_documents.items())),
            "class_feature_totals": dict(sorted(self.class_feature_totals.items())),
            "feature_counts": {
                label: {str(key): value for key, value in sorted(counts.items())}
                for label, counts in sorted(self.feature_counts.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SparseNBModel:
        raw_documents = value.get("class_documents")
        raw_totals = value.get("class_feature_totals")
        raw_counts = value.get("feature_counts")
        if not isinstance(raw_documents, dict):
            raise ValueError("class_documents must be an object")
        if not isinstance(raw_totals, dict):
            raise ValueError("class_feature_totals must be an object")
        if not isinstance(raw_counts, dict):
            raise ValueError("feature_counts must be an object")
        documents = {
            str(key): _require_int(count, f"class_documents.{key}")
            for key, count in raw_documents.items()
        }
        totals = {
            str(key): _require_int(count, f"class_feature_totals.{key}")
            for key, count in raw_totals.items()
        }
        counts: dict[str, dict[int, int]] = {}
        for raw_label, raw_features in raw_counts.items():
            if not isinstance(raw_features, dict):
                raise ValueError("feature_counts entries must be objects")
            label = str(raw_label)
            feature_map: dict[int, int] = {}
            for feature_id, count in raw_features.items():
                key = _require_int(feature_id, f"feature_counts.{label}.feature_id")
                feature_map[key] = _require_int(
                    count,
                    f"feature_counts.{label}.{feature_id}",
                )
            counts[label] = feature_map
        if set(documents) != set(totals) or set(documents) != set(counts):
            raise ValueError("Sparse NB model class maps disagree")
        feature_dim = _require_int(value.get("feature_dim", 0), "feature_dim")
        alpha = _require_float(value.get("alpha", 0.0), "alpha")
        format_version = _require_int(value.get("format_version", 1), "format_version")
        if feature_dim < 256:
            raise ValueError("feature_dim must be >= 256")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        return cls(
            format_version=format_version,
            feature_dim=feature_dim,
            alpha=alpha,
            text_key=str(value.get("text_key", "text")),
            label_key=str(value.get("label_key", "class")),
            class_documents=documents,
            class_feature_totals=totals,
            feature_counts=counts,
        )

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> SparseNBModel:
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Sparse NB model must be a JSON object")
        return cls.from_dict({str(key): item for key, item in decoded.items()})


class SparseNBBuilder:
    """Incremental builder used by streaming trainers and bounded adapter heads."""

    def __init__(
        self,
        *,
        text_key: str = "text",
        label_key: str = "class",
        feature_dim: int = 32768,
        alpha: float = 0.5,
    ) -> None:
        if feature_dim < 256:
            raise ValueError("feature_dim must be >= 256")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.text_key = text_key
        self.label_key = label_key
        self.feature_dim = feature_dim
        self.alpha = alpha
        self._class_documents: Counter[str] = Counter()
        self._class_totals: Counter[str] = Counter()
        self._feature_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
        self._examples = 0

    @property
    def example_count(self) -> int:
        return self._examples

    def add(self, text: str, label: str) -> None:
        clean_text = text.strip()
        clean_label = label.strip()
        if not clean_text:
            raise ValueError("training text must be non-empty")
        if not clean_label:
            raise ValueError("training label must be non-empty")
        features = _hashed_features(clean_text, self.feature_dim)
        self._class_documents[clean_label] += 1
        self._class_totals[clean_label] += sum(features.values())
        self._feature_counts[clean_label].update(features)
        self._examples += 1

    def finish(self) -> SparseNBModel:
        if self._examples == 0:
            raise ValueError("Training data is empty")
        return SparseNBModel(
            format_version=1,
            feature_dim=self.feature_dim,
            alpha=self.alpha,
            text_key=self.text_key,
            label_key=self.label_key,
            class_documents=dict(self._class_documents),
            class_feature_totals=dict(self._class_totals),
            feature_counts={
                label: dict(counts) for label, counts in self._feature_counts.items()
            },
        )


def train_sparse_nb(
    records: Iterable[Mapping[str, object]],
    *,
    text_key: str = "text",
    label_key: str = "class",
    feature_dim: int = 32768,
    alpha: float = 0.5,
    cancelled: Callable[[], bool] | None = None,
    on_example: Callable[[int], None] | None = None,
) -> tuple[SparseNBModel, int]:
    builder = SparseNBBuilder(
        text_key=text_key,
        label_key=label_key,
        feature_dim=feature_dim,
        alpha=alpha,
    )
    for record in records:
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancellation requested")
        payload = record.get("payload")
        label_payload = record.get("label")
        if not isinstance(payload, dict) or not isinstance(label_payload, dict):
            raise ValueError("Training rows require object payload and label fields")
        text = payload.get(text_key)
        raw_label = label_payload.get(label_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"payload.{text_key} must be a non-empty string")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(f"label.{label_key} must be a non-empty string")
        builder.add(text, raw_label)
        if on_example is not None:
            on_example(builder.example_count)
    return builder.finish(), builder.example_count


def evaluate_sparse_nb(
    model: SparseNBModel,
    records: Iterable[Mapping[str, object]],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    correct = 0
    total = 0
    for record in records:
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancellation requested")
        payload = record.get("payload")
        label_payload = record.get("label")
        if not isinstance(payload, dict) or not isinstance(label_payload, dict):
            raise ValueError("Evaluation rows require object payload and label fields")
        text = payload.get(model.text_key)
        expected = label_payload.get(model.label_key)
        if not isinstance(text, str) or not isinstance(expected, str):
            raise ValueError("Evaluation row text/label fields are invalid")
        predicted, _ = model.predict(text)
        correct += int(predicted == expected)
        total += 1
    return correct, total


def read_jsonl_records(paths: Sequence[Path]) -> Iterable[dict[str, object]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                decoded = json.loads(line)
                if not isinstance(decoded, dict):
                    raise ValueError(
                        f"{path}:{line_number}: row must be a JSON object"
                    )
                yield {str(key): value for key, value in decoded.items()}


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _require_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _hashed_features(text: str, feature_dim: int) -> Counter[int]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _WS.sub(" ", normalized).strip()
    tokens = _TOKEN.findall(normalized)
    features: Counter[int] = Counter()
    for token in tokens:
        features[_hash_feature(f"w1:{token}", feature_dim)] += 1
    for left, right in zip(tokens, tokens[1:], strict=False):
        features[_hash_feature(f"w2:{left}_{right}", feature_dim)] += 1
    compact = f" {normalized} "
    for width in (3, 4, 5):
        if len(compact) < width:
            continue
        for index in range(len(compact) - width + 1):
            gram = compact[index : index + width]
            features[_hash_feature(f"c{width}:{gram}", feature_dim)] += 1
    return features


def _hash_feature(value: str, feature_dim: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % feature_dim
