from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ml_lab.core.models import DatasetSplit

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[\w'-]+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class LeakageExample:
    example_id: str
    split: DatasetSplit
    lineage_group: str
    fingerprint: str
    normalized_fingerprint: str
    near_signature: str


@dataclass(frozen=True, slots=True)
class LeakageIssue:
    kind: str
    left_id: str
    right_id: str
    left_split: DatasetSplit
    right_split: DatasetSplit
    blocking: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_split": self.left_split.value,
            "right_split": self.right_split.value,
            "blocking": self.blocking,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class LeakageReport:
    issues: tuple[LeakageIssue, ...]

    @property
    def blocking_count(self) -> int:
        return sum(issue.blocking for issue in self.issues)

    @property
    def has_blockers(self) -> bool:
        return self.blocking_count > 0

    def to_dict(self) -> dict[str, object]:
        by_kind: dict[str, int] = defaultdict(int)
        for issue in self.issues:
            by_kind[issue.kind] += 1
        return {
            "format_version": 1,
            "blocking_count": self.blocking_count,
            "issue_count": len(self.issues),
            "by_kind": dict(sorted(by_kind.items())),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_fingerprint(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def normalized_payload(payload: object) -> object:
    if isinstance(payload, str):
        text = unicodedata.normalize("NFKC", payload).casefold()
        return _WS.sub(" ", text).strip()
    if isinstance(payload, list):
        return [normalized_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [normalized_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {
            str(key): normalized_payload(value)
            for key, value in sorted(payload.items(), key=lambda pair: str(pair[0]))
        }
    return payload


def normalized_fingerprint(payload: object) -> str:
    normalized = normalized_payload(payload)
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def near_signature(payload: object) -> str:
    text = " ".join(_flatten_text(normalized_payload(payload)))
    tokens = _TOKEN.findall(text)
    joined = " ".join(tokens)
    shingles = _shingles(joined, width=5)
    if not shingles:
        shingles = [joined or canonical_json(normalized_payload(payload))]

    weights = [0] * 64
    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        bits = int.from_bytes(digest, "big")
        for index in range(64):
            weights[index] += 1 if bits & (1 << index) else -1
    signature = 0
    for index, weight in enumerate(weights):
        if weight >= 0:
            signature |= 1 << index
    return f"{signature:016x}"


def scan_leakage(
    examples: Iterable[LeakageExample],
    *,
    near_hamming: int = 3,
) -> LeakageReport:
    """Scan leakage in one pass without materializing the input iterable.

    Indexes grow with the dataset because exact, normalized, lineage, and LSH-near
    matches must remain discoverable, but callers can stream rows directly from SQLite
    rather than duplicating the full dataset in a second Python list first.
    """
    issues: list[LeakageIssue] = []
    emitted: set[tuple[str, str, str]] = set()
    exact_groups: dict[str, list[LeakageExample]] = defaultdict(list)
    normalized_groups: dict[str, list[LeakageExample]] = defaultdict(list)
    lineage_groups: dict[str, list[LeakageExample]] = defaultdict(list)
    bands: dict[tuple[int, int], list[LeakageExample]] = defaultdict(list)
    compared_near: set[tuple[str, str]] = set()

    for example in examples:
        for candidate in exact_groups[example.fingerprint]:
            _emit(
                "EXACT_DUPLICATE",
                candidate,
                example,
                "identical canonical payload hash",
                issues,
                emitted,
            )
        exact_groups[example.fingerprint].append(example)

        for candidate in normalized_groups[example.normalized_fingerprint]:
            if _pair_seen(example, candidate, emitted, {"EXACT_DUPLICATE"}):
                continue
            _emit(
                "NORMALIZED_DUPLICATE",
                candidate,
                example,
                "identical normalized payload hash",
                issues,
                emitted,
            )
        normalized_groups[example.normalized_fingerprint].append(example)

        for candidate in lineage_groups[example.lineage_group]:
            if candidate.split == example.split:
                continue
            _emit(
                "LINEAGE_LEAKAGE",
                candidate,
                example,
                f"lineage group {example.lineage_group!r} crosses partitions",
                issues,
                emitted,
            )
        lineage_groups[example.lineage_group].append(example)

        signature = int(example.near_signature, 16)
        candidates: dict[str, LeakageExample] = {}
        for band in range(4):
            value = (signature >> (band * 16)) & 0xFFFF
            for candidate in bands[(band, value)]:
                candidates[candidate.example_id] = candidate
        for candidate in candidates.values():
            first, second = sorted((example.example_id, candidate.example_id))
            pair = (first, second)
            if pair in compared_near:
                continue
            compared_near.add(pair)
            if _pair_seen(
                example,
                candidate,
                emitted,
                {"EXACT_DUPLICATE", "NORMALIZED_DUPLICATE"},
            ):
                continue
            distance = (signature ^ int(candidate.near_signature, 16)).bit_count()
            if distance <= near_hamming:
                _emit(
                    "NEAR_DUPLICATE",
                    candidate,
                    example,
                    f"64-bit SimHash Hamming distance {distance} <= {near_hamming}",
                    issues,
                    emitted,
                )
        for band in range(4):
            value = (signature >> (band * 16)) & 0xFFFF
            bands[(band, value)].append(example)

    issues.sort(key=lambda item: (not item.blocking, item.kind, item.left_id, item.right_id))
    return LeakageReport(tuple(issues))


def _pair_seen(
    left: LeakageExample,
    right: LeakageExample,
    emitted: set[tuple[str, str, str]],
    kinds: set[str],
) -> bool:
    first, second = sorted((left.example_id, right.example_id))
    return any((kind, first, second) in emitted for kind in kinds)


def _emit(
    kind: str,
    left: LeakageExample,
    right: LeakageExample,
    detail: str,
    issues: list[LeakageIssue],
    emitted: set[tuple[str, str, str]],
) -> None:
    first, second = sorted((left.example_id, right.example_id))
    marker = (kind, first, second)
    if marker in emitted:
        return
    emitted.add(marker)
    issues.append(
        LeakageIssue(
            kind=kind,
            left_id=left.example_id,
            right_id=right.example_id,
            left_split=left.split,
            right_split=right.split,
            blocking=left.split != right.split,
            detail=detail,
        )
    )


def _flatten_text(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for key in sorted(value, key=str):
            result.extend(_flatten_text(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_text(item))
        return result
    if value is None:
        return []
    return [str(value)]


def _shingles(text: str, *, width: int) -> list[str]:
    compact = _WS.sub(" ", text).strip()
    if not compact:
        return []
    if len(compact) <= width:
        return [compact]
    return [compact[index : index + width] for index in range(len(compact) - width + 1)]
