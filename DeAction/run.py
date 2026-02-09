#!/usr/bin/env python3
"""Benchmark-driven DeAction systematic analysis entry point."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from DeAction.deaction import DeAction
from DeAction.narrative_summary import NarrativeSummaryGenerator


def _normalize_bool(value) -> Optional[bool]:  # noqa: ANN001 - heterogeneous source data
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "misaligned", "yes"}:
            return True
        if lowered in {"false", "aligned", "no"}:
            return False
    return None


def _ensure_directory(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)


@dataclass
class BenchmarkEntry:
    trajectory_id: str
    payload: Dict


class NarrativeSummaryCache:
    """Lightweight JSON-backed cache for per-step narrative summaries."""

    def __init__(self, cache_file: str, logger: logging.Logger) -> None:
        self.cache_file = cache_file
        self.logger = logger.getChild("NarrativeSummaryCache")
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Dict]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Dict]]:
        if not self.cache_file:
            return {}
        if not os.path.exists(self.cache_file):
            _ensure_directory(self.cache_file)
            with open(self.cache_file, "w", encoding="utf-8") as handle:
                json.dump({}, handle)
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
            self.logger.warning("Narrative cache %s malformed; resetting", self.cache_file)
        except json.JSONDecodeError:
            self.logger.warning("Narrative cache %s is not valid JSON; resetting", self.cache_file)
        except OSError as exc:
            self.logger.error("Unable to read narrative cache %s: %s", self.cache_file, exc)
        return {}

    def _persist(self) -> None:
        if not self.cache_file:
            return
        _ensure_directory(self.cache_file)
        try:
            with self._lock:
                payload = json.dumps(self._cache, indent=2, ensure_ascii=False)
            with open(self.cache_file, "w", encoding="utf-8") as handle:
                handle.write(payload)
        except OSError as exc:
            self.logger.error("Failed to write narrative cache %s: %s", self.cache_file, exc)

    def get(self, trajectory_id: str, step_idx: int) -> Optional[Dict]:
        """Return a cached narrative copy if available."""
        with self._lock:
            trajectory_block = self._cache.get(trajectory_id)
            if not isinstance(trajectory_block, dict):
                return None
            entry = trajectory_block.get(str(step_idx))
            return copy.deepcopy(entry)

    def set(self, trajectory_id: str, step_idx: int, narrative_entry: Dict) -> None:
        """Persist a narrative for reuse."""
        with self._lock:
            trajectory_block = self._cache.setdefault(trajectory_id, {})
            trajectory_block[str(step_idx)] = copy.deepcopy(narrative_entry)
        self._persist()


class BenchmarkRunner:
    def __init__(
        self,
        *,
        benchmark_file: str,
        systematic_analysis_model: str,
        fast_check_model: Optional[str],
        result_file: str,
        use_narrative_summary: bool,
        narrative_summary_model: Optional[str],
        narrative_summary_cache_file: Optional[str],
        model_reasoning_effort: Optional[str] = None,
        fast_check_reasoning_effort: Optional[str] = None,
        max_workers: int,
        annotate_actions: bool,
        add_history_screenshots: bool,
    ) -> None:
        self.benchmark_file = benchmark_file
        self.systematic_analysis_model = systematic_analysis_model
        self.fast_check_model = fast_check_model
        self.result_file = result_file
        self.use_narrative_summary = use_narrative_summary
        self.narrative_summary_model = narrative_summary_model or systematic_analysis_model
        self.model_reasoning_effort = model_reasoning_effort
        self.fast_check_reasoning_effort = fast_check_reasoning_effort
        self.logger = logging.getLogger("BenchmarkRunner")
        self.interrupted = False
        self.max_workers = max_workers if max_workers and max_workers > 0 else 1
        self.results_lock = threading.Lock()
        self.annotate_actions = annotate_actions
        self.add_history_screenshots = add_history_screenshots

        self.systematic_analysis = DeAction(
            model=systematic_analysis_model,
            fast_check_model=fast_check_model,
            model_reasoning_effort=self.model_reasoning_effort,
            fast_check_reasoning_effort=self.fast_check_reasoning_effort,
            max_retry=3,
            annotate_actions=self.annotate_actions,
        )
        self.use_narrative_summary = (
            self.use_narrative_summary and self.systematic_analysis.supports_narrative_memory()
        )
        self.narrative_summary_generator: Optional[NarrativeSummaryGenerator] = None
        if self.use_narrative_summary:
            try:
                self.narrative_summary_generator = NarrativeSummaryGenerator(
                    model=self.narrative_summary_model,
                    logger=self.logger,
                )
            except Exception as exc:
                self.logger.error("Failed to initialize narrative summary generator: %s", exc)
                self.use_narrative_summary = False

        self.narrative_summary_cache: Optional[NarrativeSummaryCache] = None
        if self.use_narrative_summary and narrative_summary_cache_file:
            try:
                self.narrative_summary_cache = NarrativeSummaryCache(narrative_summary_cache_file, self.logger)
            except Exception as exc:
                self.logger.error("Failed to initialize narrative summary cache: %s", exc)
                self.narrative_summary_cache = None

        self.results: Dict[str, Dict] = self._load_existing_results()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ------------------------------------------------------------------
    # Signal handling / persistence
    # ------------------------------------------------------------------
    def _handle_signal(self, signum, frame) -> None:  # noqa: D401, ANN001
        del frame
        self.logger.warning("Received signal %s; attempting graceful shutdown", signum)
        self.interrupted = True
        self._save_results()

    def _load_existing_results(self) -> Dict[str, Dict]:
        if not os.path.exists(self.result_file):
            _ensure_directory(self.result_file)
            with open(self.result_file, "w", encoding="utf-8") as handle:
                json.dump({}, handle, indent=2, ensure_ascii=False)
            return {}

        try:
            with open(self.result_file, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            self.logger.warning("Result file %s is not valid JSON; starting fresh", self.result_file)
        except OSError as exc:
            self.logger.error("Unable to read %s: %s", self.result_file, exc)
        return {}

    def _save_results(self) -> None:
        _ensure_directory(self.result_file)
        try:
            with self.results_lock:
                serialized = json.dumps(self.results, indent=2, ensure_ascii=False)
            with open(self.result_file, "w", encoding="utf-8") as handle:
                handle.write(serialized)
        except OSError as exc:
            self.logger.error("Failed to persist results: %s", exc)

    # ------------------------------------------------------------------
    # Benchmark ingestion helpers
    # ------------------------------------------------------------------
    def _load_benchmark_entries(self) -> List[BenchmarkEntry]:
        if not os.path.exists(self.benchmark_file):
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_file}")

        with open(self.benchmark_file, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

        entries: List[BenchmarkEntry] = []
        iterator: Iterable = raw.items() if isinstance(raw, dict) else enumerate(raw)
        for key, payload in iterator:
            if not isinstance(payload, dict):
                self.logger.warning("Skipping malformed benchmark entry at %s", key)
                continue
            trajectory_id = payload.get("trajectory_id") or str(key)
            entries.append(BenchmarkEntry(trajectory_id=trajectory_id, payload=payload))
        return entries

    def _normalize_completed_steps(self, payload: Dict) -> Dict[int, Dict]:
        raw_steps = payload.get("steps")
        if not raw_steps:
            return {}

        normalized: Dict[int, Dict] = {}
        iterator: Iterable
        if isinstance(raw_steps, dict):
            iterator = raw_steps.items()
        elif isinstance(raw_steps, list):
            iterator = enumerate(raw_steps, start=1)
        else:
            return {}

        for key, step_payload in iterator:
            if not isinstance(step_payload, dict):
                continue
            idx = step_payload.get("step_idx")
            if idx is None:
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
            try:
                normalized[int(idx)] = step_payload
            except (TypeError, ValueError):
                continue
        return normalized

    def _ensure_result_record(self, entry: BenchmarkEntry, trajectory_path: Optional[str]) -> Dict:
        payload = entry.payload
        with self.results_lock:
            record = self.results.get(entry.trajectory_id)
            if record is None:
                record = {
                    "trajectory_id": entry.trajectory_id,
                    "trajectory_path": trajectory_path,
                    "instruction": payload.get("instruction", ""),
                    "steps": {},
                    "total_steps": payload.get("total_steps", 0),
                    "status": "pending",
                    "metadata": payload.get("metadata"),
                }
                self.results[entry.trajectory_id] = record

            record["trajectory_path"] = trajectory_path
            record["instruction"] = payload.get("instruction", record.get("instruction", ""))
            record.setdefault("steps", {})
            record.setdefault("status", "pending")
            if payload.get("metadata"):
                record["metadata"] = payload["metadata"]
            return record

    # ------------------------------------------------------------------
    # Screenshot resolution helpers
    # ------------------------------------------------------------------
    def _normalize_path(self, candidate: Optional[str], base_dir: Optional[str]) -> Optional[str]:
        if not candidate:
            return None
        expanded = os.path.expanduser(candidate)
        if os.path.isabs(expanded) and os.path.exists(expanded):
            return expanded
        if base_dir:
            guess = os.path.join(base_dir, expanded)
            if os.path.exists(guess):
                return guess
        return expanded if os.path.exists(expanded) else None

    def _find_screenshot(self, trajectory_path: Optional[str], step_idx: int) -> Optional[str]:
        if not trajectory_path:
            return None
        pattern = os.path.join(trajectory_path, f"step_{step_idx}*.png")
        matches = sorted(glob.glob(pattern))
        for candidate in matches:
            if self._matches_expected_step(candidate, step_idx):
                return candidate
        return None

    def _collect_history_screenshots(
        self,
        completed_steps: Dict[str, Dict],
        current_step_idx: int,
        trajectory_path: Optional[str],
        limit: int = 100,
    ) -> List[bytes]:
        ordered_steps: List[Tuple[int, Dict]] = []
        for key in sorted(completed_steps, key=lambda v: int(v) if str(v).isdigit() else float("inf")):
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if idx >= current_step_idx:
                continue
            ordered_steps.append((idx, completed_steps[key]))
        if not ordered_steps:
            return []

        history_bytes: List[bytes] = []
        for idx, step in ordered_steps[-limit:]:
            candidate_path = self._normalize_path(step.get("screenshot_path"), trajectory_path)
            if not candidate_path:
                target_idx = idx - 1 if idx > 0 else 0
                candidate_path = self._find_screenshot(trajectory_path, target_idx)
            if not candidate_path:
                candidate_path = self._normalize_path(step.get("post_screenshot_path"), trajectory_path)
            if not candidate_path:
                continue
            screenshot_bytes = self._read_binary(candidate_path)
            if screenshot_bytes:
                history_bytes.append(screenshot_bytes)
        return history_bytes

    def _extract_step_index(self, path: Optional[str]) -> Optional[int]:
        if not path:
            return None
        match = re.search(r"step_(\d+)", os.path.basename(path))
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _matches_expected_step(self, path: Optional[str], expected_idx: int) -> bool:
        return self._extract_step_index(path) == expected_idx

    def _resolve_pre_post_paths(
        self,
        step_idx: int,
        steps: Dict[int, Dict],
        trajectory_path: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        prev_idx = step_idx - 1 if step_idx > 0 else 0
        previous_step = steps.get(prev_idx)
        pre_path = None
        if previous_step:
            candidate = previous_step.get("post_screenshot_path") or previous_step.get("screenshot_path")
            candidate = self._normalize_path(candidate, trajectory_path)
            if candidate and self._matches_expected_step(candidate, prev_idx):
                pre_path = candidate
        if pre_path is None:
            pre_path = self._find_screenshot(trajectory_path, prev_idx)

        current_step = steps.get(step_idx)
        post_path = None
        if current_step:
            for key in ("post_screenshot_path", "screenshot_path"):
                candidate = self._normalize_path(current_step.get(key), trajectory_path)
                if candidate and self._matches_expected_step(candidate, step_idx):
                    post_path = candidate
                    break
        if post_path is None:
            post_path = self._find_screenshot(trajectory_path, step_idx)
        return pre_path, post_path

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def run(self) -> None:
        entries = self._load_benchmark_entries()
        self.logger.info("Loaded %d benchmark trajectories", len(entries))
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {executor.submit(self._process_entry, entry): entry for entry in entries}
                for future in as_completed(future_map):
                    entry = future_map[future]
                    if self.interrupted:
                        break
                    try:
                        future.result()
                    except Exception as exc:
                        self.logger.error("Trajectory %s failed: %s", entry.trajectory_id, exc)
        else:
            for entry in entries:
                if self.interrupted:
                    break
                try:
                    self._process_entry(entry)
                except Exception as exc:
                    self.logger.error("Trajectory %s failed: %s", entry.trajectory_id, exc)
        self._save_results()

    def _process_entry(self, entry: BenchmarkEntry) -> None:
        payload = entry.payload
        trajectory_path = payload.get("trajectory_path")
        if trajectory_path:
            trajectory_path = os.path.expanduser(trajectory_path)
        record = self._ensure_result_record(entry, trajectory_path)

        instruction = record.get("instruction")
        if not instruction:
            self.logger.warning("Skipping %s - missing instruction", entry.trajectory_id)
            return

        normalized_steps = self._normalize_completed_steps(payload)
        if not normalized_steps:
            self.logger.warning("No completed steps for %s", entry.trajectory_id)
            return

        labeled_indices = [
            idx for idx, step in normalized_steps.items() if _normalize_bool(step.get("label")) is not None
        ]
        if not labeled_indices:
            self.logger.warning("Skipping %s - no labeled steps", entry.trajectory_id)
            return

        max_step_idx = max(labeled_indices)
        step_plan = [idx for idx in sorted(normalized_steps) if idx <= max_step_idx]

        fallback_total = payload.get("total_steps") or len(step_plan)
        expected_total = len(step_plan) if step_plan else fallback_total
        record["total_steps"] = expected_total

        previous_actions: List[str] = []
        previous_summaries: List[Dict] = []
        completed_steps = record.get("steps", {})
        for key in sorted(completed_steps, key=lambda v: int(v)):
            step_block = completed_steps[key]
            action = step_block.get("agent_output")
            if action:
                previous_actions.append(action)
            narrative_summary = step_block.get("narrative_summary")
            if narrative_summary:
                previous_summaries.append(narrative_summary)

        for step_idx in step_plan:
            if self.interrupted:
                break
            step_key = str(step_idx)
            if step_key in completed_steps:
                continue
            step_payload = normalized_steps[step_idx]
            agent_output = self._extract_agent_output(step_payload)
            if not agent_output:
                self.logger.warning("Skipping step %s.%s - missing agent output", entry.trajectory_id, step_idx)
                continue

            normalized_label = _normalize_bool(step_payload.get("label"))
            should_eval = normalized_label is not None

            pre_path, post_path = self._resolve_pre_post_paths(step_idx, normalized_steps, trajectory_path)
            pre_bytes = None
            if pre_path and (should_eval or self.use_narrative_summary):
                pre_bytes = self._read_binary(pre_path)

            self.logger.info(
                "Processing %s step %s (label=%s, evaluate=%s)",
                entry.trajectory_id,
                step_idx,
                step_payload.get("label"),
                should_eval,
            )

            systematic_analysis_block: Dict[str, Optional[object]] = {
                "systematic_analysis": "deaction",
                "systematic_analysis_model": self.systematic_analysis_model,
                "fast_check_model": self.fast_check_model,
                "is_misaligned": None,
                "thought": None,
                "time": None,
            }

            if should_eval and pre_bytes is not None:
                history_screenshots: List[bytes] = []
                if self.add_history_screenshots:
                    history_screenshots = self._collect_history_screenshots(
                        completed_steps, step_idx, trajectory_path
                    )

                agent_history = {
                    "observation_type": "screenshot",
                    "thoughts": [""] * len(previous_actions),
                    "actions": previous_actions.copy(),
                    "narratives": previous_summaries.copy(),
                    "includes_current_action": False,
                }
                start = time.time()
                try:
                    obs_payload = {"screenshot": pre_bytes}
                    if history_screenshots:
                        obs_payload["history_screenshots"] = history_screenshots
                    is_misaligned, thought = self.systematic_analysis.check_once(
                        instruction=instruction,
                        response=agent_output,
                        obs=obs_payload,
                        agent_history=agent_history,
                        max_retries=self.systematic_analysis.get_max_retry(),
                    )
                    systematic_analysis_block["is_misaligned"] = is_misaligned
                    systematic_analysis_block["thought"] = thought
                except Exception as exc:
                    systematic_analysis_block["error"] = str(exc)
                    self.logger.error(
                        "Systematic analysis failed for %s step %s: %s",
                        entry.trajectory_id,
                        step_idx,
                        exc,
                    )
                finally:
                    systematic_analysis_block["time"] = time.time() - start
            elif should_eval:
                systematic_analysis_block["error"] = "missing_pre_screenshot"
            else:
                systematic_analysis_block["skipped"] = "missing_label"

            step_result = {
                "step_idx": step_idx,
                "label": normalized_label if normalized_label is not None else step_payload.get("label"),
                "category": step_payload.get("category"),
                "agent_output": agent_output,
                "screenshot_path": pre_path,
                "post_screenshot_path": post_path,
                "systematic_analysis_evaluation": systematic_analysis_block,
            }

            if step_payload.get("metadata"):
                step_result["metadata"] = step_payload["metadata"]

            cached_summary = None
            if self.use_narrative_summary and self.narrative_summary_cache:
                cached_summary = self.narrative_summary_cache.get(entry.trajectory_id, step_idx)
            if cached_summary:
                step_result["narrative_summary"] = cached_summary
                previous_summaries.append(cached_summary)
            elif self.use_narrative_summary and self.narrative_summary_generator and post_path:
                post_bytes = self._read_binary(post_path)
                if post_bytes and pre_bytes:
                    try:
                        preprocess = getattr(self.systematic_analysis, "preprocess_remove_comments", lambda text: text)
                        code_action = preprocess(agent_output)
                        narrative_summary = self.narrative_summary_generator.generate_narrative(
                            user_objective=instruction,
                            action=code_action,
                            step_idx=step_idx,
                            pre_screenshot=pre_bytes,
                            post_screenshot=post_bytes,
                        )
                        step_result["narrative_summary"] = narrative_summary
                        previous_summaries.append(narrative_summary)
                        if self.narrative_summary_cache:
                            self.narrative_summary_cache.set(entry.trajectory_id, step_idx, narrative_summary)
                    except Exception as exc:
                        self.logger.error(
                            "Narrative summary failed for %s step %s: %s",
                            entry.trajectory_id,
                            step_idx,
                            exc,
                        )
                        step_result["narrative_summary_error"] = str(exc)
                elif self.use_narrative_summary:
                    step_result["narrative_summary"] = None

            completed_steps[step_key] = step_result
            previous_actions.append(agent_output)
            self._save_results()

        completed_count = len(record.get("steps", {}))
        if expected_total and completed_count >= expected_total:
            record["status"] = "completed"
        elif completed_count:
            record["status"] = "in_progress"
        else:
            record["status"] = "pending"

    def _extract_agent_output(self, step_payload: Dict) -> Optional[str]:
        for key in ("agent_output", "action", "response", "code"):
            value = step_payload.get(key)
            if value:
                return value if isinstance(value, str) else json.dumps(value)
        return None

    def _read_binary(self, path: Optional[str]) -> Optional[bytes]:
        if not path:
            return None
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except OSError as exc:
            self.logger.error("Unable to read screenshot %s: %s", path, exc)
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeAction systematic analysis on benchmark manifest")
    parser.add_argument("--benchmark_file", required=True, help="Path to benchmark JSON file")
    parser.add_argument("--result_file", default="output/deaction_results.json")
    parser.add_argument("--systematic_analysis_model", default="gpt-5")
    parser.add_argument("--model_reasoning_effort", default=None)
    parser.add_argument("--fast_check_reasoning_effort", default="medium")
    parser.add_argument("--fast_check_model", help="Optional fast check model for DeAction")
    parser.add_argument("--narrative_summary_model", help="Override narrative summary model")
    parser.add_argument(
        "--narrative_summary_cache_file",
        default="output/narrative_summary_cache.json",
        help="Path to reuse cached narrative summary generations",
    )
    parser.add_argument("--no_narrative_summary", action="store_true", help="Disable narrative summary generation")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of trajectories to evaluate in parallel")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument(
        "--annotate_actions",
        action="store_true",
        help="Overlay coordinate annotations on screenshots when supported",
    )
    parser.add_argument(
        "--add_history_screenshots",
        action="store_true",
        help="Attach historical screenshots for evaluations",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    systematic_analysis_model = args.systematic_analysis_model
    if args.fast_check_model and (
        systematic_analysis_model is None
        or (isinstance(systematic_analysis_model, str) and systematic_analysis_model.strip().lower() in ("", "none"))
    ):
        systematic_analysis_model = "none"

    runner = BenchmarkRunner(
        benchmark_file=args.benchmark_file,
        systematic_analysis_model=systematic_analysis_model,
        fast_check_model=args.fast_check_model,
        result_file=args.result_file,
        use_narrative_summary=not args.no_narrative_summary,
        narrative_summary_model=args.narrative_summary_model,
        narrative_summary_cache_file=args.narrative_summary_cache_file,
        model_reasoning_effort=args.model_reasoning_effort,
        fast_check_reasoning_effort=args.fast_check_reasoning_effort,
        max_workers=args.max_workers,
        annotate_actions=args.annotate_actions,
        add_history_screenshots=args.add_history_screenshots,
    )
    runner.run()


if __name__ == "__main__":
    main()
