"""Anthropic Message Batches API wrapper.

Handles submitting, polling, collecting results, and persisting state
so interrupted batch runs can be resumed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

_MAX_BATCH_SIZE = 100_000


@dataclass
class BatchRequestItem:
    """A single request destined for the Anthropic batch API."""

    custom_id: str
    params: dict[str, Any]


@dataclass
class BatchResultItem:
    """A single result returned from a completed batch."""

    custom_id: str
    result_type: str  # "succeeded", "errored", "expired", "canceled"
    text: str = ""
    error: str | None = None
    stop_reason: str | None = None


@dataclass
class BatchState:
    """Tracks active/completed batch IDs for resume support."""

    batch_ids: list[str] = field(default_factory=list)
    phase: str = ""
    completed: bool = False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "batch_ids": self.batch_ids,
                    "phase": self.phase,
                    "completed": self.completed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BatchState:
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            batch_ids=raw.get("batch_ids", []),
            phase=raw.get("phase", ""),
            completed=raw.get("completed", False),
        )


def _chunk_requests(
    requests: list[BatchRequestItem], max_size: int = _MAX_BATCH_SIZE
) -> list[list[BatchRequestItem]]:
    """Split requests into chunks respecting the Anthropic batch size limit."""
    if not requests:
        return []
    chunks: list[list[BatchRequestItem]] = []
    for i in range(0, len(requests), max_size):
        chunks.append(requests[i : i + max_size])
    return chunks


def submit_batches(
    requests: list[BatchRequestItem],
    *,
    client: anthropic.Anthropic | None = None,
    echo: callable = print,
) -> list[str]:
    """Submit requests as one or more Anthropic Message Batches.

    Returns a list of batch IDs (one per chunk).
    """
    if client is None:
        client = anthropic.Anthropic()

    chunks = _chunk_requests(requests)
    batch_ids: list[str] = []

    for i, chunk in enumerate(chunks):
        echo(f"  Submitting batch chunk {i + 1}/{len(chunks)} ({len(chunk)} requests)...")
        api_requests = [
            {"custom_id": item.custom_id, "params": item.params} for item in chunk
        ]
        batch = client.messages.batches.create(requests=api_requests)
        batch_ids.append(batch.id)
        echo(f"  Batch {batch.id} created (status: {batch.processing_status})")

    return batch_ids


def poll_batches(
    batch_ids: list[str],
    *,
    client: anthropic.Anthropic | None = None,
    poll_interval: float = 30.0,
    echo: callable = print,
) -> None:
    """Block until all batches reach a terminal state (ended)."""
    if client is None:
        client = anthropic.Anthropic()

    pending = set(batch_ids)
    while pending:
        time.sleep(poll_interval)
        still_pending: set[str] = set()
        for bid in pending:
            batch = client.messages.batches.retrieve(bid)
            status = batch.processing_status
            counts = batch.request_counts
            total = counts.processing + counts.succeeded + counts.errored + counts.expired + counts.canceled
            done = counts.succeeded + counts.errored + counts.expired + counts.canceled
            if status == "ended":
                echo(
                    f"  Batch {bid}: ended "
                    f"({counts.succeeded} succeeded, {counts.errored} errored, "
                    f"{counts.expired} expired, {counts.canceled} canceled)"
                )
            else:
                echo(f"  Batch {bid}: {status} ({done}/{total} complete)")
                still_pending.add(bid)
        pending = still_pending


def collect_results(
    batch_ids: list[str],
    *,
    client: anthropic.Anthropic | None = None,
    echo: callable = print,
) -> list[BatchResultItem]:
    """Stream results from all completed batches, returning a flat list."""
    if client is None:
        client = anthropic.Anthropic()

    results: list[BatchResultItem] = []
    for bid in batch_ids:
        echo(f"  Collecting results from batch {bid}...")
        for entry in client.messages.batches.results(bid):
            cid = entry.custom_id
            result = entry.result
            if result.type == "succeeded":
                text_parts = []
                for block in result.message.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                results.append(BatchResultItem(
                    custom_id=cid,
                    result_type="succeeded",
                    text="".join(text_parts),
                    stop_reason=result.message.stop_reason,
                ))
            elif result.type == "errored":
                err_msg = str(result.error) if result.error else "unknown error"
                results.append(BatchResultItem(
                    custom_id=cid,
                    result_type="errored",
                    error=err_msg,
                ))
            else:
                results.append(BatchResultItem(
                    custom_id=cid,
                    result_type=result.type,
                    error=f"request {result.type}",
                ))

    succeeded = sum(1 for r in results if r.result_type == "succeeded")
    failed = len(results) - succeeded
    echo(f"  Collected {len(results)} results ({succeeded} succeeded, {failed} failed)")
    return results


def submit_poll_collect(
    requests: list[BatchRequestItem],
    *,
    client: anthropic.Anthropic | None = None,
    state_path: Path | None = None,
    phase_name: str = "",
    poll_interval: float = 30.0,
    echo: callable = print,
) -> list[BatchResultItem]:
    """Full lifecycle: submit → poll → collect, with optional state persistence."""
    if client is None:
        client = anthropic.Anthropic()

    if not requests:
        echo(f"  No requests to submit for {phase_name or 'batch'}.")
        return []

    state = BatchState(phase=phase_name)

    echo(f"\n=== {phase_name or 'Batch'}: submitting {len(requests)} requests ===")
    batch_ids = submit_batches(requests, client=client, echo=echo)
    state.batch_ids = batch_ids
    if state_path:
        state.save(state_path)

    echo(f"\n=== {phase_name or 'Batch'}: polling for completion ===")
    poll_batches(batch_ids, client=client, poll_interval=poll_interval, echo=echo)

    echo(f"\n=== {phase_name or 'Batch'}: collecting results ===")
    results = collect_results(batch_ids, client=client, echo=echo)

    state.completed = True
    if state_path:
        state.save(state_path)

    return results


def resume_collect(
    state_path: Path,
    *,
    client: anthropic.Anthropic | None = None,
    poll_interval: float = 30.0,
    echo: callable = print,
) -> list[BatchResultItem]:
    """Resume from a saved batch state: poll remaining, then collect."""
    if client is None:
        client = anthropic.Anthropic()

    state = BatchState.load(state_path)
    if not state.batch_ids:
        echo("  No batch IDs found in state file.")
        return []
    if state.completed:
        echo(f"  Phase '{state.phase}' already completed, collecting results...")
        return collect_results(state.batch_ids, client=client, echo=echo)

    echo(f"  Resuming phase '{state.phase}' with {len(state.batch_ids)} batches...")
    poll_batches(state.batch_ids, client=client, poll_interval=poll_interval, echo=echo)
    results = collect_results(state.batch_ids, client=client, echo=echo)

    state.completed = True
    state.save(state_path)
    return results
