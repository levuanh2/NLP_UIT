"""Local Transformers cross-encoder reranker."""

import gc
import os
from typing import Any

from app.domain.retrieval import RetrievalCandidate
from app.indexing.metadata_store.repository import LegalRepository
from app.retrieval.reranking.base import BaseReranker


class VietnameseReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str,
        local_files_only: bool,
        trust_remote_code: bool,
        repository: LegalRepository | None = None,
        batch_size: int = 16,
        max_length: int = 2304,
        parameter_budget_approved: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.repository = repository
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._parameter_budget_verified = parameter_budget_approved
        self._parameter_budget_approved = parameter_budget_approved
        # E5 instrumentation, observational only: the last rerank() call
        # publishes every score it computed here, including for candidates that
        # were then truncated away. Nothing in the ranking path reads these.
        self.last_scores: dict[str, float] = {}
        self.last_input_order: list[str] = []

    def load(self) -> None:
        if not self.local_files_only:
            raise RuntimeError("Reranker must use local model files only")
        if self.trust_remote_code:
            raise RuntimeError("Remote reranker code is disabled for this pipeline")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=False,
        )
        # ponytail: fp16 only on CUDA; CPU fp16 is slower than fp32, not faster
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
        parameter_count = sum(
            parameter.numel() for parameter in self._model.parameters()
        )
        if parameter_count >= 4_000_000_000:
            self.unload()
            raise RuntimeError(
                f"Reranker exceeds the 4B competition ceiling: {parameter_count:,}"
            )
        self._model.to(self.device)
        self._model.eval()

    def verify_parameter_budget(self, total_parameter_count: int) -> None:
        """Enable inference only after checking all configured model parameters."""
        if total_parameter_count <= 0:
            raise ValueError("Total configured model parameter count must be positive")
        if total_parameter_count >= 4_000_000_000:
            raise RuntimeError(
                "Configured embedding + reranker + LLM models exceed the 4B ceiling: "
                f"{total_parameter_count:,}"
            )
        self._parameter_budget_verified = True

    def _scoring_text(self, candidate: RetrievalCandidate) -> str:
        """Score the same text dense retrieval embedded, hierarchy line included.

        The stored embedding_text prefixes a chunk with its document name and
        legal path, which is where a document number lives; the bare chunk text
        almost never repeats it. Scoring the bare text meant the reranker judged
        relevance blind to which Điều the chunk belongs to. Falls back to the
        chunk text when embedding_text is absent, so an index built before this
        column existed still reranks.
        """
        child = None
        if self.repository is not None:
            child = self.repository.get_child(candidate.child_id)
        enriched = getattr(child, "embedding_text", None) if child else None
        if enriched:
            return enriched
        text = candidate.text or (child.text if child else None)
        if not text:
            raise RuntimeError(
                f"Reranker could not resolve child text: {candidate.child_id}"
            )
        return text

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int = 10,
    ) -> list[RetrievalCandidate]:
        if top_k <= 0 or not candidates:
            return []
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Local reranker is not loaded; call load() after verifying the "
                "configured embedding + reranker + LLM parameter budget"
            )
        if not self._parameter_budget_verified:
            raise RuntimeError(
                "Reranker inference is disabled until verify_parameter_budget() "
                "confirms the configured embedding + reranker + LLM total is below 4B"
            )
        texts = [self._scoring_text(candidate) for candidate in candidates]

        scored: list[tuple[RetrievalCandidate, float]] = []
        import torch

        with torch.inference_mode():
            for start in range(0, len(candidates), self.batch_size):
                batch_candidates = candidates[start : start + self.batch_size]
                batch_texts = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [query] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    values = logits[:, 0]
                elif logits.ndim == 2 and logits.shape[1] >= 2:
                    values = logits[:, -1]
                else:
                    raise RuntimeError(
                        f"Unsupported reranker logits shape: {logits.shape}"
                    )
                scored.extend(
                    zip(batch_candidates, values.float().cpu().tolist(), strict=True)
                )
        scored.sort(key=lambda item: (-item[1], item[0].child_id))

        # E5: record the full score table before truncation, so a diagnostic can
        # ask what score a dropped candidate actually received instead of
        # inferring it from rank. Writes only; never read below.
        self.last_scores = {
            candidate.child_id: float(value) for candidate, value in scored
        }
        self.last_input_order = [candidate.child_id for candidate in candidates]

        # E2: optionally fuse the reranker's ordering with the fusion ordering it
        # was handed, instead of discarding the latter. Set RERANKER_BLEND_RRF to
        # the reciprocal-rank constant (60 matches rrf_k) to enable; unset or 0
        # keeps the reranker authoritative, which is the baseline behaviour.
        # Ranks are fused rather than raw scores because a cross-encoder logit
        # and an RRF score share no scale.
        blend_k = int(os.environ.get("RERANKER_BLEND_RRF", "0") or 0)
        if blend_k > 0:
            fusion_rank = {
                candidate.child_id: position
                for position, candidate in enumerate(candidates, start=1)
            }
            rerank_rank = {
                candidate.child_id: position
                for position, (candidate, _) in enumerate(scored, start=1)
            }
            scored = sorted(
                scored,
                key=lambda item: (
                    -(
                        1.0 / (blend_k + rerank_rank[item[0].child_id])
                        + 1.0 / (blend_k + fusion_rank[item[0].child_id])
                    ),
                    item[0].child_id,
                ),
            )

        return [
            candidate.model_copy(
                update={
                    "score": float(score),
                    "source": "reranker",
                    "rank": rank,
                    "rerank_score": float(score),
                }
            )
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        ]

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._parameter_budget_verified = self._parameter_budget_approved
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
