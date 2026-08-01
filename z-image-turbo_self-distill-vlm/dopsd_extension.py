"""Small method-extension surface for the shared D-OPSD training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


class TeacherContextProvider(Protocol):
    def context_embeddings(
        self,
        *,
        prompts: Sequence[str],
        optimizer_step: int,
        gradient_accumulation_microstep: int,
        timestep_index: int,
        source_row_ids: Sequence[str] | None = None,
    ) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DopsdExtensionContext:
    dataset_jsonl: Path
    vl_model: Any
    processor: Any
    device: Any
    dtype: Any
    output_dir: str
    is_main_process: bool
    embedding_fn: Any


class DopsdTrainingExtension:
    """Default no-op extension used by the upstream D-OPSD CLI."""

    def teacher_conditioning_mode(self) -> str:
        """How the teacher receives its prompt embeddings.

        ``multimodal`` is the historical D-OPSD path: load Qwen3-VL and build
        image-augmented embeddings.  A method may select ``student_text`` to
        pass the exact same embedding object to teacher and student and avoid
        loading a visual-language model that it cannot use.
        """

        return "multimodal"

    def teacher_adapter_update_mode(self) -> str:
        """How the frozen adapter is updated after an optimizer step."""

        return "ema"

    def student_prompt_keys(self, *, evaluation: bool) -> Sequence[str]:
        if evaluation:
            return (
                "short_en",
                "short_zh",
                "medium_zh",
                "medium_en",
                "user_prompt_en",
                "user_prompt_zh",
            )
        return (
            "short_en",
            "detailed_en",
            "short_zh",
            "detailed_zh",
            "medium_zh",
            "medium_en",
            "user_prompt_en",
            "user_prompt_zh",
        )

    def initialize_adapter_state(self, transformer: Any) -> None:
        return None

    def validate_teacher_adapter_state(
        self,
        transformer: Any,
        *,
        optimizer_step: int,
        event: str,
    ) -> None:
        """Validate a method-owned teacher at initialization/checkpoint seams."""

        return None

    def validate_student_prompts(
        self,
        prompts: Sequence[str],
        source_row_ids: Sequence[str] | None = None,
    ) -> None:
        return None

    def build_teacher_context(
        self,
        context: DopsdExtensionContext,
    ) -> TeacherContextProvider | None:
        return None
