#!/usr/bin/env python
"""Generate evaluation images locally with Z-Image-Turbo and optional LoRA weights."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_paths import get_local_image_path, resolve_existing_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Z-Image-Turbo sampling for eval prompts.")
    parser.add_argument("--jsonl-path", required=True, help="Input JSONL from eval_data_gen.py.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. Images are saved under images/ and the JSONL is saved directly here.",
    )
    parser.add_argument("--output-name", required=True, help="Name used for the output JSONL and image folder.")
    parser.add_argument(
        "--class-to-lora",
        default="{}",
        help="JSON object or JSON file mapping class_name to a LoRA path. Empty values use the base model.",
    )
    parser.add_argument("--base-model", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Pipeline dtype.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--default-height", type=int, default=1024)
    parser.add_argument("--default-width", type=int, default=1024)
    parser.add_argument("--target-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--size-multiple", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=73483)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output JSONL if it exists.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on the first prompt or image that does not produce a file, instead of "
            "logging it and continuing. Without this a run that lost images still exits 0 "
            "and writes a JSONL whose empty path slots only a downstream reader notices."
        ),
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: dict[str, Any], output_jsonl_path: Path) -> None:
    with open(output_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_class_to_lora(value: str) -> dict[str, str]:
    candidate_path = Path(value).expanduser()
    if candidate_path.exists():
        with open(candidate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(value)

    if not isinstance(data, dict):
        raise ValueError("--class-to-lora must be a JSON object or a JSON file containing an object")
    return {str(k): str(v) for k, v in data.items()}


def sanitize_name(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def round_to_multiple(value: float, base: int) -> int:
    return max(base, int(round(value / base) * base))


def get_hw_from_image(
    row: dict[str, Any],
    jsonl_path: Path,
    default_height: int,
    default_width: int,
    target_pixels: int,
    size_multiple: int,
) -> tuple[int, int]:
    from PIL import Image

    try:
        image_path = get_local_image_path(row, jsonl_path=jsonl_path, data_root=PROJECT_ROOT)
        with Image.open(image_path) as image:
            width0, height0 = image.size
        if width0 <= 0 or height0 <= 0:
            return default_height, default_width

        ratio = width0 / height0
        width = math.sqrt(target_pixels * ratio)
        height = math.sqrt(target_pixels / ratio)
        return round_to_multiple(height, size_multiple), round_to_multiple(width, size_multiple)
    except Exception as exc:
        print(f"[WARN] Failed to read source aspect ratio: {type(exc).__name__}: {exc}")
        return default_height, default_width


def build_base_pipeline(base_model: str, dtype: Any, device: str) -> Any:
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(base_model, torch_dtype=dtype)
    return pipe.to(device)


def switch_lora(
    pipe: Any,
    base_transformer: Any,
    lora_path: str,
    dtype: Any,
    device: str,
) -> Any:
    import torch
    from peft import PeftModel

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not lora_path.strip():
        pipe.transformer = base_transformer
        print("Switched to base transformer.")
    else:
        print(f"Switching to LoRA: {lora_path}")
        pipe.transformer = PeftModel.from_pretrained(
            base_transformer,
            lora_path,
            torch_dtype=dtype,
        ).to(device)
        # peft upcasts adapter weights to fp32 when it loads them
        # (autocast_adapter_dtype defaults to True), so without this the LoRA
        # branch runs its matmuls in fp32 while the rest of the transformer
        # runs in `dtype` -- two arithmetics inside one forward, and neither of
        # them the one a training-time collector reproduces.
        pipe.transformer.to(dtype)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pipe


def generate_one_image(
    pipe: Any,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
) -> Any:
    import torch

    generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
    return pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device=generator_device).manual_seed(seed),
    ).images[0]


def path_for_jsonl(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return str(path)


def get_prompt_fields(row: dict[str, Any]) -> tuple[str, list[str]]:
    rewrite_prompt = str(row.get("rewrite_prompt_en") or row.get("rewrite_prompt_zh") or "").strip()
    test_prompts_raw = row.get("test_prompts_en")
    if not isinstance(test_prompts_raw, list):
        test_prompts_raw = row.get("test_prompts_zh", [])
    test_prompts = [p.strip() for p in test_prompts_raw if isinstance(p, str) and p.strip()]
    return rewrite_prompt, test_prompts[:4]


def run_sampling(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm

    jsonl_path = resolve_existing_path(args.jsonl_path, PROJECT_ROOT)
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_root = output_dir / "images" / sanitize_name(args.output_name)
    output_jsonl_path = output_dir / f"{sanitize_name(args.output_name)}.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    if output_jsonl_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists. Pass --overwrite to replace it: {output_jsonl_path}")
        output_jsonl_path.unlink()

    rows = load_jsonl(jsonl_path)
    class_to_lora = parse_class_to_lora(args.class_to_lora)
    dtype = dtype_from_name(args.torch_dtype)

    class_to_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        class_name = str(row.get("class_name", "unknown")).strip() or "unknown"
        class_to_rows.setdefault(class_name, []).append(row)

    print("Building base pipeline...")
    pipe = build_base_pipeline(args.base_model, dtype, args.device)
    base_transformer = pipe.transformer
    current_lora_path: str | None = None

    total_count = 0
    for class_name, samples in class_to_rows.items():
        lora_path = class_to_lora.get(class_name, "")
        target_lora_path = lora_path.strip()
        use_lora = bool(target_lora_path)
        model_tag = "lora" if use_lora else "base"

        print(f"Processing class={class_name}, rows={len(samples)}, lora={target_lora_path or '[base]'}")
        if target_lora_path != current_lora_path:
            pipe = switch_lora(pipe, base_transformer, target_lora_path, dtype, args.device)
            current_lora_path = target_lora_path

        for sample_idx, row in enumerate(tqdm(samples, desc=class_name)):
            sample_id = sanitize_name(row.get("_id", sample_idx))
            sample_height, sample_width = get_hw_from_image(
                row=row,
                jsonl_path=jsonl_path,
                default_height=args.default_height,
                default_width=args.default_width,
                target_pixels=args.target_pixels,
                size_multiple=args.size_multiple,
            )

            out_row = dict(row)
            out_row["rewrite_prompt_en_path"] = ""
            out_row["test_prompts_en_paths"] = []

            item_dir = image_root / sanitize_name(class_name) / model_tag / f"item_{sample_id}"
            item_dir.mkdir(parents=True, exist_ok=True)

            rewrite_prompt, test_prompts = get_prompt_fields(row)
            if args.strict and not rewrite_prompt:
                raise ValueError(f"Sample {sample_id} carries no rewrite prompt")
            if args.strict and not test_prompts:
                raise ValueError(f"Sample {sample_id} carries no test prompts")
            if rewrite_prompt:
                seed = args.base_seed + sample_idx * 100
                save_path = item_dir / "rewrite.png"
                try:
                    image = generate_one_image(
                        pipe=pipe,
                        prompt=rewrite_prompt,
                        seed=seed,
                        height=sample_height,
                        width=sample_width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        device=args.device,
                    )
                    image.save(save_path)
                    out_row["rewrite_prompt_en_path"] = path_for_jsonl(save_path, output_dir)
                except Exception as exc:
                    if args.strict:
                        raise RuntimeError(
                            f"Rewrite generation failed for sample {sample_id}"
                        ) from exc
                    print(f"[ERROR] Rewrite generation failed for sample {sample_id}: {exc}")

            for prompt_index, prompt in enumerate(test_prompts):
                seed = args.base_seed + sample_idx * 100 + prompt_index + 1
                save_path = item_dir / f"test_{prompt_index}.png"
                try:
                    image = generate_one_image(
                        pipe=pipe,
                        prompt=prompt,
                        seed=seed,
                        height=sample_height,
                        width=sample_width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        device=args.device,
                    )
                    image.save(save_path)
                    out_row["test_prompts_en_paths"].append(path_for_jsonl(save_path, output_dir))
                except Exception as exc:
                    if args.strict:
                        raise RuntimeError(
                            f"Test generation failed for sample {sample_id}, "
                            f"prompt {prompt_index}"
                        ) from exc
                    print(f"[ERROR] Test generation failed for sample {sample_id}, prompt {prompt_index}: {exc}")
                    out_row["test_prompts_en_paths"].append("")

            append_jsonl(out_row, output_jsonl_path)
            total_count += 1

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Done. Wrote {total_count} rows to {output_jsonl_path}")
    print(f"Images are under {image_root}")


def main() -> None:
    run_sampling(parse_args())


if __name__ == "__main__":
    main()
