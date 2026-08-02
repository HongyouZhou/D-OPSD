# Evaluation Bench (small customized LoRA setting)

## ⛹️‍♂️ Data Format and New Package Installation

We construct the evaluation bench base on training JSONL, in our implementation, each input JSONL row should contain a local image path and English prompt fields.
Relative image paths are resolved against the JSONL directory, the project root,
and the current working directory. (The data format can also be modified according to your specific situation.)

Example:

```json
{
  "class_name": "Millennium",
  "local_path_list": ["dataset/style_Millennium/target_images/00.png"],
  "h*w": "832*1264",
  "short_en": "...",
  "medium_en": "...",
  "detailed_en": "...",
  "user_prompt_en": "..."
}
```

Supported image path keys include `local_path_list`, `local_paths`,
`image_path_list`, `image_paths`, `image_path`, and `path`.

For evaluation, the new packages to install include:

```bash
conda activate dopsd
pip install open_clip_torch lpips
```
## 🏋️‍♀️ 1. Generate Evaluation Prompts
Run:
```bash
python eval/eval_data_gen.py \
  --jsonl-paths dataset/style_Millennium/data.jsonl \
  --output-jsonl-path eval_outputs/prompts.jsonl \
  --qwen-vl-model-path /path/to/Qwen3-VL-8B-Instruct \
  --concept-mode auto \
  --max-items-per-jsonl 4 \
  --overwrite
```

To pass multiple input JSONL files, list them after `--jsonl-paths` separated by
spaces:

```bash
python eval/eval_data_gen.py \
  --jsonl-paths \
    dataset/style_Millennium/data.jsonl \
    dataset/object_dog/data.jsonl \
    dataset/object_teapot/data.jsonl \
  --output-jsonl-path eval_outputs/prompts.jsonl \
  --qwen-vl-model-path Qwen/Qwen3-VL-8B-Instruct \
  --concept-mode auto \
  --overwrite
```

`--concept-mode` can be `object`, `style`, or `auto`. In `auto` mode, rows whose
English prompts contain `[V] style` are treated as style personalization rows.

The output rows include:

- `concept_mode`
- `rewrite_prompt_en`
- `test_prompts_en`

For object personalization, `[V]` is the target subject and the test prompts
change scenes. For style personalization, `[V]` is the target style and the test
prompts change subjects and compositions while preserving the style.

## 🚴‍♂️ 2. Generate Images

We now provide the inference code for Z-Image-Turbo. 
For other models, you can modify the pipeline based on our approach.
```bash
python eval/run_z.py \
  --jsonl-path eval_outputs/prompts.jsonl \
  --output-dir eval_outputs/dreambooth \
  --output-name Millennium \
  --class-to-lora '{"Millennium": "/path/to/lora"}' \
  --base-model Tongyi-MAI/Z-Image-Turbo \
  --num-inference-steps 9 \
  --base-seed 30 \
  --overwrite
```

`--class-to-lora` accepts either a JSON object string or a path to a JSON file.
Use an empty string as a class value to evaluate the base model without LoRA.

--num-inference-steps 9 actually results in 8 DiT forwards

By default a prompt that fails to produce an image is logged and skipped: the
run still exits 0 and the row records an empty path. Pass `--strict` to stop at
the first such failure instead, which is what a comparison across checkpoints
wants — a run that quietly lost images gives each checkpoint a different
denominator.

The LoRA is cast to `--torch-dtype` after it is loaded. peft holds adapter
weights in fp32, so without that cast the LoRA branch runs in fp32 inside an
otherwise-bf16 transformer, and the sampler is not the one a training-time
collector on the same weights reproduces.

`run_z.py` writes:

- `eval_outputs/dreambooth/dreambooth.jsonl`
- `eval_outputs/dreambooth/images/dreambooth/.../*.png`

The JSONL rows include:

- `rewrite_prompt_en_path`
- `test_prompts_en_paths`

## 🏂 3. Score Results
### Note: 
At present, we are unable to release the code related to the metrics computed using the reward model employed in the paper. However, we have implemented the computation of the other metrics reported in the paper. For quality evaluation, open-source alternatives such as [HPSv3](https://github.com/MizzenAI/HPSv3) can be used as substitutes added in our framework.


Run:
```bash
python eval/run_score.py \
  --jsonl-path eval_outputs/dreambooth/Millennium.jsonl \
  --dinov3-repo /path/to/dinov3 \
  --dinov3-weights /path/to/dinov3_weights.pth \
  --clip-pretrained /path/to/open_clip_weights.bin \
  --qwen-vl-model-path Qwen/Qwen3-VL-8B-Instruct
```

By default, `run_score.py` overwrites the input JSONL with metric fields and
writes a summary text file next to it. Use `--output-jsonl-path` and
`--summary-txt-path` to write separate files.

dinov3-repo can be downloaded from [here](https://github.com/facebookresearch/dinov3), dinov3-weights can be downloaded from [here](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) (we use ViT-S+/16 distilled by default), and open_clip_weights can be downloaded from [here](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14/tree/main).

If you only want a subset of metrics, use:
```bash
python eval/run_score.py \
  --jsonl-path eval_outputs/dreambooth/Millennium.jsonl \
  --skip-dino-lpips \
  --skip-clip \
  --qwen-vl-model-path /path/to/Qwen3-VL-8B-Instruct
```


## 🤝🏻 Acknowledgement
We sincerely thank the opensource weights from  [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [DINOv3](https://github.com/facebookresearch/dinov3), [LPIPS](https://github.com/richzhang/perceptualsimilarity), [DFN-CLIP](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14) and so on. 
We only use these weights and data for research purpose.