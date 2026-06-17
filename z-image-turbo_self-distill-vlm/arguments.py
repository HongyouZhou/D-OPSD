import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Training")

    #deepspeed
    parser.add_argument("--deepspeed-config", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--enable-gc", action=argparse.BooleanOptionalAction, default=False, help="Enable model gradient checkpointing.")

    # logging:
    parser.add_argument("--output-dir", type=str, default="dopsd-exps")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--diagnostics-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostics-dir", type=str, default=None)
    parser.add_argument("--diagnostics-log-every", type=int, default=1)
    parser.add_argument("--diagnostics-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostics-memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--checkpoint-steps", type=int, default=200000)
    parser.add_argument("--max-train-steps", type=int, default=200000)


    # Gen model
    parser.add_argument("--pretrained_model", type=str, default="z-turbo")
    parser.add_argument("--use-lora",type=float, default=1, help="use if > 1")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--num-training-steps", type=int, default=8, help="number of diffusion steps for training.")
    parser.add_argument(
        "--training-timesteps",
        type=str,
        default=None,
        help=(
            "Optional comma-separated timestep grid in [0, 1000). Length must match "
            "--num-training-steps. Defaults to the upstream 4-step or 8-step grid."
        ),
    )
    parser.add_argument(
        "--teacher-timestep-indices",
        type=str,
        default="all",
        help="Comma-separated zero-based timesteps that receive teacher D-OPSD loss, or 'all'.",
    )
    parser.add_argument(
        "--teacher-timestep-warmup-steps",
        type=int,
        default=0,
        help="Use warmup teacher timestep indices for this many optimizer steps before switching to --teacher-timestep-indices.",
    )
    parser.add_argument(
        "--teacher-timestep-warmup-indices",
        type=str,
        default="all",
        help="Comma-separated zero-based warmup teacher timesteps, or 'all'.",
    )
    parser.add_argument(
        "--teacher-timestep-adaptive-top-k",
        type=int,
        default=0,
        help="If >0, after adaptive warmup select the top-k teacher timesteps from --teacher-timestep-indices.",
    )
    parser.add_argument(
        "--teacher-timestep-adaptive-warmup-steps",
        type=int,
        default=0,
        help="Use all candidate teacher timesteps for this many optimizer steps before adaptive top-k selection.",
    )
    parser.add_argument(
        "--teacher-timestep-adaptive-metric",
        type=str,
        default="loss_x0",
        choices=["loss_x0", "gap_x0_mse", "gap_v_mse"],
        help="EMA score used by adaptive teacher timestep selection.",
    )
    parser.add_argument(
        "--teacher-timestep-adaptive-ema",
        type=float,
        default=0.9,
        help="EMA decay for adaptive teacher timestep scores.",
    )
    parser.add_argument(
        "--teacher-target-mode",
        type=str,
        default="raw",
        choices=[
            "raw",
            "raw_force075",
            "trust_region_trajectory_control",
            "x0_drift_trust_region_trajectory_control",
            "variance_controlled_residual_ema",
            "energy_regularized_mode_seeking",
            "safe_angle_temporal_consensus",
        ],
        help="Teacher target conditioning mode for field matching.",
    )
    parser.add_argument(
        "--teacher-target-domain",
        type=str,
        default="x0",
        choices=["x0", "v"],
        help="Prediction domain for teacher target conditioning.",
    )
    parser.add_argument(
        "--teacher-target-gamma",
        type=float,
        default=1.0,
        help="Residual strength for conditioned teacher targets.",
    )
    parser.add_argument(
        "--teacher-residual-norm-cap-ratio",
        type=float,
        default=None,
        help="Per-batch mean residual norm cap ratio for residual_norm_cap mode.",
    )
    parser.add_argument(
        "--teacher-control-energy-lambda",
        type=float,
        default=0.0,
        help="Energy penalty lambda for trajectory_control teacher targets.",
    )
    parser.add_argument(
        "--teacher-control-roughness-beta",
        type=float,
        default=0.0,
        help="Temporal roughness penalty beta for trajectory_control teacher targets.",
    )
    parser.add_argument(
        "--teacher-control-force-budget-ratio",
        type=float,
        default=1.0,
        help="Mean residual force budget ratio for trust-region teacher targets.",
    )
    parser.add_argument(
        "--teacher-control-trust-tau-delta",
        type=float,
        default=0.0,
        help="Trust-region delta radius relative to the matched-force anchor.",
    )
    parser.add_argument(
        "--teacher-control-anchor-cosine-min",
        type=float,
        default=-1.0,
        help="Minimum cosine similarity between final control and matched-force anchor.",
    )
    parser.add_argument(
        "--teacher-residual-ema-decay",
        type=float,
        default=0.9,
        help="Detached residual cache EMA decay for variance_controlled_residual_ema targets.",
    )
    parser.add_argument(
        "--teacher-residual-innovation-mix",
        type=float,
        default=0.5,
        help="Current residual innovation mix for variance_controlled_residual_ema targets.",
    )
    parser.add_argument(
        "--teacher-cache-case-id",
        type=str,
        default=None,
        help="Stable case id used in variance_controlled_residual_ema cache keys.",
    )
    parser.add_argument("--teacher-mode-eta", type=float, default=0.25)
    parser.add_argument("--teacher-energy-ratio-min-vs-raw", type=float, default=0.90)
    parser.add_argument("--teacher-energy-ratio-max-vs-raw", type=float, default=1.25)
    parser.add_argument("--teacher-mode-min-batch", type=int, default=2)
    parser.add_argument("--teacher-mode-cosine-floor", type=float, default=0.0)
    parser.add_argument("--teacher-matched-force-reference-ratio", type=float, default=0.75)
    parser.add_argument("--teacher-mode-residual-norm-eps", type=float, default=1e-6)
    parser.add_argument("--teacher-f3b-eta-mode", type=float, default=0.25)
    parser.add_argument("--teacher-f3b-raw-cosine-min", type=float, default=0.90)
    parser.add_argument("--teacher-f3b-temporal-smooth-lambda", type=float, default=1.0)
    parser.add_argument("--teacher-f3b-energy-ratio-max-vs-raw", type=float, default=1.0)
    parser.add_argument("--teacher-f3b-bank-size-per-timestep", type=int, default=8)
    parser.add_argument("--teacher-f3b-min-consensus-samples", type=int, default=3)
    parser.add_argument("--teacher-f3b-bank-cosine-floor", type=float, default=0.0)
    parser.add_argument("--teacher-f3b-matched-force-reference-ratio", type=float, default=0.75)
    parser.add_argument("--teacher-f3b-residual-norm-eps", type=float, default=1e-6)
    parser.add_argument(
        "--teacher-target-variant",
        type=str,
        default="raw",
        help="Human-readable teacher target variant name for diagnostics.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.9, help="EMA decay for teacher model.")

    #vae
    parser.add_argument("--vae-dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="VAE precision.")

    # dataset
    parser.add_argument("--data-path-train-jsonl", type=str, default="../data/x.jsonl", help="Path to the training data jsonl file.")
    parser.add_argument("--data-path-test-jsonl", type=str, default="../data/x.jsonl", help="Path to the testing data jsonl file.")
    parser.add_argument("--batch-size", type=int, default=4, help="local batch size.")
    parser.add_argument("--batch-size-test", type=int, default=1, help="local batch size test.")

    # precision
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use-8bit-adam", action=argparse.BooleanOptionalAction, default=False,)

    # optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate-gen", type=float, default=1e-6)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=30)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)



    args = parser.parse_args()

    return args
