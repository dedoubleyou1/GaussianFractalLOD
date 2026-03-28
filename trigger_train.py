"""Trigger training on a deployed Modal app.

Usage:
    # First deploy (one time):
    modal deploy modal_train.py

    # Then trigger runs:
    python trigger_train.py --run-name test1 --max-levels 12

    # With config overrides:
    python trigger_train.py --run-name test2 --max-levels 12 \
        --set reg_scale_weight=0.01 child_opacity_scale=0.2

    # Resume from checkpoint:
    python trigger_train.py --run-name test1 --max-levels 12 \
        --resume /checkpoints/lego_test1/phase2_level_10.pt

    # Check status:
    modal app list
"""

import argparse
import modal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-name", default="gaussianfractallod")
    parser.add_argument("--scene", default="lego")
    parser.add_argument("--num-roots", type=int, default=1)
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--max-levels", type=int, default=12)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--set", nargs="*", default=[],
        help="Config overrides as key=value pairs, e.g. --set reg_scale_weight=0.01",
    )
    args = parser.parse_args()

    # Parse config overrides
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        # Try to parse as number
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string
        overrides[key] = value

    train_fn = modal.Function.from_name(args.app_name, "train")

    print(f"Spawning train on deployed app '{args.app_name}'...")
    print(f"  run_name={args.run_name}")
    print(f"  max_levels={args.max_levels}")
    if args.resume:
        print(f"  resume={args.resume}")
    if overrides:
        print(f"  config_overrides={overrides}")

    call = train_fn.spawn(
        scene=args.scene,
        num_roots=args.num_roots,
        sh_degree=args.sh_degree,
        max_levels=args.max_levels,
        resume_from=args.resume,
        auto_eval=not args.no_eval,
        run_name=args.run_name,
        config_overrides=overrides if overrides else None,
    )

    print(f"\nSpawned! Function call ID: {call.object_id}")
    print(f"Results persist for 7 days.")
    print(f"Monitor checkpoints: modal volume ls gflod-checkpoints lego_{args.run_name}/")


if __name__ == "__main__":
    main()
