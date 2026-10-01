import argparse
from pathlib import Path

from deepseti.injection.benchmark import (
    generate_all_benchmark_cases,
    load_benchmark_config,
    validate_injection_split,
)
from deepseti.injection.dataset import (
    build_injected_benchmark,
    load_split_frame_pool,
    save_benchmark_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["validation", "test"],
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    config_path = (
        project_root
        / "configs"
        / "injection"
        / "benchmark.json"
    )

    manifest_path = (
        project_root
        / "configs"
        / "data"
        / "observations.csv"
    )

    raw_root = project_root / "data" / "raw"

    output_dir = (
        project_root
        / "data"
        / "synthetic"
        / args.split
    )

    config = load_benchmark_config(config_path)

    validate_injection_split(
        args.split,
        config,
    )

    raw_frames, frame_metadata = load_split_frame_pool(
        manifest_path=manifest_path,
        raw_root=raw_root,
        split=args.split,
    )

    cases = generate_all_benchmark_cases(config)

    clean_frames, injected_frames, signal_masks, metadata = (
        build_injected_benchmark(
            raw_frames=raw_frames,
            frame_metadata=frame_metadata,
            cases=cases,
            seed=config["seed"],
        )
    )

    save_benchmark_dataset(
        output_dir=output_dir,
        clean_frames=clean_frames,
        injected_frames=injected_frames,
        signal_masks=signal_masks,
        metadata=metadata,
    )

    print(f"Split: {args.split}")
    print(f"Real frame pool: {raw_frames.shape}")
    print(f"Benchmark cases: {len(cases)}")
    print(f"Injected frames: {injected_frames.shape}")
    print(f"Saved to: {output_dir}")
    print(f"Clean controls: {clean_frames.shape}")


if __name__ == "__main__":
    main()