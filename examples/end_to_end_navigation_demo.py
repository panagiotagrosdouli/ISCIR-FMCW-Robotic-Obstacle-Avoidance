"""Run the complete detection-to-command pipeline on a synthetic head-on target."""

from dataclasses import asdict
import json
from pathlib import Path

from iscir.evaluation import ScenarioConfig, evaluate_navigation_scenario, head_on_scenario


def main() -> None:
    scenario = head_on_scenario(
        ScenarioConfig(
            frame_count=100,
            detection_probability=0.95,
            range_noise_std_m=0.2,
            velocity_noise_std_mps=0.1,
            clutter_rate_per_frame=0.25,
        )
    )
    metrics = evaluate_navigation_scenario(scenario, seed=42)

    output_path = Path("paper/experiments/generated/navigation_head_on.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"frames={metrics.frame_count}")
    print(f"mean speed={metrics.mean_command_speed_mps:.3f} m/s")
    print(f"minimum speed={metrics.minimum_command_speed_mps:.3f} m/s")
    print(f"reduced-speed fraction={metrics.reduced_speed_fraction:.3f}")
    print(f"emergency stops={metrics.emergency_stop_count}")
    print(f"first intervention frame={metrics.first_intervention_frame}")
    print(f"mean pipeline latency={metrics.mean_pipeline_latency_ms:.3f} ms")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
