"""Demonstrate tracking-to-control behavior for an approaching radar target."""

from __future__ import annotations

from iscir.navigation import RadarNavigationPipeline
from iscir.sensing.tracker import RadarDetection


def main() -> None:
    pipeline = RadarNavigationPipeline()
    dt_s = 0.5

    print("frame  range  velocity  track  level      command")
    print("-----  -----  --------  -----  ---------  -------")

    for frame, range_m in enumerate((8.0, 7.4, 6.8, 6.2, 5.6, 5.0, 4.4, 3.8, 3.2, 2.6, 2.0)):
        detection = RadarDetection(
            range_m=range_m,
            radial_velocity_mps=-1.2,
            confidence=0.95,
        )
        result = pipeline.update([detection], dt_s=dt_s)
        command = result.command
        track_text = "-" if command.source_track_id is None else str(command.source_track_id)
        print(
            f"{frame:5d}  {range_m:5.1f}  {-1.2:8.1f}  {track_text:>5}  "
            f"{command.level.value:9s}  {command.linear_speed_mps:7.3f}"
        )

    print("\nFinal command:", result.command)


if __name__ == "__main__":
    main()
