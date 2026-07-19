"""Run a minimal FMCW-based robotic obstacle-avoidance simulation."""

from __future__ import annotations

import matplotlib.pyplot as plt

from iscir import FMCWSensor, Obstacle, ReactiveController, RobotState, Scene


def main() -> None:
    scene = Scene(
        robot=RobotState(x=0.0, y=0.0, heading=0.0),
        goal=(10.0, 0.0),
        obstacles=[
            Obstacle(x=3.0, y=0.1, radius=0.45),
            Obstacle(x=6.0, y=-0.8, radius=0.5),
            Obstacle(x=7.5, y=1.0, radius=0.4, vy=-0.05),
        ],
    )
    sensor = FMCWSensor(range_std_m=0.04, velocity_std_mps=0.03, rng_seed=4)
    controller = ReactiveController()

    dt = 0.1
    trajectory_x = [scene.robot.x]
    trajectory_y = [scene.robot.y]

    for _ in range(1000):
        detections = sensor.scan(scene)
        linear_speed, angular_speed = controller.command(scene.robot, scene.goal, detections)
        scene.robot.step(linear_speed, angular_speed, dt)
        scene.step_obstacles(dt)
        trajectory_x.append(scene.robot.x)
        trajectory_y.append(scene.robot.y)

        if scene.has_collision():
            print("Simulation stopped: collision detected.")
            break
        if scene.distance_to_goal() < 0.35:
            print("Goal reached successfully.")
            break
    else:
        print("Simulation stopped: maximum duration reached.")

    plt.plot(trajectory_x, trajectory_y, label="robot trajectory")
    plt.scatter([scene.goal[0]], [scene.goal[1]], marker="*", s=140, label="goal")
    for obstacle in scene.obstacles:
        circle = plt.Circle((obstacle.x, obstacle.y), obstacle.radius, fill=False)
        plt.gca().add_patch(circle)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("ISCIR FMCW Robotic Obstacle Avoidance")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
