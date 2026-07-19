from iscir import FMCWSensor, Obstacle, ReactiveController, RobotState, Scene


def test_sensor_detects_obstacle_ahead() -> None:
    scene = Scene(robot=RobotState(), goal=(5.0, 0.0), obstacles=[Obstacle(x=2.0, y=0.0)])
    sensor = FMCWSensor(range_std_m=0.0, velocity_std_mps=0.0, rng_seed=1)

    detections = sensor.scan(scene)

    assert len(detections) == 1
    assert detections[0].range_m == 2.0


def test_controller_turns_for_close_frontal_obstacle() -> None:
    scene = Scene(robot=RobotState(), goal=(5.0, 0.0), obstacles=[Obstacle(x=1.0, y=0.0)])
    sensor = FMCWSensor(range_std_m=0.0, velocity_std_mps=0.0, rng_seed=1)
    controller = ReactiveController(safety_distance_m=1.5)

    linear_speed, angular_speed = controller.command(
        scene.robot, scene.goal, sensor.scan(scene)
    )

    assert linear_speed < controller.cruise_speed_mps
    assert angular_speed != 0.0
