## Potential Field node

This node watches the laser scanner and publishes an obstacle vector
that represents the potential field response at this particular
location, for use by the safe motion planner to determine safe motion
commands.

### Parameters

| Name              | Type      | Default               |
| ----------------- | --------- | --------------------- |
| `~laser_topic`  | String    | `lidar/scan`   |
| `~potential_field_topic`   | String    | `potential_field_sum`     |
| `~sample_rate`  | Integer    | `5`  |
| `~min_angle`  | Float    | `-PI/2`   |
| `~max_angle`  | Float    | `PI/2`   |
| `~side_obstacle_force`  | Float    | `2.0`   |
| `~front_obstacle_force`  | Float    | `8.0`   |
