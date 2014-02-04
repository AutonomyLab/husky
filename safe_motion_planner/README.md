## Safe Motion Planner node

This node listens for planned motion commands and passes them through
the potential field method for safe navigation in the presence of
obstacles.

### Parameters

| Name              | Type      | Default               |
| ----------------- | --------- | --------------------- |
| `~joy_topic`  | String    | `joy`   |
| `~potential_field_topic`   | String    | `potential_field_sum`     |
| `~cmd_vel_topic`  | String    | `husky/cmd_vel`  |
| `~plan_cmd_vel_topic`  | String    | `husky/plan_cmd_vel`   |
| `~joy_vector_magnitude`   | Float    | `1.5`     |
| `~drive_scale`   | Float    | `1.0`     |
| `~turn_scale`   | Float    | `1.0`     |
| `~safe_reverse_speed`   | Float    | `0.2`     |
| `~deadman_button`   | Integer    | `0`     |
| `~planner_button`   | Integer    | `3`     |
| `~cmd_rate`   | Integer    | `10`     |
