## People Follow node

This node listens to the perception nodes for periodic gesture detection
and person detection and controls the behavior of the robot accordingly.

### Publishing regions

A note on data types received by this node from the detector nodes.

Regions are published to a specified topic in the form of a polygon (list of Point32 objects) where each group of four points in the polygon's list correspond to a single rectangle of interest.

Thus, a published polygon of 12 points (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11) would represent three separate rectangles (p0 p1 p2 p3) (p4 p5 p6 p7) and (p8 p9 p10 p11).

This slightly complicated convention is used in all of the image processing for the Husky, and it is done this way to avoid the inconvenience of defining a new ROS message that consists of a list of lists, which is not trivial.

### Parameters

| Name              | Type      | Default               |
| ----------------- | --------- | --------------------- |
| `~gesture_topic`  | String    | `periodic_gestures`   |
| `~person_topic`   | String    | `detected_people`     |
| `~cmd_vel_topic`  | String    | `husky/plan_cmd_vel`  |
