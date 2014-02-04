## Periodic Gestures node

This node listens to the Motion Detection node for areas of significant
motion, and then monitors those areas of the camera feed for periodic
motions which can be passed to a classifier to determine whether or not
they correspond to legitimate gestures.

### Publishing regions

Regions are published to a specified topic in the form of a polygon (list of Point32 objects) where each group of four points in the polygon's list correspond to a single rectangle of interest.

Thus, a published polygon of 12 points (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11) would represent three separate rectangles (p0 p1 p2 p3) (p4 p5 p6 p7) and (p8 p9 p10 p11).

This slightly complicated convention is used in all of the image processing for the Husky, and it is done this way to avoid the inconvenience of defining a new ROS message that consists of a list of lists, which is not trivial.

### Parameters

| Name              | Type      | Default               |
| ----------------- | --------- | --------------------- |
| `~gesture_topic`  | String    | `periodic_gestures`   |
| `~image_topic`   | String    | `axis/image_raw/decompressed`     |
| `~motion_topic`  | String    | `motion_detection`  |
| `~temporal_window`  | Integer    | `120`  |
| `~min_gesture_freq`  | Float    | `0.5`   |
| `~max_gesture_freq`  | Float    | `2.0`   |
| `~camera_framerate`  | Integer    | `30`  |
| `~peak_sensitivity`  | Float    | `9.0`   |
| `~min_peak`   | Float    | `50.0`     |
| `~spatial_window_x`  | Integer    | `10`  |
| `~spatial_window_y`  | Integer    | `10`  |
| `~overlap_factor`   | Float    | `2.0`     |
