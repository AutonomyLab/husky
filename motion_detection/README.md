## Motion Detection node

This node watches the camera feed of a stationary camera, publishing
areas of the image that contain significant motion as determined by
a mixture-of-gaussians background subtraction algorithm.

### Publishing regions

Regions are published to a specified topic in the form of a polygon (list of Point32 objects) where each group of four points in the polygon's list correspond to a single rectangle of interest.

Thus, a published polygon of 12 points (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11) would represent three separate rectangles (p0 p1 p2 p3) (p4 p5 p6 p7) and (p8 p9 p10 p11).

This slightly complicated convention is used in all of the image processing for the Husky, and it is done this way to avoid the inconvenience of defining a new ROS message that consists of a list of lists, which is not trivial.

### Parameters

| Name              | Type      | Default               |
| ----------------- | --------- | --------------------- |
| `~motion_region_size`  | Integer    | `25`   |
| `~motion_threshold`   | Integer    | `1`     |
| `~hysteresis_delay`  | Integer    | `5`  |
| `~hysteresis_decay`  | Integer    | `2`  |
| `~image_topic`   | String    | `axis/image_raw/decompressed`     |
| `~motion_topic`  | String    | `motion_detection`  |
