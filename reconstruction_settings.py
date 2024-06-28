PI = 3.14159265359


### reco settings ###
num_projections = 360
angular_range = 2 * PI
source_isocenter_distance = 785
source_detector_distance = 1200
detector_shape = (700,)
detector_spacing = (0.64,)
detector_origin = (-223.68,)

### for reconstruction ###
reco_shape_full = (256, 256)
reco_spacing_full = (1.0, 1.0)
reco_origin_full = (-127.5, -127.5)

### for reconstruction onto a coarser grid ###
reco_shape_small = (128, 128)
reco_spacing_small = (2.0, 2.0)
reco_origin_small = (-127.0, -127.0)
