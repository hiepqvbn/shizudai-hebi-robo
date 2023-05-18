import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2 ** 1)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)



        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        self.frames = self.pipeline.wait_for_frames()
        # print(len(frames))
        self.depth_frame = self.frames.get_depth_frame()
        self.color_frame = self.frames.get_color_frame()

        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())
        if not self.depth_frame or not self.color_frame:
            return False, None, None
        return True, self.depth_image, self.color_image, self.depth_frame, self.color_frame

    #kyori
    def get_dist(self, point):
        return self.depth_image[point[1], point[0]]

    # zahyou
    def get_coor(self, point):
        distance = self.get_dist(point)
        self.depth_intrinsics = rs.video_stream_profile(self.depth_frame.profile).get_intrinsics()
        camera_coordinate = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [point[1],point[0]], distance)
        #unit:[mm]
        point_xyz = []
        point_xyz.append(camera_coordinate[1])
        point_xyz.append(camera_coordinate[2])
        point_xyz.append(-camera_coordinate[0])
        return point_xyz

    def release(self):
        self.pipeline.stop()