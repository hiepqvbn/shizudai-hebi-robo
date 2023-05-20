import cv2
import pyrealsense2
from realsense_depth import *

point = (400, 300)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

# Initialize Camera Intel Realsense
dc = DepthCamera()

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

while True:
    ret, depth_img, color_img, depth_frame, color_frame = dc.get_frame()
    
    ###########
    # depth_frame = dc.decimate.process(depth_frame)
    
    # points = dc.pc.calculate(depth_frame)
    # dc.pc.map_to(color_frame)
    # v, t = points.get_vertices(), points.get_texture_coordinates()
    # verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    # texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
    # # print(len(texcoords))
    ###############

    #change bgr to hsv(hue, saturation, value)
    # color_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2HSV)

    #'blue': [[128, 255, 255], [90, 50, 70]]

    # Show distance for a specific point
    cv2.circle(color_img, point, 4, (0, 0, 255))
    distance = dc.get_dist(point)
    color = color_img[point[1], point[0]]
    camera_coordinate = dc.get_coor(point)
    point_xyz = []
    for i in camera_coordinate:
        point_xyz.append(round(i/100,1))

    cv2.putText(color_img, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.putText(color_img, "{}".format(color), (point[0], point[1] + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.putText(color_img, "{}dm".format(point_xyz), (point[0], point[1] + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow("depth frame", depth_img)
    cv2.imshow("Color frame", color_img)
    key = cv2.waitKey(1)
    # break
    if key == 27:
        break