import imp
import cv2
import pyrealsense2
from realsense_depth import *
# import imutils

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
    hsv_color_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2HSV)

    #'blue': [[128, 255, 255], [90, 50, 70]]
    lower_bound = np.array([90, 90, 100])   
    upper_bound = np.array([120, 255, 255])

    #find the colors within the boundaries
    mask = cv2.inRange(hsv_color_img, lower_bound, upper_bound)

    #define kernel size
    kernel = np.ones((7,7), np.uint8)

    #remove unnecessary noise from mask
    mask = cv2. morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2. morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #segment only the detected region
    segmented_img = cv2.bitwise_and(color_img, color_img, mask=mask)

    #find contours from the mask
    contours, hierachy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(color_img, contours, -1, (0,0,255), 3)

    #find center of contour

    # contours = imutils.grab_contours(contours)

    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        cv2.circle(color_img, (cX, cY), 7, (255, 255, 255), -1)
        _point = (cX,cY)
        distance = dc.get_dist(_point)
        color = hsv_color_img[_point[1], _point[0]]
        point_xyz = dc.get_coor(_point)
        for i in range(len(point_xyz)):
            point_xyz[i] = round(point_xyz[i]/100,1)
        
        # cv2.putText(color_img, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.putText(color_img, "{}".format(color), (_point[0], _point[1] + 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.putText(color_img, "{}dm".format(point_xyz), (_point[0], _point[1] + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)




    # Show distance for a specific point
    cv2.circle(color_img, point, 4, (0, 0, 255))
    distance = dc.get_dist(point)
    color = hsv_color_img[point[1], point[0]]
    point_xyz = dc.get_coor(point)
    for i in range(len(point_xyz)):
        point_xyz[i] = round(point_xyz[i]/100,1)

    # cv2.putText(color_img, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.putText(color_img, "{}".format(color), (point[0], point[1] + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    # cv2.putText(color_img, "{}dm".format(point_xyz), (point[0], point[1] + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    # cv2.imshow("depth frame", depth_img)
    cv2.imshow("Color frame", color_img)
    # cv2.imshow("HSV Color frame", hsv_color_img)
    # cv2.imshow("Output frame", output_img)
    # cv2.imshow("mask", mask)
    key = cv2.waitKey(1)
    # break
    if key == 27:
        break