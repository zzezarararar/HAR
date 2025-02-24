from collections import namedtuple
import math
from statistics import median
import time
from turtle import left, right
import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from scipy.signal import find_peaks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
nose_standing = [] #0 
left_shoulder_standing = [] #11
right_shoulder_standing = [] #12
nose_walking = [] #0 
left_shoulder_walking = [] #11
right_shoulder_walking = [] #12
left_foot_standing = [] #31
right_foot_standing = [] #32
left_foot_walking = [] #31
right_foot_walking = [] #32
plot_frame = []
deltaR_nose_standing = []
deltaR_left_shoulder_standing = []
deltaR_right_shoulder_standing = []
deltaR_shoulder_standing = []
deltaR_left_foot_standing = []
deltaR_right_foot_standing = []
deltaR_foot_standing = []
deltaR_nose_walking = []
deltaR_left_shoulder_walking = []
deltaR_right_shoulder_walking = []
deltaR_shoulder_walking = []
deltaR_left_foot_walking = []
deltaR_right_foot_walking = []
deltaR_foot_walking = []
Q1 = [] # Q1 for moving window
Q2 = [] # Q2 for recognition

## window_size: window size for moving average, down_sam_count: down sampling counter, s: bed boundary, end: end the program flag 
##t1: start time when getting in the bed boundary, no_skele_frame_count: no skeletons frame counter
#[window_size,Q2_size,frame_count,down_sam_count,s,end,t1,no_skele_frame_count] = [4,2,0,0,0,0,0,0]
window_size = 4
Q2_size = 2
frame_count = 0
down_sam_count = 0 # down sampling counter

def moving_average(jointdt):
    global down_sam_count,Q2_size,window_size
    Q1.append(jointdt)
    if len(Q1) == window_size:
        ##list for storing filtered data
        joint_data_filtered=[[str(j) for j in range(3)] for i in range(33)]
        
        ##filter 33 joints
        for no_of_joint in range(33):
            ## check if joint is detected ? 
            if [type(Q1[i][no_of_joint]) for i in range(window_size)] == [np.ndarray for i in range(window_size)] :
                for axis in range(3):
                    ##moving average
                    data_filtered = sum([Q1[i][no_of_joint][axis] for i in range(window_size)])/window_size
                    joint_data_filtered[no_of_joint][axis] = data_filtered
        Q1.pop(0)
        
        down_sam_count = down_sam_count+1
        if down_sam_count == 4:
            down_sam_count=0
            return joint_data_filtered
            
def render_ids_3d(render_image,joints_2D,depth_map,depth_intrinsic):
    thickness = 1
    text_color = (0, 0, 0)
    rows, cols, channel = render_image.shape[:3]
    distance_kernel_size = 5
    joint_dataa = []
    
    #store each person coordinate 
    joint_dataa = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    
        #loop for each joint
    for joint_index in range(len(joints_2D)):
        
    # check if the joint was detected and has valid coordinate
        if joints_2D[joint_index].x <= 1 and joints_2D[joint_index].y <= 1 and joints_2D[joint_index].visibility >= 0.5 :
            distance_in_kernel = []
            x_pixel = joints_2D[joint_index].x * 1280
            y_pixel = joints_2D[joint_index].y * 720
            low_bound_x = max(0,int(x_pixel - math.floor(distance_kernel_size / 2)),)
            upper_bound_x = min(cols - 1,int(x_pixel + math.ceil(distance_kernel_size / 2)),)
            low_bound_y = max(0,int(y_pixel - math.floor(distance_kernel_size / 2)),)
            upper_bound_y = min(rows - 1,int(y_pixel + math.ceil(distance_kernel_size / 2)),)
            for x in range(low_bound_x, upper_bound_x):
                for y in range(low_bound_y, upper_bound_y):
                    distance_in_kernel.append(depth_map.get_distance(x,y))
            median_distance = np.percentile(np.array(distance_in_kernel), 50)
            depth_pixel = [int(x_pixel),int(y_pixel),]
            #if median_distance >= 0.3 valid coordinate 
            if median_distance >= 0.3:
                #find joint coordinate wrt camera
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsic, depth_pixel, median_distance)
                point_3d = np.round([float(i) for i in point_3d], 3)
                #point_str = [str(x) for x in point_3d]
                #plot coor in cv 
                #cv2.putText(im_color,str(point_3d),(int(x_pixel), int(y_pixel)),cv2.FONT_HERSHEY_DUPLEX,0.4,text_color,thickness,)
                ##collect each joint coor
                joint_dataa[joint_index] = point_3d
                ##add time 
            
    #joint_dataa.append(timecheck())
    
    return joint_dataa

#Time
def timecheck() :
    now = time.asctime(time.localtime(time.time()))
    day = now[0:3]
    month = now[4:7]
    date = now[8:10]
    h = int(now[11:13])
    m = int(now[14:16])
    s = int(now[17:19])
    year = now[20:24]
    return [day,date,month,year,h,m,s]

def timedur(A,B) :
    #A start,B end are lists of timecheck
    sA = A[4]*3600 + A[5]*60 + A[6]
    sB = B[4]*3600 + B[5]*60 + B[6]
    #duration in sec
    sd = sB-sA
    return sd 

def R(r):
    ## R = root(x^2+y^2+z^2)
    return math.sqrt(sum([i**2 for i in r]))

def remove_outlier(data):
    [Q1,Q3] = [0,0]
    return_data = []
    Q1 = np.percentile(data, 25,interpolation = 'midpoint')
    Q3 = np.percentile(data, 75,interpolation = 'midpoint')
    IQR = Q3-Q1
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR
    for i in range(len(data)):
        if  lower <= data[i] <= upper:
            return_data.append(data[i])
    return return_data

bagfile = "champcalistandwalk.bag"

# Main content begins
if __name__ == "__main__":
    try:
        # Configure depth and color streams of the intel realsense
        config = rs.config()
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, bagfile)
        config.enable_stream(rs.stream.depth,1280,720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color,1280,720, rs.format.rgb8, 30)
        # Start the realsense pipeline
        pipeline = rs.pipeline()

        #Start streaming from file
        pipeline.start(config)
        
        # Create align object to align depth frames to color frames
        align = rs.align(rs.stream.color)
        
        # Get the intrinsics information for calculation of 3D point
        unaligned_frames = pipeline.wait_for_frames()
        frames = align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics
        # Initialize the cubemos api with a valid license key in default_license_dir()   
       
        # Create window for initialisation
        window_name = "Calibration stand walk"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        time_start = timecheck()
        with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
            
            while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
                unaligned_frames = pipeline.wait_for_frames()
                frames = align.process(unaligned_frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth or not color:
                    continue

            # Convert images to numpy arrays
                depth_image = np.asanyarray(depth.get_data())
                color_image = np.asanyarray(color.get_data())
                im_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # render the skeletons on top of the acquired image and display it
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            #cm.render_result(skeletons, color_image, joint_confidence)
                results = pose.process(color_image)
                skeleton = []
                if not results.pose_landmarks:
                    frame_count+=1
                    
                    cv2.imshow(window_name,im_color)
            #esc
                    if cv2.waitKey(1) == 27 :
                        end = 1
                        break
                    
                    continue

                # Draw the pose annotation on the image.
                color_image.flags.writeable = True
                
                mp_drawing.draw_landmarks(
                    im_color,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=6),
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2))
                
                skeleton = results.pose_landmarks.landmark            
                joint_datatata = render_ids_3d(color_image,skeleton , depth, depth_intrinsic)
                avg_data = moving_average(joint_datatata)
                real_time = timecheck()
                time_pass = timedur(time_start,real_time)
                if time_pass <= 5:
                    if type(avg_data) == list:
                        nose_standing.append(avg_data[0])
                        right_shoulder_standing.append(avg_data[12])
                        left_shoulder_standing.append(avg_data[11])
                        left_foot_standing.append(avg_data[31])
                        right_foot_standing.append(avg_data[32])
                    cv2.putText(im_color,'Please stand still for 5 sec', (200,400) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                    cv2.putText(im_color,'Countdown ' +str(time_pass)+' sec', (200,500) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                if 5 < time_pass <= 9:
                    pass
                    cv2.putText(im_color,'Please start walking within 4 sec', (200,400) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                    cv2.putText(im_color,'Countdown ' +str(time_pass-5)+' sec', (200,500) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                if 9 < time_pass <= 14:
                    if type(avg_data) == list:
                        nose_walking.append(avg_data[0])
                        right_shoulder_walking.append(avg_data[12])
                        left_shoulder_walking.append(avg_data[11])
                        left_foot_walking.append(avg_data[31])
                        right_foot_walking.append(avg_data[32])
                    cv2.putText(im_color,'Please keep walking for 5 sec', (200,400) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                    cv2.putText(im_color,'Countdown ' +str(time_pass-9)+' sec', (200,500) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                if cv2.waitKey(1) == 27 or time_pass > 14:
                    standing_frame = []
                    walking_frame = []
                    for i in range(0,len(nose_standing)-1) :
                        deltaR_nose_standing.append(R(np.array(nose_standing[i])-np.array(nose_standing[i+1])))
                        deltaR_left_shoulder_standing.append(R(np.array(left_shoulder_standing[i])-np.array(left_shoulder_standing[i+1])))
                        deltaR_right_shoulder_standing.append(R(np.array(right_shoulder_standing[i])-np.array(right_shoulder_standing[i+1])))
                        deltaR_shoulder_standing.append((deltaR_left_shoulder_standing[-1]+deltaR_right_shoulder_standing[-1])/2)
                        deltaR_left_foot_standing.append(R(np.array(left_foot_standing[i])-np.array(left_foot_standing[i+1])))
                        deltaR_right_foot_standing.append(R(np.array(right_foot_standing[i])-np.array(right_foot_standing[i+1])))
                        deltaR_foot_standing.append((deltaR_left_foot_standing[-1]+deltaR_right_foot_standing[-1])/2)
                        standing_frame.append(i)
                    
                    for i in range(0,len(nose_walking)-1) :
                        deltaR_nose_walking.append(R(np.array(nose_walking[i])-np.array(nose_walking[i+1])))
                        deltaR_left_shoulder_walking.append(R(np.array(left_shoulder_walking[i])-np.array(left_shoulder_walking[i+1])))
                        deltaR_right_shoulder_walking.append(R(np.array(right_shoulder_walking[i])-np.array(right_shoulder_walking[i+1])))
                        deltaR_shoulder_walking.append((deltaR_left_shoulder_walking[-1]+deltaR_right_shoulder_walking[-1])/2)
                        deltaR_left_foot_walking.append(R(np.array(left_foot_walking[i])-np.array(left_foot_walking[i+1])))
                        deltaR_right_foot_walking.append(R(np.array(right_foot_walking[i])-np.array(right_foot_walking[i+1])))
                        deltaR_foot_walking.append((deltaR_left_foot_walking[-1]+deltaR_right_foot_walking[-1])/2)
                        walking_frame.append(i+len(nose_standing))
                    break

                cv2.imshow(window_name,im_color)
            
        pipeline.stop()
        cv2.destroyAllWindows()
        plt.plot(standing_frame,deltaR_nose_standing,'r-',standing_frame,deltaR_shoulder_standing,'g-',standing_frame,deltaR_foot_standing,'b-')
        plt.plot(walking_frame,deltaR_nose_walking,'r-',walking_frame,deltaR_shoulder_walking,'g-',walking_frame,deltaR_foot_walking,'b-')
        #plt.show()

        # Standing data
        deltaR_nose_standing = remove_outlier(deltaR_nose_standing)
        deltaR_shoulder_standing = remove_outlier(deltaR_shoulder_standing)
        deltaR_foot_standing = remove_outlier(deltaR_foot_standing)
        nose_standing_peak,_ = find_peaks(deltaR_nose_standing, height=0)
        shoulder_standing_peak,_ = find_peaks(deltaR_shoulder_standing, height=0)
        foot_standing_peak,_ = find_peaks(deltaR_foot_standing, height=0)
        nose_standing_data = [deltaR_nose_standing[i] for i in shoulder_standing_peak]
        shoulder_standing_data = [deltaR_shoulder_standing[i] for i in shoulder_standing_peak]
        foot_standing_data = [deltaR_foot_standing[i] for i in foot_standing_peak]
        n_stand = max(nose_standing_data)
        s_stand = max(shoulder_standing_data)
        f_stand = max(foot_standing_data)

        #Walkind data
        deltaR_nose_walking = remove_outlier(deltaR_nose_walking)
        deltaR_shoulder_walking = remove_outlier(deltaR_shoulder_walking)
        nose_walking_peak,_ = find_peaks([-i for i in deltaR_nose_walking], height=-10)
        shoulder_walking_peak,_ = find_peaks([-i for i in deltaR_shoulder_walking], height=-10)
        foot_walking_peak,_ = find_peaks([-i for i in deltaR_foot_walking], height=-10)
        nose_walking_data = [deltaR_nose_walking[i] for i in nose_walking_peak]
        shoulder_walking_data = [deltaR_shoulder_walking[i] for i in shoulder_walking_peak]
        foot_walking_data = [deltaR_foot_walking[i] for i in foot_walking_peak]
        n_walk = min(nose_walking_data)
        s_walk = min(shoulder_walking_data)
        f_walk = min(foot_walking_data)

        #Print the calibration results
        print( f_stand+(abs(f_stand-f_walk))/4) #'First threshold'
        print( n_stand+(abs(n_stand-n_walk))/4 ) #'Second threshold'
        plt.show()
    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))