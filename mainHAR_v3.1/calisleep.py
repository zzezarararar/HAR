from collections import namedtuple
import math
from statistics import median
import time
import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from scipy.signal import find_peaks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
left_hip = [] #23 
right_hip = [] #24
left_knee = [] #25
right_knee = [] #26
del_shoulder_hip_sleeping = []
del_foot_hip_sleeping = []
standing_frame = []
sleeping_frame = []
del_shoulder_hip_standing = []
del_foot_hip_standing = []
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
        if down_sam_count == 2:
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
        if joints_2D[joint_index].x <= 1 and joints_2D[joint_index].y <= 1 and joints_2D[joint_index].visibility >= 0.6 :
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

bagfile = "champcalisleep.bag"

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
        window_name = "Calibration stand sleep"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        time_start = timecheck()
        with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,model_complexity = 2) as pose:
            
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
                    cv2.imshow(window_name, im_color)
                    
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
                if 1 < time_pass <= 6:
                    if type(avg_data) == list:
                        left_shoulder_hip = abs(avg_data[23][1] - avg_data[11][1])
                        left_foot_hip = abs(avg_data[23][1] - avg_data[29][1])
                        right_shoulder_hip = abs(avg_data[24][1] - avg_data[12][1])
                        right_foot_hip = abs(avg_data[24][1] - avg_data[30][1])
                        del_shoulder_hip_standing.append((left_shoulder_hip+right_shoulder_hip)/2)
                        del_foot_hip_standing.append((left_foot_hip+right_foot_hip)/2)
                        frame_count += 1
                        standing_frame.append(frame_count)
                    cv2.putText(im_color,'Please stand still for 5 sec', (200,400) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                    cv2.putText(im_color,'Countdown ' +str(time_pass-1)+' sec', (200,500) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                if 6 < time_pass <= 10:
                    pass
                    cv2.putText(im_color,'Please lie down to sleep within 4 sec', (200,400) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                    cv2.putText(im_color,'Countdown ' +str(time_pass-6)+' sec', (200,500) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                if 10 < time_pass <= 15:
                    if type(avg_data) == list:
                        if type(avg_data[30][1]) == str or type(avg_data[29][1]) == str:
                            #print(avg_data[29][1], avg_data[30][1])
                            if type(avg_data[30][1]) == str:
                                avg_data[30][1] = avg_data[29][1]
                            else:
                                avg_data[29][1] = avg_data[30][1]
                        #print(avg_data[23][1], avg_data[11][1])
                        left_shoulder_hip = abs(avg_data[23][1] - avg_data[11][1])
                        #print(avg_data[23][1], avg_data[29][1])
                        left_foot_hip = abs(avg_data[23][1] - avg_data[29][1])
                        #print(avg_data[24][1], avg_data[12][1])
                        right_shoulder_hip = abs(avg_data[24][1] - avg_data[12][1])
                        #print(avg_data[24][1], avg_data[30][1])
                        right_foot_hip = abs(avg_data[24][1] - avg_data[30][1])
                        del_shoulder_hip_sleeping.append((left_shoulder_hip+right_shoulder_hip)/2)
                        del_foot_hip_sleeping.append((left_foot_hip+right_foot_hip)/2)
                        frame_count += 1
                        sleeping_frame.append(frame_count)
                    cv2.putText(im_color,'Please lie down 5 sec', (200,400) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                    cv2.putText(im_color,'Countdown ' +str(time_pass-10)+' sec', (200,500) , cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255),2 , cv2.LINE_4)
                if cv2.waitKey(1) == 27 or time_pass > 14:
                    plt.plot(sleeping_frame,del_shoulder_hip_sleeping,'r',sleeping_frame,del_foot_hip_sleeping,'b',standing_frame,del_shoulder_hip_standing,'r',standing_frame,del_foot_hip_standing,'b')
                    break
                #if type(joint_datatata[0]) == np.ndarray :
                #    if type(joint_datatata[23]) == np.ndarray and type(joint_datatata[25]) == np.ndarray :
                #        left_sitting.append((joint_datatata[11][1] - joint_datatata[23][1]) / (joint_datatata[11][1] - joint_datatata[25][1])  )
                #        left_frame.append(frame_count)
                #    if type(joint_datatata[24]) == np.ndarray and type(joint_datatata[26]) == np.ndarray :
                #        right_sitting.append((joint_datatata[12][1] - joint_datatata[24][1]) / (joint_datatata[12][1] - joint_datatata[26][1]) )
                #        right_frame.append(frame_count)      
                cv2.imshow(window_name, im_color)
                #cv2.imshow('Zezar',im_color)
            
        pipeline.stop()
        
        cv2.destroyAllWindows()
        #Sleeping data
        shoulder_hip_sleep_peak,_ = find_peaks(del_shoulder_hip_sleeping, height=0)
        foot_hip_sleep_peak,_ = find_peaks(del_foot_hip_sleeping, height=0)
        shoulder_hip_sleep_data = [del_shoulder_hip_sleeping[i] for i in shoulder_hip_sleep_peak]
        foot_hip_sleep_data = [del_foot_hip_sleeping[i] for i in foot_hip_sleep_peak]
        s_h_sleep = max(shoulder_hip_sleep_data)
        f_h_sleep = max(foot_hip_sleep_data)

        #Standind data
        shoulder_hip_stand_peak,_ = find_peaks([-i for i in del_shoulder_hip_standing], height=-10)
        foot_hip_stand_peak,_ = find_peaks([-i for i in del_foot_hip_standing], height=-10)
        shoulder_hip_stand_data = [del_shoulder_hip_standing[i] for i in shoulder_hip_stand_peak]
        foot_hip_stand_data = [del_foot_hip_standing[i] for i in foot_hip_stand_peak]
        s_h_stand = min(shoulder_hip_stand_data)
        f_h_stand = min(shoulder_hip_stand_data)

        #Print the calibration results
        print( s_h_sleep+(abs(s_h_sleep-s_h_stand))/4) #shoulder-hip
        print( f_h_sleep+(abs(f_h_sleep-f_h_stand))/4) #heel-hip
        plt.show()
    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))