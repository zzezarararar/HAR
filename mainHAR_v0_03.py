###############################################################
### This file is main HAR                                   ###
### Version: 0.03                                           ###
### Description: This file conclude User Interface,         ###
### video, realtime activity recognition, and call other    ###
### file to process calibration.                            ###
###############################################################

from collections import namedtuple
import math
import time
import tkinter as tk
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import numpy as np
import pyrealsense2 as rs
from tkinter import messagebox
from scipy.signal import find_peaks
import subprocess
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

## Q1 for moving window, Q2 for recognition
[Q1,Q2] =[[],[]]
## w:window size, n:Q2 size,c:down sampling count, s :bed boundary- 0=N ,1 = Y :,end: program end? 
##t1 : start time when getting in the bed boundary, f: count no skeletons frame
[w,n,frame_count,c,s,end,t1,f] = [4,2,0,0,0,0,0,0]
## Activity
act = ' '
## Time
qt1 = []
qt2 = []
##durtion dur = [sit_dur,stand_dur,walk_dur,sleep_dur]
dur = [0,0,0,0,0,0,0]
act_in = [0,0,0,0,0,0]
##List for checking activity
a = ['Sitting','Standing','Walking','Sleeping','Falling down','',' ']
## explode for plot
explodee=[0, 0, 0, 0]
colors = ['deepskyblue','green','lightcoral','goldenrod','darkorange']
## data for table
data = []
calibrate_point = []
in_bed = False

bed_boundary = [[-0.2000540941953659, 0.6348057985305786, 2.5470001697540283], 
                [-0.2253461480140686, 0.33657339215278625, 3.383000135421753], 
                [-0.7048231959342957, 0.3146856725215912, 3.1630001068115234], 
                [-0.7926374673843384, 0.6519354581832886, 2.56000018119812]]

walk_first_threshold = 0.024 #2 down: 0.024
walk_second_threshold = 0.035 #2 down: 0.019

sit_right_threshold = 0.1875
sit_left_threshold = 0.1875

sleep_s_h_threshold = 0.22
sleep_f_h_threshold = 0.25

def moving_average(jointdt):
    global c,n,w
    Q1.append(jointdt)
    if len(Q1) == w:
        ##list for storing filtered data
        joint_data_filtered=[[str(j) for j in range(3)] for i in range(33)]
        
        ##filter 18 joint
        for no_of_joint in range(33):
            ## check if joint is detected ? 
            if [type(Q1[i][no_of_joint]) for i in range(w)] == [np.ndarray for i in range(w)] :
                
                for axis in range(3):
                    ##moving average
                    data_filtered = sum([Q1[i][no_of_joint][axis] for i in range(w)])/w
                    joint_data_filtered[no_of_joint][axis] = data_filtered
        Q1.pop(0)

        ##Downsampling
        c = c+1
        if c == 3:
            Q2.append(joint_data_filtered)
            if len(Q2) == n:
                check_bed_boundary(Q2[0])
                recognition(Q2)
                Q2.pop(0)
            c=0

def recognition(Qr):
    global act,in_bed
    ##if person is in the bed boundary
    if in_bed:
        ## Check sleeping on the bed
        if sleeping(Qr[0]):
            act = 'Sleeping'
        ## Check sitting on the bed
        elif sitting(Qr[0]):
            act ='Sitting'
        # Stand up after sitting
        elif act == 'Sitting':
            act = 'Standing'
        ## Check standing/walking on the bed
        else:
            walking_standing(Qr)

    ##if person is not in the bed boundary
    else:
        ## Check falling down
        if sleeping(Qr[0]):
            # Sleeping outside the bed means falling down
            act = 'Falling down'
       
        ## Check sitting on a chair
        elif sitting(Qr[0]):
            act ='Sitting'
        # Stand up after sitting
        # elif act == 'Sitting':
        #     act = 'Standing'
        ## Check standing/walking
        else:
            walking_standing(Qr)

def walking_standing(Qws):
    global act, walk_first_threshold,walk_second_threshold
    first_threshold = 0
    second_threshold = 0
    ## joint [neck,right shoulder,left shoudlder]
    jlist = [0,11,12,31,32]
    ## deltaR: [0]-neck,[1]-left shoulder,[2]-right shoulder,[3]-left foot,[4]-right foot
    deltaR = [0,0,0]
    first_frame = [Qws[0][j] for j in jlist]
    second_frame = [Qws[1][j] for j in jlist]
    nose_joint = first_frame[0]+second_frame[0]
    left_shoulder_joint = first_frame[1]+second_frame[1]
    right_shoulder_joint = first_frame[2]+second_frame[2]
    left_foot_joint = first_frame[3]+second_frame[3]
    right_foot_joint = first_frame[4]+second_frame[4]
    if [type(a) for a in nose_joint]==[np.float64 for i in range(6)] and [type(a) for a in left_foot_joint]==[np.float64 for i in range(6)] and [type(a) for a in right_foot_joint]==[np.float64 for i in range(6)]:
        nose_vector = []
        left_foot_vector = []
        right_foot_vector = []
        for i in range(3):
            nose_vector.append(nose_joint[i]-nose_joint[i+3])
            left_foot_vector.append(left_foot_joint[i]-left_foot_joint[i+3])
            right_foot_vector.append(right_foot_joint[i]-right_foot_joint[i+3])
        
        deltaR[0] = R(nose_vector)
        deltaR[1] = R(left_foot_vector)
        deltaR[2] = R(right_foot_vector)

    elif [type(a) for a in nose_joint]==[np.float64 for i in range(6)] and [type(a) for a in left_shoulder_joint]==[np.float64 for i in range(6)] and [type(a) for a in right_shoulder_joint]==[np.float64 for i in range(6)]:
        nose_vector = []
        left_shoulder_vector = []
        right_shoulder_vector = []
        for i in range(3):
            nose_vector.append(nose_joint[i]-nose_joint[i+3])
            left_shoulder_vector.append(left_shoulder_joint[i]-left_shoulder_joint[i+3])
            right_shoulder_vector.append(right_shoulder_joint[i]-right_shoulder_joint[i+3])
        
        deltaR[0] = R(nose_vector)
        deltaR[1] = R(left_shoulder_vector)
        deltaR[2] = R(right_shoulder_vector)
        
    if walk_first_threshold != 0 and walk_second_threshold != 0:
        if deltaR[0] >= walk_first_threshold and (deltaR[1] >= walk_second_threshold or deltaR[2] >= walk_second_threshold):
            act = 'Walking'
        elif deltaR[0] <= walk_first_threshold or (deltaR[1] <= walk_second_threshold or deltaR[2] <= walk_second_threshold):
            act = 'Standing'
    
def R(r):
    ## R = root(x^2+y^2+z^2)
    return math.sqrt(sum([i**2 for i in r]))

def sleeping(joint_data_fil) :
    global sleep_s_h_threshold, sleep_f_h_threshold
    j_right = [30,24,12] # index of right heel, hip & shoulder
    j_left = [29,23,11] # index of left heel, hip & shoulder 
    joint_right = [joint_data_fil[j][1] for j in j_right]
    joint_left = [joint_data_fil[j][1] for j in j_left]
    if act == 'Sleeping' or act == 'Falling down':
        delta_r_shoulder_hip = 1
        delta_l_shoulder_hip = 1
        if [type(joint_right[a]) for a in range(1,3)] == [np.float64 for i in range(2)]:
            delta_r_shoulder_hip = abs(joint_right[2]-joint_right[1])
        if [type(joint_left[a]) for a in range(1,3)] == [np.float64 for i in range(2)]:
            delta_l_shoulder_hip = abs(joint_left[2]-joint_left[1])
            ##sitting threshold 0.21
        
        return(delta_r_shoulder_hip < sleep_s_h_threshold or delta_l_shoulder_hip < sleep_s_h_threshold) #Med: s_h 0.28
        
    else :
        if [type(a) for a in joint_right] == [np.float64 for i in range(3)] and [type(a) for a in joint_left]== [np.float64 for i in range(3)]:
            delta_r_shoulder_hip = float()
            delta_r_heel_hip = float()
            delta_l_shoulder_hip = float()
            delta_l_heel_hip = float() 
            delta_r_shoulder_hip = abs(joint_right[2]-joint_right[1])
            delta_r_heel_hip = abs(joint_right[1]-joint_right[0])
            delta_l_shoulder_hip = abs(joint_left[2]-joint_left[1])
            delta_l_heel_hip = abs(joint_left[1]-joint_left[0])
            #print('Right: ',  delta_r_shoulder_hip, delta_r_heel_hip)
            #print('Left: ', delta_l_shoulder_hip, delta_l_heel_hip)
            right_side = (delta_r_shoulder_hip <= sleep_s_h_threshold and delta_r_heel_hip <= sleep_f_h_threshold) #Med: s_h 0.28 f_h 0.31 low: s_h 0.22 f_h 0.25
            left_side = (delta_l_shoulder_hip <= sleep_s_h_threshold and delta_l_heel_hip <= sleep_f_h_threshold) #Med: s_h 0.28 f_h 0.31 low: s_h 0.22 f_h 0.25
            return(right_side or left_side)
        
        else :
            delta_r_shoulder_hip = 1
            delta_l_shoulder_hip = 1
            if [type(joint_right[a]) for a in range(1,3)] == [np.float64 for i in range(2)]:
                delta_r_shoulder_hip = abs(joint_right[2]-joint_right[1])
            if [type(joint_left[a]) for a in range(1,3)] == [np.float64 for i in range(2)]:
                delta_l_shoulder_hip = abs(joint_left[2]-joint_left[1])
                ##sitting threshold 0.21
            
            return(delta_r_shoulder_hip < sleep_s_h_threshold or delta_l_shoulder_hip < sleep_s_h_threshold) #Med: s_h 0.28

def sitting(joint_data_fil):
    global sit_right_threshold, sit_left_threshold
    ## index : right hip 24, right knee 26, left hip 23 ,left knee 25
    j_right = [24,26] # index of right hip & knee
    j_left = [23,25] # index of left hip & knee
    delta_r = 1 ; delta_l = 1 
    joint_right = [joint_data_fil[j][1] for j in j_right]
    if [type(a) for a in joint_right]== [np.float64 for i in range(2)]:
        delta_r = abs(joint_right[1]-joint_right[0])
    joint_left = [joint_data_fil[j][1] for j in j_left]
    if [type(a) for a in joint_left]== [np.float64 for i in range(2)]:
        delta_l = abs(joint_left[1]-joint_left[0])  
        ##sitting threshold 0.21
    return(delta_r < sit_right_threshold or delta_l< sit_left_threshold)
    
def check_bed_boundary(joint_data_fil):
    global bed_boundary, in_bed
    x_pos = [] ; z_pos = []
    for i in bed_boundary:
        x_pos.append(i[0])
        z_pos.append(i[2])
    x_max = max(x_pos) ; x_min = min(x_pos)
    z_max = max(z_pos) ; z_min = min(z_pos)
    xnose = [] ; znose = []
    xsh = [] ; zsh = []
    
    if  type(joint_data_fil[0])== list :
        ## x nose
        xnose = joint_data_fil[0][0]
        ## z nose
        znose = joint_data_fil[0][2]
    if  type(joint_data_fil[12])== list :
        ## x right shoulder
        xsh = joint_data_fil[12][0]
        ## z right shoulder
        zsh = joint_data_fil[12][2]
    
    if [type(xnose)]== [np.float64] and [type(znose)]== [np.float64] and [type(xsh)]== [np.float64] and [type(zsh)]== [np.float64]:
        ##bed boundary
        in_bed = (x_min<xnose<x_max and z_min<znose<z_max) or (x_min<xsh<x_max and z_min<zsh<z_max)
        #return(x_min<xshn<x_max and z_min<zshn<z_max)
    
def plot_check(value,xx,yy):
    global frame_count
    ##value to be plot 
    yy.append(value)
    ## plot frame as x axis
    xx.append(frame_count)

def check_full_body(data_joint):
    ## detected joint count
    cf = 0
    for k in range(0,33):
        if type(data_joint[k]) == np.ndarray :
            ##joint is deteched --> increase cf
            cf = cf+1
    
    return cf

#This function is called every frames.
def render_ids_3d(render_image,skeletons_2d,depth_map,depth_intrinsic):
    global w
    thickness = 1
    text_color = (0, 0, 0)
    rows, cols, channel = render_image.shape[:3]
    distance_kernel_size = 5
    # calculate 3D keypoints and display them
    #for skeleton_index in range(len(skeletons_2d)): # len(skeletons_2d) : number of people in frame
        #skeleton_2D = skeletons_2d[skeleton_index]
        #joint opencv coordinate for plotting
    joints_2D = skeletons_2d
    #store each person coordinate 
    joint_dataa = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    count = 0
        #loop for each joint
    for joint_index in range(len(joints_2D)):
        count += 1
    # check if the joint was detected and has valid coordinate
        if joints_2D[joint_index].x <= 1 and joints_2D[joint_index].y <= 1 and joints_2D[joint_index].visibility >= 0.7 :
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
    joint_dataa.append(timecheck())
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

def timeformat(h,m,s):
    ## ex h=2 m = 5 s = 8 --> time: 02:05:08
    if h < 10 :
        h = '0'+str(h)
    else :
        h = str(h)
    if m < 10 :
        m = '0'+str(m)
    else :
        m = str(m)
    if s < 10 :
        s = '0'+str(s)
    else :
        s = str(s)
    return( h + "h "+ m +"m "+ s +"s ")

def timeconvert():
    ##convert time in sec to hour min sec
    global dur
    total = np.sum(dur)
    dur[5] = total
    dur_str = []
    for i in range(6):
        sdur = dur[i]
        ##hour
        hd = math.floor(sdur/3600)
        sdur = sdur - hd*3600
        ##minute
        md = math.floor(sdur/60)
        ##second
        sdur = sdur - md*60
        dur_str.append( timeformat(hd,md,sdur)) 
    return dur_str

def startend(q) :
    ##function to record each activity's time and duration 
    global act,qt1,qt2,act_in,dur,end,data
    ##act_in = [sit,stn,wlk,s,waking up]
    ##a = ['Sitting','Standing','Walking','Sleeping','Falling down','Waking up',' ']
    ##initial
    if act_in == [0,0,0,0,0,0]:
        i = a.index(act)
        if i <= 5 :
            act_in[i] = 1
            qt1 = q[33]
            qt1str = timeformat(qt1[4],qt1[5],qt1[6]) 
            data.append([qt1str,act])
    else:
        ## if Activity isn't the same as the previous frame -->
        ##--> find the old Activity’s duration and timestamp the new Activity's time
        ## new activity
        iact = act_in.index(1) # to identify old activity
        ## if new activity != old activity 
        if a.index(act) != iact:
            qt2 = q[33]
            ## time for changing activity >= 1 
            ## timedur = duration between the old and new act
            if timedur(qt1,qt2) >= 1 : 
                if iact == 5:
                    if data[-2][-1] == "Sleeping":
                        iact = 3
                    elif data[-2][-1] == "Falling down":
                        iact = 4
                dur[iact] += timedur(qt1,qt2)
                act_in[iact] = 0
                actn = a.index(act)
                if actn <= 5:
                    act_in[actn] = 1
                    qt1 = q[33]
                    qt1str = timeformat(qt1[4],qt1[5],qt1[6]) 
                    data.append([qt1str,act])
                else:
                    act_in = [0,0,0,0,0]
            else:
                act = a[iact]
    #print(act)
    ## count time when program is stopped
    if end == 1 :
        qt2 = q[33]
        dur[iact] += timedur(qt1,qt2)

def pieplot_color():
    global dur,colors
    fig, ax = plt.subplots(figsize=(6, 6))
    x = dur[0:5]
    lb = ['Sitting','Standing','Walking','Sleeping','Falling down']
    durstr = timeconvert()
    labels = [1,2,3,4,5]
    for i in range(len(lb)) :
        labels[i] = lb[i] + '\n' +durstr[i]
    #plt.figure(0)
    
    patches, texts, pcts = ax.pie(x, labels=labels, autopct='%.1f%%',wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
    textprops={'size': 'x-large'},
    startangle=90,
    colors=colors)
    for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
    plt.setp(pcts, color='saddlebrown')
    plt.setp(texts, fontweight=600)
    ax.set_title('Activities\n'+'('+durstr[5]+')', fontsize=18,color='saddlebrown',fontweight=600)
    plt.tight_layout()
    plt.show()


def pieexpl():
    global dur, colors,explodee
    x = dur[0:5]
    cmap = plt.get_cmap('Greys')
    labels = ['Sitting','Standing','Walking','Sleeping','Falling down']
    lb = ['Sitting','Standing','Walking','Sleeping','Falling down']
    durstr = timeconvert()
    labels = [1,2,3,4,5]
    for i in range(len(lb)) :
        labels[i] = lb[i] +'\n'+ durstr[i]
    for j in range(len(lb)):
        colors_grey = list(cmap(np.linspace(0.45, 0.85, len(x))))
    colors_grey[j] = colors[j]
    plt.figure(j+1)
    fig, ax = plt.subplots(figsize=(6, 6))
    explodee[j] = 0.1
    patches, texts, pcts = ax.pie(
    x, labels=labels, autopct='%.1f%%',
    wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
    textprops={'size': 'x-large'},
    startangle=90,
    colors =colors_grey,
    explode=tuple(explodee) ) 
    for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
    plt.setp(pcts, color='saddlebrown')
    plt.setp(texts, fontweight=600)
    ax.set_title('Activities\n'+'('+durstr[5]+')', fontsize=18,color='saddlebrown',fontweight=600)
    plt.tight_layout()
    fig.savefig(str(i)+'.png',dpi = 300)
    explodee = [0,0,0,0]
    plt.show()

def table():
    global data
    #plt.figure(5)
    fig, ax =plt.subplots(1,1)
    #fig, ax = plt.subplots(figsize=(1,1))
    # #data=[["8.30","sleep"],
    # ["9.00","sit"],
    # ["10.00",'walk']]
    column_labels=["Time", "Activity"]
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data,colLabels=column_labels,fontsize = 14,colColours=["plum"] * 3,cellLoc='center',loc="center")
    plt.show()

def click(event,x,y,flags,param):
    global calibrate_point
    point = []
    #Get the pixel point when left click
    if event == cv2.EVENT_FLAG_LBUTTON:
        point = [x,y]
        calibrate_point.append(point)

def set_bed_boundary(color_img,depth_map,depth_intrinsic):
    global calibrate_point
    calibration_window = 'Calibrate'
    cv2.namedWindow(calibration_window) 
    cv2.setMouseCallback(calibration_window,click)
    
    while True:
        cv2.imshow(calibration_window,color_img)
        if len(calibrate_point)==4:
            break
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    #print(calibrate_point)
    bed_boundary = []
    bed_coordinate = []
    for i in calibrate_point:
        p = depth_map.get_distance(i[0],i[1])
        bed_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrinsic, i, p)
        bed_boundary.append(bed_coordinate)
    return bed_boundary

def check_num_joint(joint_datatata) :
    nd = 0
    
    for i in joint_datatata :
        if type(i) == np.ndarray  :
            nd = nd +1
    num_joint_detect =  nd
    #print(nd)
    return num_joint_detect
def back(x):
    pass
def runvideo(video_path) :
       global config
       # Configure depth and color streams of the intel realsense
       config = rs.config()
       # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
       rs.config.enable_device_from_file(config, video_path) #remove this for realtime
       config.enable_stream(rs.stream.depth,1280,720, rs.format.z16, 30)
       config.enable_stream(rs.stream.color,1280,720, rs.format.rgb8, 30)
       

       



class RealSenseDepthStreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense Depth Stream App")

        # Variable to track the current mode
        self.mode = "video"

        # Create a button to toggle between modes
        self.mode_button = tk.Button(root, text="Video Mode",bg="light cyan", command=self.toggle_mode)
        self.mode_button.pack(pady=10)

        # Create buttons
        self.start_button = tk.Button(root, text="Start",bg="light cyan", command=self.start_video_stream)
        self.start_button.pack(pady=10)

        self.calibed_button = tk.Button(root, text="Bed",bg="light cyan", command=self.cali_video_bed_boundary)
        self.calibed_button.pack(pady=10)

        self.caliwalk_button = tk.Button(root, text="Stand-Walk",bg="light cyan", command=self.cali_video_stand_walk)
        self.caliwalk_button.pack(pady=10)

        self.calisit_button = tk.Button(root, text="Stand-Sit",bg="light cyan", command=self.cali_video_stand_sit)
        self.calisit_button.pack(pady=10)

        self.calisleep_button = tk.Button(root, text="Stand-Sleep",bg="light cyan", command=self.cali_video_stand_sleep)
        self.calisleep_button.pack(pady=10)
        # Variables
        self.is_streaming = False
        self.pipeline = None
    def toggle_mode(self):
        # Toggle between video and real-time mode
        if self.mode == "video":
            self.mode = "real-time"
            self.mode_button.config(bg="LavenderBlush1") # Reset background color
            self.mode_button.config(text="Real-Time Mode")
            self.start_button.config(bg="LavenderBlush1",command=self.start_rt_stream)
            self.calibed_button.config(bg="LavenderBlush1",command=self.cali_rt_bed_boundary)
            self.caliwalk_button.config(bg="LavenderBlush1",command=self.cali_rt_stand_walk)
            self.calisit_button.config(bg="LavenderBlush1",command=self.cali_rt_stand_sit)
            self.calisleep_button.config(bg="LavenderBlush1",command=self.cali_rt_stand_sleep)
            

            
        else:
            self.mode = "video"
            self.mode_button.config(text="Video Mode")
            self.mode_button.config(bg="light cyan")  # Set background color to blue for video mode
            self.start_button.config(bg="light cyan",command=self.start_video_stream)
            self.calibed_button.config(bg="light cyan",command=self.cali_video_bed_boundary)
            self.caliwalk_button.config(bg="light cyan",command=self.cali_video_stand_walk)
            self.calisit_button.config(bg="light cyan",command=self.cali_video_stand_sit)
            self.calisleep_button.config(bg="light cyan",command=self.cali_video_stand_sleep)
        

   

    def start_video_stream(self):
        bagfile = "gameallnew.bag"
        global frame_count, act, bed_boundary, dur,colors,qt1,qt2,act_in,dur,end,data,in_bed
        end = 0
        fig, ax = plt.subplots()
        activity_count = [0, 0, 0, 0, 0]
        percent_activity = [0, 0, 0, 0, 0]
        activity_list = ['Standing','Walking','Sitting','Sleeping','Falling down']
        frame_count = 0
        act = ' '
        if not self.is_streaming:
            self.is_streaming = True
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                rs.config.enable_device_from_file(config, bagfile)
                config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                config.enable_stream(rs.stream.color,1280,720, rs.format.rgb8, 30)
                self.pipeline.start(config)

                # Create align object to align depth frames to color frames
                align = rs.align(rs.stream.color)

                # Get the intrinsics information for calculation of 3D point
                unaligned_frames = self.pipeline.wait_for_frames()
                frames = align.process(unaligned_frames)
                depth = frames.get_depth_frame()
                depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics
                # Initialize the cubemos api with a valid license key in default_license_dir()

                joint_confidence = 0.25

                window_name = "HAR"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        
                with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,model_complexity = 1) as pose:
                    while True:
                    # Create a pipeline object. This object configures the streaming camera and owns it's handle
                        unaligned_frames = self.pipeline.wait_for_frames()
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

                    #cm.render_result(skeletons, color_image, joint_confidence)
                        results = pose.process(color_image)
                        skeleton = []
                        if not results.pose_landmarks:
                            frame_count += 1
                            f +=1
                            if f >= 4 and in_bed :
                                act = 'Sleeping'
                                #print(in_bed)
                                timesleep = [i for i in range(33)]
                                timesleep.append(timecheck())
                                startend(timesleep)
                                f = 0
                            cv2.putText(im_color,' Activity = ' +act, (200,100) , cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255),2 , cv2.LINE_4)
                            cv2.imshow(window_name, im_color)
                            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                                break
                            continue
                        # Draw the pose annotation on the image.
                        color_image.flags.writeable = True
                        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                        mp_drawing.draw_landmarks(
                            im_color,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=6),
                            connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2))

                        skeleton = results.pose_landmarks.landmark            
                        joint_datatata = render_ids_3d(color_image,skeleton , depth, depth_intrinsic)
                        if check_num_joint(joint_datatata) < 6 :
                            f +=1
                            if f >= 4 and in_bed :
                                act = 'Sleeping'
                                timesleep = [i for i in range(33)]
                                timesleep.append(timecheck())
                                startend(timesleep)
                                f = 0
                        else :
                            f = 0
                            ## low pass filter raw data --> down sampling --> do the recognition
                            moving_average(joint_datatata)
                            startend(joint_datatata)

                        frame_count += 1
                        
                        cv2.putText(im_color,' Activity = ' +act, (100,650) , cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255),2 , cv2.LINE_4)
                        if act != ' ':
                            if act == 'Standing' :
                                activity_count[0] +=1
                            elif act == 'Walking' :
                                activity_count[1] += 1
                            elif act == 'Sitting' :
                                activity_count[2] += 1
                            elif act == 'Sleeping' :
                                activity_count[3] += 1
                            elif act == 'Falling down' :
                                activity_count[4] += 1 
                                
                            for i in range(5) :
                                percent_activity[i] = activity_count[i] / sum(activity_count) *100
                            ax.clear()
                            bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:olive'] #purple, brown, pink, cyan 
                            ax.bar(activity_list, percent_activity, color= colors)
                            ax.set_ylabel('percentage of each activity',fontsize = 16)
                            ax.set_ylim(0, 100)

                            # Set tick font size
                            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                                label.set_fontsize(12)
                            # ax.pie(activity_count, labels=activity_list,colors=[color_pie[key] for key in activity_list] ,autopct = '%1.1f%%',startangle=90)
                            # ax.axis('equal')
                            
                            fig.canvas.draw()

                            chart_img = np.array(fig.canvas.renderer.buffer_rgba())
                            chart_img = cv2.resize(chart_img, (im_color.shape[1]//3, im_color.shape[0]//3))

                            mask = (chart_img[:, :, 0] != 255).astype(np.uint8) * 255


                            # Overlay the chart on the frame with transparency
                            
                            chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)
                            im_color[0:chart_img.shape[0], 0:chart_img.shape[1], :] = cv2.addWeighted(im_color[0:chart_img.shape[0], 0:chart_img.shape[1], :], 0, chart_img, 1, 0)
                            #im_color = cv2.addWeighted(im_color, 0.8, chart_img[:,:,:3],0.2,0)
                        cv2.putText(color_image,' Activity = ' +act, (100,650) , cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255),2 , cv2.LINE_4)    
                        cv2.imshow(window_name, im_color)
                        

                        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                                end = 1
                                ax.clear()
                                pieplot_color()
                                ax.clear()
                                table()
                                ax.clear()
                                activity_count = [0, 0, 0, 0, 0]
                                percent_activity = [0, 0, 0, 0, 0]
                                frame_count = 0
                                joint_dataa = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
                                
                                
                                break

            except Exception as e:
                print(f"Exception in thread: {e}")
            finally:
                self.is_streaming = False
                if self.pipeline:
                    self.pipeline.stop()
                    ## Time
                    qt1 = []
                    qt2 = []
                    dur = [0,0,0,0,0,0]
                    act_in = [0,0,0,0,0,0]
                    a = ['Sitting','Standing','Walking','Sleeping','Falling down','Waking up',' ']
                    ## explode for plot
                    explodee=[0, 0, 0, 0]
                    ## data for table
                    data = []
                    calibrate_point = []
                    in_bed = False
                cv2.destroyAllWindows()

    def cali_video_bed_boundary(self):
        global bed_boundary
        bagfile = "gameallnew.bag"
        if not self.is_streaming:
            config = rs.config()
            # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
            rs.config.enable_device_from_file(config, bagfile) #remove this for realtime
            config.enable_stream(rs.stream.depth,1280,720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color,1280,720, rs.format.rgb8, 30)
            # Start the realsense pipeline
            pipeline = rs.pipeline()
            #Start streaming from file
            pipeline.start(config)
            # Create align object to align depth frames to color frames
            align = rs.align(rs.stream.color)
            #Get color frame for calibtation
            unaligned_frames = pipeline.wait_for_frames()
            frames = align.process(unaligned_frames)
            color = frames.get_color_frame()
            color_image = np.asanyarray(color.get_data())
            color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
            depth = frames.get_depth_frame()
            depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics
            cv2.imshow('Sample video',color_image)
            bed_boundary = set_bed_boundary(color_image,depth,depth_intrinsic)
            print(bed_boundary)
    
    def cali_video_stand_walk(self):
        global walk_first_threshold,walk_second_threshold
        file_to_run = 'caliwalk.py'
        e = None
        try:
            result = subprocess.run(['python', file_to_run], capture_output=True, text=True, check=True)
            output_lines = result.stdout.strip().split('\n')  # Split the output into lines

            if len(output_lines) == 2:
                walk_second_threshold = float(output_lines[0])  # Convert the first line to float
                walk_first_threshold = float(output_lines[1])    # Convert the second line to int

            print('Calibration stand-walk threshold:\n' + 'Nose threshold :' + str(walk_first_threshold)+ 
                  '(\n Old : 0.024 \nFoot threshold :' + str(walk_second_threshold) + '\n Old : 0.035')
        except subprocess.CalledProcessError as error:
            e = error
            print("Error:", e)
            print("Output:", e.output)

    def cali_video_stand_sit(self):
        global sit_right_threshold, sit_left_threshold
        file_to_run = 'calisit.py'
        e = None
        try:
            result = subprocess.run(['python', file_to_run], capture_output=True, text=True, check=True)
            print("Output:", result.stdout)
            output_lines = result.stdout.strip().split('\n')  # Split the output into lines

            if len(output_lines) == 2:
                sit_left_threshold= float(output_lines[0])  # Convert the first line to float
                sit_right_threshold = float(output_lines[1])    # Convert the second line to int

            print('Calibration stand-sit threshold:\n' + 'right threshold :' + str(sit_right_threshold)+ 
                  '\nleft threshold :' + str(sit_left_threshold) + '\n Old : 0.1875')
        except subprocess.CalledProcessError as error:
            e = error
            print("Error:", e)
            print("Output:", e.output)

    def cali_video_stand_sleep(self):
        global sleep_s_h_threshold, sleep_f_h_threshold
        file_to_run = 'calisleep.py'
        e = None
        try:
            result = subprocess.run(['python', file_to_run], capture_output=True, text=True, check=True)
            print("Output:", result.stdout)
            output_lines = result.stdout.strip().split('\n')  # Split the output into lines

            if len(output_lines) == 2:
                sleep_s_h_threshold = float(output_lines[0])  # Convert the first line to float
                sleep_f_h_threshold = float(output_lines[1])    # Convert the second line to int

            print('Calibration stand-sleep threshold:\n' + 'shoulder-hip threshold :' + str(sleep_s_h_threshold)+ 
                  '(\n Old : 0.22 \nheel-hip threshold :' + str(sleep_f_h_threshold) + '\n Old : 0.25')
        except subprocess.CalledProcessError as error:
            e = error
            print("Error:", e)
            print("Output:", e.output)

    def start_rt_stream(self):
        global frame_count, act, bed_boundary, dur,colors,qt1,qt2,act_in,dur,end,data,in_bed,f
        end = 0
        fig, ax = plt.subplots()
        activity_count = [0, 0, 0, 0, 0]
        percent_activity = [0, 0, 0, 0, 0]
        activity_list = ['Standing','Walking','Sitting','Sleeping','Falling down']
        frame_count = 0
        act = ' '
        if not self.is_streaming:
            self.is_streaming = True
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                config.enable_stream(rs.stream.color,1280,720, rs.format.rgb8, 30)
                self.pipeline.start(config)

                # Create align object to align depth frames to color frames
                align = rs.align(rs.stream.color)

                # Get the intrinsics information for calculation of 3D point
                unaligned_frames = self.pipeline.wait_for_frames()
                frames = align.process(unaligned_frames)
                depth = frames.get_depth_frame()
                depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics
                # Initialize the cubemos api with a valid license key in default_license_dir()

                joint_confidence = 0.25

                window_name = "HAR"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        
                with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,model_complexity = 1) as pose:
                    while True:
                    # Create a pipeline object. This object configures the streaming camera and owns it's handle
                        unaligned_frames = self.pipeline.wait_for_frames()
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

                    #cm.render_result(skeletons, color_image, joint_confidence)
                        results = pose.process(color_image)
                        skeleton = []
                        if not results.pose_landmarks:
                            frame_count += 1
                            f +=1
                            if f >= 4 and in_bed :
                                act = 'Sleeping'
                                #print(in_bed)
                                timesleep = [i for i in range(33)]
                                timesleep.append(timecheck())
                                startend(timesleep)
                                f = 0
                            cv2.putText(im_color,' Activity = ' +act, (200,100) , cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255),2 , cv2.LINE_4)
                            cv2.imshow(window_name, im_color)
                            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                                break
                            continue
                        # Draw the pose annotation on the image.
                        color_image.flags.writeable = True
                        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                        mp_drawing.draw_landmarks(
                            im_color,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                        skeleton = results.pose_landmarks.landmark            
                        joint_datatata = render_ids_3d(color_image,skeleton , depth, depth_intrinsic)
                        if check_num_joint(joint_datatata) < 6 :
                            f +=1
                            if f >= 4 and in_bed :
                                act = 'Sleeping'
                                timesleep = [i for i in range(33)]
                                timesleep.append(timecheck())
                                startend(timesleep)
                                f = 0
                        else :
                            f = 0
                            ## low pass filter raw data --> down sampling --> do the recognition
                            moving_average(joint_datatata)
                            startend(joint_datatata)

                        frame_count += 1
                        
                        cv2.putText(im_color,' Activity = ' +act, (100,650) , cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255),2 , cv2.LINE_4)
                        if act != ' ':
                            if act == 'Standing' :
                                activity_count[0] +=1
                            elif act == 'Walking' :
                                activity_count[1] += 1
                            elif act == 'Sitting' :
                                activity_count[2] += 1
                            elif act == 'Sleeping' :
                                activity_count[3] += 1
                            elif act == 'Falling down' :
                                activity_count[4] += 1 
                                
                            for i in range(5) :
                                percent_activity[i] = activity_count[i] / sum(activity_count) *100
                            ax.clear()
                            bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:olive'] #purple, brown, pink, cyan 
                            ax.bar(activity_list, percent_activity, color= bar_colors)
                            ax.set_ylabel('percentage of each activity',fontsize = 16)
                            ax.set_ylim(0, 100)

                            # Set tick font size
                            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                                label.set_fontsize(12)
                            # ax.pie(activity_count, labels=activity_list,colors=[color_pie[key] for key in activity_list] ,autopct = '%1.1f%%',startangle=90)
                            # ax.axis('equal')
                            
                            fig.canvas.draw()

                            chart_img = np.array(fig.canvas.renderer.buffer_rgba())
                            chart_img = cv2.resize(chart_img, (im_color.shape[1]//3, im_color.shape[0]//3))

                            mask = (chart_img[:, :, 0] != 255).astype(np.uint8) * 255


                            # Overlay the chart on the frame with transparency
                            
                            chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)
                            im_color[0:chart_img.shape[0], 0:chart_img.shape[1], :] = cv2.addWeighted(im_color[0:chart_img.shape[0], 0:chart_img.shape[1], :], 0, chart_img, 1, 0)
                            #im_color = cv2.addWeighted(im_color, 0.8, chart_img[:,:,:3],0.2,0)
                        cv2.putText(color_image,' Activity = ' +act, (100,650) , cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255),2 , cv2.LINE_4)    
                        cv2.imshow(window_name, im_color)
                        #print(x)

                        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                                end = 1
                                pieplot_color()
                                table()
                                ax.clear()
                                joint_dataa = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
                                ## Time
                                qt1 = []
                                qt2 = []
                                ##durtion dur = [sit_dur,stand_dur,walk_dur,sleep_dur]
                                dur = [0,0,0,0,0]
                                ##global activity
                                ##[sit,stn,wlk,sleep]
                                act_in = [0,0,0,0,0,0]
                                ##List for checking activity
                                a = ['Sitting','Standing','Walking','Sleeping','Falling down','Waking up',' ']
                                ## explode for plot
                                explodee=[0, 0, 0, 0]
                                ## data for table
                                data = []
                                calibrate_point = []
                                in_bed = False
                                
                                break

            except Exception as e:
                print(f"Exception in thread: {e}")
            finally:
                self.is_streaming = False
                if self.pipeline:
                    self.pipeline.stop()
                cv2.destroyAllWindows()
                

    def cali_rt_bed_boundary(self):
        global bed_boundary
        if not self.is_streaming:
            config = rs.config()
            # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
            config.enable_stream(rs.stream.depth,1280,720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color,1280,720, rs.format.rgb8, 30)
            # Start the realsense pipeline
            pipeline = rs.pipeline()
            #Start streaming from file
            pipeline.start(config)
            # Create align object to align depth frames to color frames
            align = rs.align(rs.stream.color)
            #Get color frame for calibtation
            unaligned_frames = pipeline.wait_for_frames()
            frames = align.process(unaligned_frames)
            color = frames.get_color_frame()
            color_image = np.asanyarray(color.get_data())
            color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
            depth = frames.get_depth_frame()
            depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics
            #cv2.imshow('Sample video',color_image)
            bed_boundary = set_bed_boundary(color_image,depth,depth_intrinsic)
            print(bed_boundary)
    
    def cali_rt_stand_walk(self):
        global walk_first_threshold,walk_second_threshold
        file_to_run = 'rt_caliwalk.py'
        e = None
        try:
            result = subprocess.run(['python', file_to_run], capture_output=True, text=True, check=True)
            output_lines = result.stdout.strip().split('\n')  # Split the output into lines

            if len(output_lines) == 2:
                walk_second_threshold = float(output_lines[0])  # Convert the first line to float
                walk_first_threshold = float(output_lines[1])    # Convert the second line to int

            print('Calibration stand-walk threshold:\n' + 'Nose threshold :' + str(walk_first_threshold)+ 
                  '(\n Old : 0.024 \nFoot threshold :' + str(walk_second_threshold) + '\n Old : 0.035')
        except subprocess.CalledProcessError as error:
            e = error
            print("Error:", e)
            print("Output:", e.output)

    def cali_rt_stand_sit(self):
        global sit_right_threshold, sit_left_threshold
        file_to_run = 'rt_calisit.py'
        e = None
        try:
            result = subprocess.run(['python', file_to_run], capture_output=True, text=True, check=True)
            print("Output:", result.stdout)
            output_lines = result.stdout.strip().split('\n')  # Split the output into lines

            if len(output_lines) == 2:
                sit_left_threshold= float(output_lines[0])  # Convert the first line to float
                sit_right_threshold = float(output_lines[1])    # Convert the second line to int

            print('Calibration stand-sit threshold:\n' + 'right threshold :' + str(sit_right_threshold)+ 
                  '\nleft threshold :' + str(sit_left_threshold) + '\n Old : 0.1875')
        except subprocess.CalledProcessError as error:
            e = error
            print("Error:", e)
            print("Output:", e.output)

    def cali_rt_stand_sleep(self):
        global sleep_s_h_threshold, sleep_f_h_threshold
        file_to_run = 'rt_calisleep.py'
        e = None
        try:
            result = subprocess.run(['python', file_to_run], capture_output=True, text=True, check=True)
            print("Output:", result.stdout)
            output_lines = result.stdout.strip().split('\n')  # Split the output into lines

            if len(output_lines) == 2:
                sleep_s_h_threshold = float(output_lines[0])  # Convert the first line to float
                sleep_f_h_threshold = float(output_lines[1])    # Convert the second line to int

            print('Calibration stand-sleep threshold:\n' + 'shoulder-hip threshold :' + str(sleep_s_h_threshold)+ 
                  '(\n Old : 0.22 \nheel-hip threshold :' + str(sleep_f_h_threshold) + '\n Old : 0.25')
        except subprocess.CalledProcessError as error:
            e = error
            print("Error:", e)
            print("Output:", e.output)
        


    def on_closing(self):
        if self.is_streaming:
            self.is_streaming = False
            if self.pipeline:
                self.pipeline.stop()
            cv2.destroyAllWindows()
        self.root.destroy()

    


if __name__ == "__main__":
    root = tk.Tk()
    app = RealSenseDepthStreamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle window close event
    root.mainloop()


