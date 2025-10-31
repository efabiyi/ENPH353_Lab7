
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

        self.data = None
        self.image_sub = rospy.Subscriber("pi_camera/image_raw", Image, self.callback)
    
    def callback(self, msg):
        self.data = msg
        rospy.logdebug("Image callback received. timestamp: %s", str(msg.header.stamp))

    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


        NUM_BINS = 9
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False
        no_line_detected = True
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        height, width = gray.shape

        roi_w = width // NUM_BINS
        roi_h = int(height*0.75)

        max_index = -1
        max_area = 0

        for i in range(NUM_BINS):
            x_start = i*roi_w
            x_end = (i+1) * roi_w if i < (NUM_BINS-1) else width
            roi = binary_img[roi_h:, x_start:x_end]

            roi_variance = np.var(roi)
        
            if roi_variance < 20: 
                continue  # Skip this region as it's too uniform to detect the line, penalize later

            contours,_ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            area = 0 
            for contour in contours:
                area += cv2.contourArea(contour)

            if area > max_area:
                max_area = area
                max_index = i

            no_line_detected = False

        if no_line_detected:
             # No line detected in the frame!! Timeout!
            self.timeout += 1
        else:
            state[max_index] = 1  # The robot is in the center of the line
            self.timeout = 0  # Reset timeout if it's in the center

        # If timeout exceeds a threshold end the episode
        if self.timeout >= 30:
            done = True



        self.overlay_state_on_image(cv_image, state)
        cv2.imshow("Camera Feed", cv_image)
        cv2.waitKey(1)

        return state, done

    def overlay_state_on_image(self, image, state):
        '''
            @brief Overlays the state information (line position) onto the camera image
            @param image : The OpenCV image to overlay the state onto
            @param state : The state array representing where the line is
        '''
        height, width, _ = image.shape

        start_point = (0, int(height*0.75))    
        end_point = (width, int(height*0.75))
        
        num_bins = len(state)

        # Draw a rectangle in each bin if state indicates line presence
        bin_width = width // num_bins

        for i in range(num_bins):
            if state[i] == 1:
                x_start = i * bin_width
                x_end = (i + 1) * bin_width
                cv2.rectangle(image, (x_start, 0), (x_end, height), (0, 255, 0), 2)  # Green rectangle
                cv2.line(image,start_point, end_point,(255,0,0),2 )

        # Add text to indicate state
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "State: " + ''.join([str(s) for s in state])
        cv2.putText(image, text, (10, 30), font, 0.8, (0, 0, 0), 2)


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # Move forward 
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0.0

        elif action == 1:  # Turn left 
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = 0.5

        elif action == 2:  # Turn right
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)
        
        wait_time = 0
        while self.data is None and wait_time < 5:
            rospy.sleep(0.1)
            wait_time += 0.1

        if self.data is None:
            rospy.logwarn("No image data received from camera.")
            done = True
            return [0]*10, -200, done, {} 

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(self.data)

        line_index = np.argmax(state) if np.any(state) else -1
        centre_bins = [3,4,5]

        # Set the rewards for your action
        if not done:
            reward = -0.1
            #if you're in the centre you get rewarded, else you're done for!
            if line_index in centre_bins:
                if action == 0:  # Move forward slowly
                    reward = 5  

                elif action == 1:  # Move left 
                    reward =  0 

                elif action == 2:  # Move right
                    reward = 0  
            else:
                if action == 0:  
                    reward = -20  

                elif action == 1: 
                    reward = -20  

                elif action == 2:  
                    reward = -20 
        else:
            reward = -200
        
        rospy.sleep(0.001)

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data

        wait_time = 0
        while self.data is None and wait_time < 5:
            rospy.sleep(0.1)
            wait_time += 0.1

        if self.data is None:
            rospy.logwarn("No image data received from camera.")
            done = True
            return [0]*10, -200, done, {} 

        rospy.wait_for_service('/gazebo/pause_physics')

        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(self.data)

        return state
