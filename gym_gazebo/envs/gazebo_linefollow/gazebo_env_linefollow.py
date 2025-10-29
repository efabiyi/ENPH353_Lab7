
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



    # TODO: Analyze the cv_image and compute the state array and
    # episode termination condition.
    #
    # The state array is a list of 10 elements indicating where in the
    # image the line is:
    # i.e.
    #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
    #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
    #
    # The episode termination condition should be triggered when the line
    # is not detected for more than 30 frames. In this case set the done
    # variable to True.
    #
    # You can use the self.timeout variable to keep track of which frames
    # have no line detected.

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

        #cv2.imshow("raw", cv_image)

        NUM_BINS = 10
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False
        uniform_area_detected = False
        
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
        
            if roi_variance < 20:  # Adjust this threshold as needed
                uniform_area_detected = True
                continue  # Skip this region as it's too uniform to detect the line

            contours,_ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            area = 0 
            for contour in contours:
                area += cv2.contourArea(contour)

            if area > max_area:
                max_area = area
                max_index = i

        if uniform_area_detected:
            self.timeout += 1

            # If max_index is outside the center region, penalize
            if max_index not in (2, 3, 4, 5, 6):
                state[max_index] = 1  # The robot is still following a line but in a side region
                self.timeout += 1  # Penalize with timeout increment
            else:
                state[max_index] = 1  # The robot is in the center of the line
                self.timeout = 0  # Reset timeout if it's in the center
        else:
            # No line detected in the frame
            self.timeout += 1

        # If timeout exceeds a threshold (e.g., 20 frames with no line detected), end the episode
        if self.timeout >= 10:
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
        num_bins = len(state)

        # Draw a rectangle in each bin if state indicates line presence
        bin_width = width // num_bins
        for i in range(num_bins):
            if state[i] == 1:
                x_start = i * bin_width
                x_end = (i + 1) * bin_width
                cv2.rectangle(image, (x_start, 0), (x_end, height), (0, 255, 0), 2)  # Green rectangle

        # Optionally, add text to indicate state
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

        if action == 0:  # Move forward slowly
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = 0.0
        elif action == 1:  # Move forward quickly
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0.0
        elif action == 2:  # Turn left slowly
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.2
        elif action == 3:  # Turn left moderately
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 4:  # Turn left quickly
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.8
        elif action == 5:  # Turn right slowly
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.2
        elif action == 6:  # Turn right moderately
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5
        elif action == 7:  # Turn right quickly
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.8


        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # Move forward slowly
                reward = 2  # Small reward for slow forward movement
            elif action == 1:  # Move forward quickly
                reward = 3.5  # Larger reward for quick forward movement
            elif action == 2:  # Turn left slowly
                reward = 2.5  # Small reward for slow left turn
            elif action == 3:  # Turn left moderately
                reward = 3.5  # Moderate reward for moderate left turn
            elif action == 4:  # Turn left quickly
                reward = 5.5  # Larger reward for sharp left turn
            elif action == 5:  # Turn right slowly
                reward = 2  # Small reward for slow right turn
            elif action == 6:  # Turn right moderately
                reward = 3  # Moderate reward for moderate right turn
            elif action == 7:  # Turn right quickly
                reward = 5  # Larger reward for sharp right turn
        else:
            reward = -200

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
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
