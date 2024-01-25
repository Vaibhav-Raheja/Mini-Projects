import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode
import math
import copy

import random
# python3 main.py --num_particles 1000 --sensor_limit 15 --measurement_noise True


def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        particles = list()

        ##### TODO:  #####
        # Modify the initial particle distribution to be within the top-right quadrant of the world, and compare the performance with the whole map distribution.
        for i in range(num_particles):

            # (Default) The whole map
            # x = np.random.uniform(0, world.width)
            # y = np.random.uniform(0, world.height)


            ## first quadrant
            x = np.random.uniform(world.width/2, world.width)
            y = np.random.uniform(world.height/2, world.height/2)
            # #y = np.random.uniform(0, world.height/2)

            particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))
            self.prev_time = rospy.get_time()
            # self.file_object_results  = open("8_1000_15.csv", "w+")

        ###############
        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle
        self.particle_weight_total = 0

        return

    def eulidean_distance(self,x1,x2):
        return np.linalg.norm(np.asarray(x1) - np.asarray(x2))

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """

        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self,x1, x2, std = 5000):
        if x1 is None: # If the robot recieved no sensor measurement, the weights are in uniform distribution.
            return 1./len(self.particles)
        else:
            tmp1 = np.array(x1)
            tmp2 = np.array(x2)
            return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))


    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """

        ## TODO #####
        # Initialize a list to store the updated weights for each particle.
        particle_weights = []
        
        for particle in self.particles:
        # Calculate the weight for the particle using the Gaussian Kernel
            readings_particle = particle.read_sensor()
            particle.weight = self.weight_gaussian_kernel(readings_robot , readings_particle,std=5000)
            particle_weights.append(particle.weight)
            self.particle_weight_total += particle.weight
        
        for particle in self.particles:
            particle.weight = particle.weight / self.particle_weight_total


        
        return particle_weights
         
        ###############
        # pass

    def resampleParticle(self,weight):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()
        

        ## TODO #####

        accum = 0
        distribution = list()
        indices = 0
        # for particle in self.particles:
        #     accum += particle.weight
        #     distribution.append(accum)
        distribution = np.cumsum(weight)
        total_weight = distribution[-1]
        distribution = distribution / total_weight

        for particle in self.particles:
            a = np.random.uniform(0, distribution[-1])
            indices = np.searchsorted(distribution,a)
            particles_new.append(Particle(self.particles[indices].x, self.particles[indices].y, self.particles[indices].maze,
                                          self.particles[indices].heading, self.particles[indices].weight, 
                                          sensor_limit = self.sensor_limit, noisy = True))





        self.particles = particles_new

    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
        """
        ## TODO #####

        if len(self.control)>0:
            dt = 0.01
            delta = self.control[-1][1]
            vr = self.control[-1][0]
            # print(vr, delta)
            for particle in self.particles:
                var = [particle.x, particle.y, particle.heading]
                r = ode(vehicle_dynamics,jac=None)
                r.set_initial_value(var,0)
                r.set_f_params(vr,delta)
                ret = r.integrate(2.2*dt)
               
                particle.x = ret[0]
                particle.y = ret[1]
                particle.heading = ret[2]#% (np.pi*2)

        ###############
        # pass


    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        count = 0 
        while True:
            ## TODO: (i) Implement Section 3.2.2. 
            
            self.particleMotionModel()
            readings_robot = self.bob.read_sensor()
            weight = self.updateWeight(readings_robot)
            # print(self.particle_weight_total)
            if self.particle_weight_total == 0:
                self.particle_weight_total = 1e-8
                
            self.resampleParticle(weight)
            

            # (ii) Display robot and particles on map. 
            self.world.show_particles(particles = self.particles, show_frequency = 7)
            self.world.show_robot(robot = self.bob)
            # print("size of particles list: ", len(self.particles))
            [est_x,est_y, est_orientation] = self.world.show_estimated_location(particles = self.particles)
            # print(self.bob.x)
            # print(self.bob.y)
            # print(self.getModelState())
            self.world.clear_objects()
            error = self.eulidean_distance((self.bob.x,self.bob.x),(est_x,est_y))
            # error_y = self.eulidean_distance(self.bob.y,est_y)
            # print(count,error)
            count += 1 
            # (iii) Compute and save position/heading error to plot. #####
            # self.file_object_results.write(str(count) + "," + str(error) +"\n")
