"""

Idea: select the potential way to go to using lidar. Pick one way and keep going untill it is potential no more.
Important implementation:
    - A potential function that extracts out the good direction to pick, along with the type of direction.
    - A function that handle the going part
    - A human handling that can handle finding and grasping humans up
    - Flag states for going to directions, wheather it sees human, wheather it grasps human, wheather it drops it.
Hopefully this would work. We are very close to the deadline right now.

"""
import math
import random
import matplotlib.pyplot as plt
from typing_extensions import ParamSpecKwargs
import numpy as np
from typing import Optional
import os
import sys
from typing import Type

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.icp import plot_points, icp_matching, plot_points_3plots
from tools.utils import enlarge
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle

# self.base.grasper.grasped_entities
class DroneSolutionV1(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None, debug = False,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         should_display_lidar=debug,
                         **kwargs)
        self.debug = debug
        self.start = True
        self.state = 0
        self.flag = 0
        self.lock = 0
        self.counter = 0
        self.goal = [0, 0]

        self.angle1 = 0
        self.position1 = [0, 0] 

        self.limit = 0
        self.pending = 0
        self.data = []

        self.step_count = 0
        self.scale = 3
        self.occupancy_map_size = 100
        self.occupancy_map = np.zeros((self.occupancy_map_size, self.occupancy_map_size))
        self.occupancy_map_dx = 0
        self.occupancy_map_dy = 0

        self.prev_angle = 0
        self.prev_position = [0, 0]
        # state: finding people or taking people back
        # flag: to help with going

        self.list_gps = [[], []]
        self.list_odo = [[], []]

    def init_dxy(self):
        # set dx and dy
        x, y = self.measured_gps_position()
        self.occupancy_map_dx = -int(x)//self.scale+self.occupancy_map_size//2
        self.occupancy_map_dy = -int(y)//self.scale+self.occupancy_map_size//2

    def process_touch_sensor(self):
        """
        Returns True if the drone hits an obstacle
        """
        if self.touch().get_sensor_values() is None:
            return False

        touched = False
        detection = max(self.touch().get_sensor_values())

        if detection > 0.3:
            touched = True

        return touched
    
    def measured_fake_angle(self):
        return self.angle1
        return self.measured_compass_angle() or self.angle1

    def measured_fake_position(self):
        return self.position1
        return self.measured_gps_position() or self.position1

    def rotation_matrix(self, angle):
        return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    def mean_square_error(self, lidar1, lidar2):
        return np.sum((lidar1 - lidar2)**2)/len(lidar1)

    def calc_icp(self, lidar_from_occupancy, lidar_from_measure):
        """
        Correct the new measurement to align it to the previous measurement.
        Need:
        - Absolute value of lidar_from_occupancy
        - Absolute value of lidar_from_measure
        -> ICP translate from lidar_from_occupancy to the lidar_from_measure 
        Return:
        - Rotational matrix 
        - Translation matrix
        """
        X_prev, Y_prev = self.get_absolute_position_lidar(lidar_from_measure, [0, 0], 0, threshold=400)
        X_cur, Y_cur = self.get_absolute_position_lidar(lidar_from_occupancy, [0, 0], 0, threshold=400)
        previous_points = np.vstack((X_prev, Y_prev))
        current_points = np.vstack((X_cur, Y_cur))
        R, t = icp_matching(previous_points, current_points)

        return R, t

        # np_position = np.array(self.position1)
        # np_position = self.rotation_matrix(alpha)@np_position


        # print("movement:", previous_points[0][0] - (self.rotation_matrix(alpha)@current_points)[0][0], previous_points[1][0] - (self.rotation_matrix(alpha)@current_points)[1][0], T)
        # print(alpha)

    # def find_best_translation(self, lidar1, lidar2):


    def fix_fake_position(self, tolerance=2):
        if self.start: # self.measured_compass_angle() != None:
            self.angle1 = self.measured_compass_angle()
            a, b = self.measured_gps_position()
            self.position1 = [a, b]     
        else:
            # update angle1 and position1
            odo = self.odometer_values()
            dx, dy, d_angle = self.process_odometer(odo)

            # print(odo)
            # if self.step_count%20==3: input()
            # print("odo", odo)

            self.angle1 += d_angle
            self.position1[0] += dx
            self.position1[1] += dy

            # tmp_angle = self.angle1 + d_angle
            # tmp_position = self.position1[0] + dx, self.position1[1] + dy

            # localization
            lidar_from_occupancy = self.get_lidar_from_occupancy(self.position1, self.angle1)
            lidar_from_measure = self.lidar().get_sensor_values()

            X_prev, Y_prev = self.get_absolute_position_lidar(lidar_from_measure, [0, 0], 0, threshold=400)
            X_cur, Y_cur = self.get_absolute_position_lidar(lidar_from_occupancy, [0, 0], 0, threshold=400)

            previous_points = np.vstack((X_prev, Y_prev))
            current_points = np.vstack((X_cur, Y_cur))
            R, t = self.calc_icp(lidar_from_occupancy, lidar_from_measure)

            if self.step_count > 45:
                print(R)
                print(t)
                print(self.angle1)
                print(self.position1)
                print(self.measured_compass_angle())
                print(self.measured_gps_position())
                assert False
                self.angle1 += np.arctan(R[1][0]/R[0][0])
                self.position1[0] += t[0]
                self.position1[1] += t[1]


            # if self.step_count > 100 and self.step_count % 50 == 2:
            #     figure = plt.figure()
            #     tmp_points = R@current_points
            #     tmp_points[0] += t[0]
            #     tmp_points[1] += t[1]
            #     plot_points_3plots(previous_points, tmp_points, current_points, figure)
            #     plt.pause(0.1)
            #     print("cur_point", current_points[0][:20], current_points[1][:20])
            #     print("Value of t", t, self.measured_gps_position()[0] - self.position1[0], self.measured_gps_position()[1] - self.measured_gps_position()[1])
            #     input()



            # print("T", self.measured_gps_position()[0] - self.position1[0], self.measured_gps_position()[1] - self.position1[1], T)
            
            # print(self.angle1, self.measured_compass_angle())
            # print(self.position1, self.measured_gps_position())

            # ////// Code to update gps_list and odo_list
            #
            # x1, y1 = self.measured_gps_position()
            # x2, y2 = self.position1
            # self.list_gps[0].append(x1)
            # self.list_gps[1].append(y1)
            # self.list_odo[0].append(x2)
            # self.list_odo[1].append(y2)
            # if self.step_count%10 == 1:
            #     plt.scatter(self.list_gps[0], self.list_gps[1], label='gps')
            #     plt.scatter(self.list_odo[0], self.list_odo[1], label='guessed')
            #     plt.legend()
            #     plt.show()
            
    def todegree(self, angle):
        return angle*180/math.pi
    
    def torad(self, i):
        return (i-90)/90*math.pi

    def potential(self, lidar, tolerance=120, reverse = False):
        output = []
        for i in range(180):
            bias = (120-abs(90-i)) if (not reverse) else (30 + abs(90-i))
            if lidar[i]-lidar[i-1] >= tolerance:
                output.append([self.torad(i), 1, bias**2])
                continue
            if lidar[i-1]-lidar[i] >= tolerance:
                output.append([self.torad((i-1)%180), -1, bias**2])
                continue
            if lidar[i] > 300:
                if lidar[i-1] < 295 or lidar[(i+1)%180] < 295:
                    output.append([self.torad(i), 0, bias**2])
        return output if output else [[0, 0, 1]]

    def human(self, semantic):
        if semantic is None: return False, [0, 0]
        for data in semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                return True, [data.angle, 0]
        return False, [0, 0]

    def center(self, semantic):
        if semantic is None: return False, [0, 0]
        for data in semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                return True, [data.angle, 0]
        return False, [0, 0]

    def deadend(self, lidar, tolerance = 100):
        if max(lidar[80:101]) < tolerance: return True
        if self.base.grasper.grasped_entities: return False
        if min(lidar[75:81]) < 20: return True
        if min(lidar[100:106]) < 20: return True
        return False

    def new(self, lidar, semantic):
        if self.pending: return False, [0, 0]

        if self.lock or self.flag != 1: return False, [0, 0]

        if self.state == 0:
            a, b = self.human(semantic)
            if a:
                self.lock = 1
                return True, b

        if self.state == 1:
            a, b = self.center(semantic)
            if a:
                self.lock = 1
                return True, b

        if self.data != []:
            lidar[90:135] = self.data[:45]
            lidar[45:90] = self.data[135:]
            potential = self.potential(lidar, reverse = True)
            self.data = []
            return True, random.choices(potential, [i[-1] for i in potential], k=1)[0]

        if self.deadend(lidar):
            if self.base.grasper.grasped_entities and self.data == []:
                self.pending = 1
                self.data = self.lidar().get_sensor_values()[:-1].copy()
                return True, [-math.pi, 0]
            potential = self.potential(lidar)
            return True, random.choices(potential, [i[-1] for i in potential], k=1)[0]

        return False, [0, 0, 1]

    def new_global(self, lidar, semantic):
        a, b = self.new(lidar, semantic)
        if not a: return False, [0, 0]
        b = [normalize_angle(b[0]+float(self.measured_fake_angle())), b[1]]
        return True, b
    
    def pos_to_grid(self, pos):
        return (int(pos[0])//self.scale+self.occupancy_map_dx,
                int(pos[1])//self.scale+self.occupancy_map_dy)

    def get_absolute_position_lidar(self, lidar, current_position, current_angle, threshold=290):
        angles = np.array(self.lidar().ray_angles) + current_angle
        distances = np.array(lidar)
        edge_distance_positions = []
        for (i, dist) in enumerate(distances):
            if dist <= threshold: # maximum lidar - gaussian noise
                edge_distance_positions.append(i)
        edge_distances = np.array([distances[i] for i in edge_distance_positions])
        edge_angles = np.array([angles[i] for i in edge_distance_positions])
        cur_x, cur_y = current_position
        # Absolute position of lidar points on the map
        x = edge_distances * np.cos(edge_angles) + cur_x
        y = edge_distances * np.sin(edge_angles) + cur_y    
        return (x, y)

    def gps_mapping(self):
        # print(self.step_count)
        self.step_count += 1
        if self.lidar().get_sensor_values() is not None:
            data = set()
            x, y = self.get_absolute_position_lidar(self.lidar().get_sensor_values(), self.measured_fake_position(), self.measured_fake_angle())
            for i in range(len(x)):
                cur_x, cur_y = self.pos_to_grid((x[i], y[i]))
                while not (0 < cur_x < len(self.occupancy_map)-1 and 0 < cur_y < len(self.occupancy_map[0])-1):
                    self.occupancy_map, del_x, del_y = enlarge(self.occupancy_map)
                    self.occupancy_map_dx += del_x                    
                    self.occupancy_map_dy += del_y
                    cur_x, cur_y = self.pos_to_grid((x[i], y[i]))
                    print(cur_x, cur_y, self.occupancy_map.shape)
                self.occupancy_map[cur_x][cur_y] += 1
                # for dx, dy in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
                #     self.occupancy_map[cur_x+dx][cur_y+dy] /= 2

            # # Heatmap for occupancy maps
            # if self.step_count % 10 == 0:
            #     plt.imshow(self.occupancy_map.T, cmap='hot', interpolation='nearest')
            #     plt.gca().invert_yaxis()
            #     # plt.ylim(ymin=0)
            #     plt.draw()
            #     plt.pause(0.001)

    def process_odometer(self, odo):
        """
        Return dx, dy, d_angle
        """
        dist, alpha, theta = odo
        tmp_angle = self.measured_fake_angle() + alpha
        dx = dist*math.cos(tmp_angle)
        dy = dist*math.sin(tmp_angle)
        return dx, dy, theta

    def get_lidar_from_occupancy(self, position, angle, threshold = 0.5):
        lidar = np.array([300 for _ in range(181)])
        # cur_cell = self.pos_to_grid(position)

        for i in range(0, 181):
            rad_angle = math.pi + angle + math.radians(2*i)
            cos_angle = math.cos(rad_angle)
            sin_angle = math.sin(rad_angle)
            try:
                for d in range(1, 302):
                    next_position = (position[0] + d*cos_angle, position[1] + d*sin_angle)
                    next_cell = self.pos_to_grid(next_position)
                    if self.occupancy_map[next_cell[0]][next_cell[1]] > threshold:
                        lidar[i] = d
                        break
            except:
                lidar[i] = 301
                
        # print(angle, '\n', lidar, '\n', self.lidar().get_sensor_values())
        # print([int(i-j) for i, j in zip(lidar, self.lidar().get_sensor_values())])
        # input()

        # plt.figure(self.SensorType.LIDAR)
        # # plt.figure(lidar)
        # plt.cla()
        # plt.axis([-500, 500, -500, 500])
        # # plt.plot(x, y, "g.:")
        # plt.grid(True)
        # plt.draw()
        # plt.pause(0.001)
        return lidar

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        self.fix_fake_position()

        self.gps_mapping()
        if self.step_count % 10 == 0:
            self.get_lidar_from_occupancy(self.measured_fake_position(), self.measured_fake_angle())
        
        if self.start:
            self.start = False
            self.init_dxy()
            return command

        lidar = np.array(self.lidar().get_sensor_values())
        lidar = lidar[:-1]

        semantic = self.semantic().get_sensor_values()

        # first, we find a command to work toward the goal
        if self.state or (self.lock and self.flag): command["grasper"] = 1

        if self.pending == 1:
            command["forward"] = 0.0
            diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle())
            command["rotation"] = 1
            if abs(diff_angle) < 0.2:
                self.flag = 0
                self.pending = 0

        elif self.goal[1] == 0:
            if self.flag == 0:
                command["forward"] = 0.0
                diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle())
                command["rotation"] = -1.0 if diff_angle < 0 else 1.0
                if abs(diff_angle) < 0.2:
                    self.flag = 1
            else:
                if self.lock == 1: self.counter += 1
                if self.counter >= 10:
                    self.counter = 0
                    self.lock = 0

        elif self.goal[1] == 1:
            if self.flag == 0:
                command["forward"] = 0.0
                diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle() + math.pi/4)
                command["rotation"] = -1.0 if diff_angle < 0 else 1.0
                if abs(diff_angle) < 0.2:
                    self.flag = 2
            else:
                self.counter += 1
                if self.counter >= 12:
                    self.flag = 0
                    self.goal[1] = 0

        elif self.goal[1] == -1:
            if self.flag == 0:
                command["forward"] = 0.0
                diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle() - math.pi/4)
                command["rotation"] = -1.0 if diff_angle < 0 else 1.0
                if abs(diff_angle) < 0.2:
                    self.flag = 2
            else:
                self.counter += 1
                if self.counter >= 12:
                    self.flag = 0
                    self.counter = 0
                    self.goal[1] = 0

        # then, we update our state
        if self.state == 0 and self.base.grasper.grasped_entities:
            self.lock = 0
            self.state = 1

        elif self.state == 1 and not self.base.grasper.grasped_entities:
            self.lock = 0
            self.state = 0
        
        # then, call new
        a, b = self.new(lidar, semantic)
        if a:
            self.goal = [normalize_angle(b[0]+self.measured_fake_angle()), b[1]]
            self.flag = 0
            self.counter = 0
            if self.debug:
                print('new goal!', b)
                input()
            self.limit = 0

        self.limit += 1
        if self.limit >= 180:
            print('limit reached!')
            a, b = self.new(lidar, semantic)
            self.goal = [normalize_angle(b[0]+self.measured_fake_angle()), b[1]]
            self.limit = 0
            self.state = 0
            self.flag = 0
            self.counter = 0
        
        # print(self.prev_position, self.measeured_fake_position())
        # lidar_from_occupancy = self.get_lidar_from_occupancy(self.measured_fake_position(), self.measured_fake_angle())
        # lidar_from_measure = self.lidar().get_sensor_values()
        # print(self.calc_icp(lidar_from_occupancy, lidar_from_measure))

        #update prev_angle
        # return
        return command