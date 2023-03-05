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
import scipy
from collections import deque

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.icp import icp_matching
from tools.utils import enlarge
import tools.dstar as ds
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
NON_DISCOVERED = -10
DRONE_RADIUS = 5

class DroneSolutionV2(DroneAbstract):
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
        self.scale = 5
        self.occupancy_map_size = 300
        self.occupancy_map = np.full((self.occupancy_map_size, self.occupancy_map_size), NON_DISCOVERED)
        self.occupancy_map_dx = 0
        self.occupancy_map_dy = 0

        
        self.goal_explore_angle = 1e9
        self.center_location = None

        self.prev_angle = 0
        self.prev_position = [0, 0]
        # state: finding people or taking people back
        # flag: to help with going

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
        return self.measured_compass_angle() or self.angle1

    def measured_fake_position(self):
        return self.measured_gps_position() or self.position1

    def fix_fake_position(self):
        if False: # self.measured_compass_angle() != None:
            self.angle1 = self.measured_compass_angle()
            a, b = self.measured_gps_position()
            self.position1 = [a, b]     
        else:
            # update angle1 and position1
            odo = self.odometer_values()
            self.angle1 += odo[-1]
            self.position1[0] += odo[0]
            self.position1[1] += odo[1]

            # localization
            lidar_from_occupancy = self.get_lidar_from_occupancy(self.position1, self.angle1)
            lidar_from_measure = self.lidar().get_sensor_values()
            R, t = self.correct(lidar_from_occupancy, lidar_from_measure)
            self.position1[0] += t[0]
            self.position1[1] += t[1]
            self.angle1 += np.arctan(R[0][0]/R[1][0])
            


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
        SCALED_DRONE_RADIUS = DRONE_RADIUS // self.scale
        if self.lidar().get_sensor_values() is not None:
            # Update expected wall
            x, y = self.get_absolute_position_lidar(self.lidar().get_sensor_values(), self.measured_fake_position(), self.measured_fake_angle())
            for i in range(len(x)):
                cur_x, cur_y = self.pos_to_grid((x[i], y[i]))
                while not (0 <= cur_x < len(self.occupancy_map) and 0 <= cur_y < len(self.occupancy_map[0])):
                    self.occupancy_map, del_x, del_y = enlarge(self.occupancy_map)
                    self.occupancy_map_dx += del_x                    
                    self.occupancy_map_dy += del_y
                    cur_x, cur_y = self.pos_to_grid((x[i], y[i]))
                    # print(cur_x, cur_y, self.occupancy_map.shape)
                for wall_x in range(cur_x - SCALED_DRONE_RADIUS, cur_x + SCALED_DRONE_RADIUS):
                    for wall_y in range(cur_y - SCALED_DRONE_RADIUS, cur_y + SCALED_DRONE_RADIUS):
                        if not (0 <= wall_x <= len(self.occupancy_map) and 0 <= wall_y <= len(self.occupancy_map[0])):
                            continue
                        if self.occupancy_map[wall_x][wall_y] >= 0:
                            self.occupancy_map[wall_x][wall_y] = self.occupancy_map[wall_x][wall_y] + 1
                        else:
                            self.occupancy_map[wall_x][wall_y] = 1

            # Update expected empty
            if self.step_count % 10 == 0:
                position = self.measured_fake_position()
                angle = self.measured_fake_angle()
                for i in range(0, 181):
                    rad_angle = math.pi + angle + math.radians(2*i)
                    cos_angle = math.cos(rad_angle)
                    sin_angle = math.sin(rad_angle)
                    for d in range(1, 302):
                        next_position = (position[0] + d*cos_angle, position[1] + d*sin_angle)
                        next_cell = self.pos_to_grid(next_position)
                        if self.occupancy_map[next_cell[0]][next_cell[1]] >= 1:
                            break
                        else:
                            self.occupancy_map[next_cell[0]][next_cell[1]] = 0

            # Heatmap for occupancy maps
            if self.step_count % 100 == 0:
                plt.imshow(self.occupancy_map.T, cmap='hot', interpolation='nearest')
                plt.gca().invert_yaxis()
                plt.draw()
                plt.pause(0.001)

    def get_lidar_from_occupancy(self, position, angle, threshold = 0):
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

    def correct(self, lidar_from_occupancy, lidar_from_measure):
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
        X_prev, Y_prev = self.get_absolute_position_lidar(lidar_from_measure, (0, 0), 0, threshold=400)
        X_cur, Y_cur = self.get_absolute_position_lidar(lidar_from_occupancy, (0, 0), 0, threshold=400)

        previous_points = np.vstack((X_prev, Y_prev))
        current_points = np.vstack((X_cur, Y_cur))

        R, t = icp_matching(previous_points, current_points)
        return R, t

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass


    def shortest_path(self, occupancy_map, start_point, end_point, threshold = 0):
        m = ds.Map(len(occupancy_map), len(occupancy_map[0]))
        m.set_obstacle([(i, j) for i in range(len(occupancy_map))\
                       for j in range(len(occupancy_map[0])) if occupancy_map[i][j] > threshold])
        start = m.map[start_point[0]][start_point[1]]
        end = m.map[end_point[0]][end_point[1]]
        dstar = ds.Dstar(m)
        print("dstar ok before", occupancy_map, start_point, end_point, threshold)
        rx, ry = dstar.run(start, end)
        print("dstar ok after", rx, ry)
        return rx, ry
    
    def yoru_ni_kakeru(self): # Racing into the night
        cur_position = self.measured_fake_position()
        cur_grid = self.pos_to_grid(cur_position)
        # BFS
        non_discovered_points = []
        dx = [-1, 0, 1, 0]
        dy = [0, -1, 0, 1]
        q = deque([cur_grid]) 
        visited = np.zeros_like(self.occupancy_map)
        visited[cur_grid[0], cur_grid[1]] = 1
        # print(cur_position, cur_grid, q, self.occupancy_map)
        while len(q):
            cur_point = q.popleft()
            for k in range(4):
                nxt_point = cur_point[0] + dx[k], cur_point[1] + dy[k]
                if not (0 <= nxt_point[0] < len(self.occupancy_map) and 0 <= nxt_point[1] < len(self.occupancy_map[0])):
                    continue
                # print("continue 1")
                if visited[nxt_point[0], nxt_point[1]]:
                    continue
                # print("continue 2")
                if self.occupancy_map[nxt_point[0], nxt_point[1]] == NON_DISCOVERED:
                    non_discovered_points.append(nxt_point)
                    continue
                # print("continue 3")
                if self.occupancy_map[nxt_point[0], nxt_point[1]] >= 1:
                    continue
                # print("continue 4")
                visited[nxt_point[0], nxt_point[1]] = 1
                q.append(nxt_point)

        non_discovered_points = np.array(non_discovered_points)
        print(f'Non discovered points: {non_discovered_points}')
        if len(non_discovered_points) <= 1:
            return
        print(f'Average of non discovered points: {np.average(non_discovered_points, axis = 0)}')
        average_point = tuple(np.average(non_discovered_points, axis = 0).astype(int))
        if cur_grid == average_point: # without this doesnt work for some reasons
            return
        print(f'Start point: {cur_grid}', f'Dist point: {average_point}')
        # thick_occupancy_map = np.copy(self.occupancy_map)
        # thick_occupancy_map = scipy.ndimage.uniform_filter(self.occupancy_map, size=10, mode='constant')
        # threshold = 10
        # thick_occupancy_map[thick_occupancy_map < threshold] = 0
        print(f'Start point type: {self.occupancy_map[cur_grid[0], cur_grid[1]]}')
        print(f'Dist point type: {self.occupancy_map[average_point[0], average_point[1]]}')
        if self.occupancy_map[cur_grid[0], cur_grid[1]] >= 1:
            for dx in range(-3, 3):
                for dy in range(-3, 3):
                    if self.occupancy_map[cur_grid[0] + dx, cur_grid[1] + dy] != -1:
                        cur_grid = cur_grid[0] + dx, cur_grid[1] + dy
                        break
                else:
                    break
        if self.occupancy_map[average_point[0], average_point[1]] >= 1:
            for dx in range(-3, 3):
                for dy in range(-3, 3):
                    if self.occupancy_map[average_point[0] + dx, average_point[1] + dy] < 1:
                        average_point = average_point[0] + dx, average_point[1] + dy
                        break
                else:
                    break
        rx, ry = self.shortest_path(self.occupancy_map, cur_grid, average_point)
        # if self.step_count % 100 == 0:
        #     plt.plot(rx, ry, "-r")
        #     plt.draw()
        #     print(rx, ry)

        #Changeable
        dest_x = 0
        dest_y = 0
        weighted_denominator = 0
        for k in range(len(rx)):
            dest_x += rx[k] * 1/(k+1)
            dest_y += ry[k] * 1/(k+1)
            weighted_denominator += 1/(k+1)
        dest_x /= weighted_denominator
        dest_y /= weighted_denominator
        print('cur point: ', cur_grid[0], cur_grid[1])
        print('dest point:', dest_x, dest_y)

        dest_angle = np.arctan2(dest_y - cur_grid[1], dest_x - cur_grid[0])
        print(f'dest angle {dest_angle}')
        return dest_angle
        
    def get_center_location(self):
        if self.center_location is not None:
            return self.center_location
        
        semantic = self.semantic().get_sensor_values()
        if semantic is None: return None
        for data in semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                center_angle = data.angle
                center_distance = data.distance
                cur_position = self.measured_fake_position()
                cur_angle = self.measured_fake_angle()
                self.center_location = cur_position[0] + math.cos(center_angle - cur_angle) * center_distance,\
                        cur_position[1] + math.sin(center_angle - cur_angle) * center_distance
                # print(self.center_location)
                return self.center_location
        return None

    def go_back_to_center(self):
        cur_position = self.measured_fake_position()
        cur_angle = self.measured_fake_angle()
        thick_occupancy_map = scipy.ndimage.uniform_filter(self.occupancy_map, size=10, mode='constant')
        threshold = 10
        thick_occupancy_map[thick_occupancy_map < threshold] = 0
        rx, ry = self.shortest_path(thick_occupancy_map, self.pos_to_grid(cur_position)
                                    , self.pos_to_grid(self.center_location), 5)
        plt.plot(rx, ry, "-r")
        plt.draw()
        print(rx, ry)


    def control(self):
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 1.0,
                   "grasper": 0}
        
        self.fix_fake_position()
        self.get_center_location()
        if self.step_count % 100 == 0 and self.center_location is not None:
            center_grid = self.pos_to_grid(self.center_location)
            circle = plt.Circle(center_grid, 10, fc='white',ec="red")
            plt.gca().add_patch(circle)
            plt.draw()

        self.gps_mapping()
        if self.step_count % 50 == 0:
            self.get_lidar_from_occupancy(self.measured_fake_position(), self.measured_fake_angle())
        
        if self.start:
            self.goal_explore_angle = 1e9
            self.start = False
            self.init_dxy()
            return command

        lidar = np.array(self.lidar().get_sensor_values())
        lidar = lidar[:-1]

        semantic = self.semantic().get_sensor_values()

        if self.step_count % 20 == 0:
            self.goal_explore_angle = self.yoru_ni_kakeru()
            # print(self.goal_explore_angle)
        # input()
        # first, we find a command to work toward the goal
        if self.state or (self.lock and self.flag): command["grasper"] = 1

        if self.step_count % 100 == 0 and command["grasper"] == 1:
            start_point = self.pos_to_grid(self.measured_fake_position())
            end_point = self.pos_to_grid(self.center_location)
            # print("GO BACK AJHVDJHSVBDHS", start_point, end_point)
            self.go_back_to_center()
        elif self.goal_explore_angle is not None and self.goal_explore_angle != 1e9:
            print("START EXPLORING")
            print(f'Current angle: {self.measured_fake_angle()}')
            print(f'Goal angle: {self.goal_explore_angle}')
            # input()
            diff_angle = abs(self.goal_explore_angle - self.measured_fake_angle()[0])
            if diff_angle < 0.2:
                print("Eren Yeager")
                command["forward"] = 1
                command["rotation"] = 0
            else:
                command["forward"] = 0
                command["rotation"] = 1


        # if self.pending == 1:
        #     # print("flag 1")
        #     command["forward"] = 0.0
        #     diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle())
        #     command["rotation"] = 1
        #     if abs(diff_angle) < 0.2:
        #         # print("flag 1.1")
        #         self.flag = 0
        #         self.pending = 0

        # elif self.goal[1] == 0:
        #     # print("flag 2")
        #     if self.flag == 0:
        #         # print("flag 2.1")
        #         command["forward"] = 0.0
        #         diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle())
        #         command["rotation"] = -1.0 if diff_angle < 0 else 1.0
        #         if abs(diff_angle) < 0.2:
        #             # print("flag 2.1.1")
        #             self.flag = 1
        #     else:
        #         # print("flag 2.2")
        #         if self.lock == 1:
        #             # print("flag 2.2.1")
        #             self.counter += 1
        #         if self.counter >= 10:
        #             # print("flag 2.2.2")
        #             self.counter = 0
        #             self.lock = 0

        # elif self.goal[1] == 1:
        #     # print("flag 3")
        #     if self.flag == 0:
        #         # print("flag 3.1")
        #         command["forward"] = 0.0
        #         diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle() + math.pi/4)
        #         command["rotation"] = -1.0 if diff_angle < 0 else 1.0
        #         if abs(diff_angle) < 0.2:
        #             # print("flag 3.1.1")
        #             self.flag = 2
        #     else:
        #         # print("flag 3.2")
        #         self.counter += 1
        #         if self.counter >= 12:
        #             # print("flag 3.2.1")
        #             self.flag = 0
        #             self.goal[1] = 0

        # elif self.goal[1] == -1:
        #     # print("flag 4")
        #     if self.flag == 0:
        #         # print("flag 4.1")
        #         command["forward"] = 0.0
        #         diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle() - math.pi/4)
        #         command["rotation"] = -1.0 if diff_angle < 0 else 1.0
        #         if abs(diff_angle) < 0.2:
        #             # print("flag 4.1.1")
        #             self.flag = 2
        #     else:
        #         # print("flag 4.2")
        #         self.counter += 1
        #         if self.counter >= 12:
        #             # print("flag 4.2.1")
        #             self.flag = 0
        #             self.counter = 0
        #             self.goal[1] = 0

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
                # input()
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

        #update prev_angle
        self.prev_position = self.measured_fake_position()
        self.prev_angle = self.measured_fake_angle()

        # return
        # plt.pause(0.1)
        return command
    
    # def control_old(self):
    #     command = {"forward": 1.0,
    #                "lateral": 0.0,
    #                "rotation": 0.0,
    #                "grasper": 0}
        
    #     self.fix_fake_position()

    #     self.gps_mapping()
    #     if self.step_count % 10 == 0:
    #         self.get_lidar_from_occupancy(self.measured_fake_position(), self.measured_fake_angle())
        
    #     if self.start:
    #         self.start = False
    #         self.init_dxy()
    #         return command

    #     lidar = np.array(self.lidar().get_sensor_values())
    #     lidar = lidar[:-1]

    #     semantic = self.semantic().get_sensor_values()

    #     # first, we find a command to work toward the goal
    #     if self.state or (self.lock and self.flag): command["grasper"] = 1

    #     if self.pending == 1:
    #         command["forward"] = 0.0
    #         diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle())
    #         command["rotation"] = 1
    #         if abs(diff_angle) < 0.2:
    #             self.flag = 0
    #             self.pending = 0

    #     elif self.goal[1] == 0:
    #         if self.flag == 0:
    #             command["forward"] = 0.0
    #             diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle())
    #             command["rotation"] = -1.0 if diff_angle < 0 else 1.0
    #             if abs(diff_angle) < 0.2:
    #                 self.flag = 1
    #         else:
    #             if self.lock == 1: self.counter += 1
    #             if self.counter >= 10:
    #                 self.counter = 0
    #                 self.lock = 0

    #     elif self.goal[1] == 1:
    #         if self.flag == 0:
    #             command["forward"] = 0.0
    #             diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle() + math.pi/4)
    #             command["rotation"] = -1.0 if diff_angle < 0 else 1.0
    #             if abs(diff_angle) < 0.2:
    #                 self.flag = 2
    #         else:
    #             self.counter += 1
    #             if self.counter >= 12:
    #                 self.flag = 0
    #                 self.goal[1] = 0

    #     elif self.goal[1] == -1:
    #         if self.flag == 0:
    #             command["forward"] = 0.0
    #             diff_angle = normalize_angle(self.goal[0] - self.measured_fake_angle() - math.pi/4)
    #             command["rotation"] = -1.0 if diff_angle < 0 else 1.0
    #             if abs(diff_angle) < 0.2:
    #                 self.flag = 2
    #         else:
    #             self.counter += 1
    #             if self.counter >= 12:
    #                 self.flag = 0
    #                 self.counter = 0
    #                 self.goal[1] = 0

    #     # then, we update our state
    #     if self.state == 0 and self.base.grasper.grasped_entities:
    #         self.lock = 0
    #         self.state = 1

    #     elif self.state == 1 and not self.base.grasper.grasped_entities:
    #         self.lock = 0
    #         self.state = 0
        
    #     # then, call new
    #     a, b = self.new(lidar, semantic)
    #     if a:
    #         self.goal = [normalize_angle(b[0]+self.measured_fake_angle()), b[1]]
    #         self.flag = 0
    #         self.counter = 0
    #         if self.debug:
    #             print('new goal!', b)
    #             # input()
    #         self.limit = 0

    #     self.limit += 1
    #     if self.limit >= 180:
    #         print('limit reached!')
    #         a, b = self.new(lidar, semantic)
    #         self.goal = [normalize_angle(b[0]+self.measured_fake_angle()), b[1]]
    #         self.limit = 0
    #         self.state = 0
    #         self.flag = 0
    #         self.counter = 0
        
    #     # print(self.prev_position, self.measeured_fake_position())

    #     #update prev_angle
    #     self.prev_position = self.measured_fake_position()
    #     self.prev_angle = self.measured_fake_angle()

    #     # return
    #     return command