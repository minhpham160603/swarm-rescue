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
        self.limit = 0
        self.pending = 0
        self.data = []
        # state: finding people or taking people back
        # flag: to help with going 

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

        if self.start:
            self.start = False
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

        # update angle1
        self.angle1 += self.odometer_values()[-1]

        # return
        return command
