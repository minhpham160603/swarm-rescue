import math
import random
from typing import List, Type
import sys 
import os

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg.playground import Playground
from spg.utils.definitions import CollisionTypes

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.sensor_disablers import EnvironmentType, NoComZone, NoGpsZone, KillZone, \
    srdisabler_disables_device
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.misc_data import MiscData

from .walls_complete_map_1 import add_walls, add_boxes


class MyMapComplete01(MapAbstract):
    environment_series = [EnvironmentType.EASY,
                          EnvironmentType.NO_COM_ZONE,
                          EnvironmentType.NO_GPS_ZONE,
                          EnvironmentType.KILL_ZONE]

    def __init__(self, environment_type: EnvironmentType = EnvironmentType.EASY):
        super().__init__(environment_type)
        self._time_step_limit = 1200
        self._real_time_limit = 240  # In seconds

        # PARAMETERS MAP
        self._size_area = (1110, 750)

        self._rescue_center = RescueCenter(size=(90, 170))
        self._rescue_center_pos = ((-505, -285), 0)

        self._no_com_zone = NoComZone(size=(270, 500))
        self._no_com_zone_pos = ((-220, 46), 0)

        self._no_gps_zone = NoGpsZone(size=(380, 252))
        self._no_gps_zone_pos = ((-360, 21), 0)

        self._kill_zone = KillZone(size=(55, 55))
        self._kill_zone_pos = ((-387, 75), 0)

        self._wounded_persons_pos = [(-516, 335), (-466, 335), (-226, 335),
                                     (-481, 75), (-61, 325), (-311, 100),
                                     (-171, -145), (-100, -155), (524, 325)]
        self._number_wounded_persons = len(self._wounded_persons_pos)
        self._wounded_persons: List[WoundedPerson] = []

        # POSITIONS OF THE DRONES
        self._number_drones = 1
        # They are positioned in a square whose side size depends on the total number of drones.
        start_area_drones = (-375, -300)
        nb_per_side = math.ceil(math.sqrt(float(self._number_drones)))
        dist_inter_drone = 30.0
        # print("nb_per_side", nb_per_side)
        # print("dist_inter_drone", dist_inter_drone)
        sx = start_area_drones[0] - (nb_per_side - 1) * 0.5 * dist_inter_drone
        sy = start_area_drones[1] - (nb_per_side - 1) * 0.5 * dist_inter_drone
        # print("sx", sx, "sy", sy)

        self._drones_pos = []
        for i in range(self._number_drones):
            x = sx + (float(i) % nb_per_side) * dist_inter_drone
            y = sy + math.floor(float(i) / nb_per_side) * dist_inter_drone
            angle = random.uniform(-math.pi, math.pi)
            self._drones_pos.append(((x, y), angle))

        self._drones: List[DroneAbstract] = []

    def construct_playground(self, drone_type: Type[DroneAbstract]) -> Playground:
        playground = ClosedPlayground(size=self._size_area)

        # RESCUE CENTER
        playground.add_interaction(CollisionTypes.GEM,
                                   CollisionTypes.ACTIVABLE_BY_GEM,
                                   wounded_rescue_center_collision)

        playground.add(self._rescue_center, self._rescue_center_pos)

        add_walls(playground)
        add_boxes(playground)

        self._explored_map.initialize_walls(playground)

        # DISABLER ZONES
        playground.add_interaction(CollisionTypes.DISABLER,
                                   CollisionTypes.DEVICE,
                                   srdisabler_disables_device)

        if self._environment_type == EnvironmentType.NO_COM_ZONE:
            playground.add(self._no_com_zone, self._no_com_zone_pos)

        if self._environment_type == EnvironmentType.NO_GPS_ZONE:
            playground.add(self._no_gps_zone, self._no_gps_zone_pos)

        if self._environment_type == EnvironmentType.KILL_ZONE:
            playground.add(self._kill_zone, self._kill_zone_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = (self._wounded_persons_pos[i], 0)
            playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground
