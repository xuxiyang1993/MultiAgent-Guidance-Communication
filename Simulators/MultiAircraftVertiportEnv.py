import math
import time

import numpy as np
import random
import gym
from gym import spaces
# from gym.utils import seeding
from collections import OrderedDict

from nodes_multi import MultiAircraftNode, MultiAircraftState
from search_multi import MCTS
from config_vertiport import Config

__author__ = "Xuxi Yang <xuxiyang@iastate.edu>"

import ipdb


class MultiAircraftEnv(gym.Env):
    """
    This is the airspace simulator where we can control multiple aircraft to their respective
    goal position while avoiding conflicts between each other. The aircraft will takeoff from
    different vertiports, and select a random vertiport as its destination.
    **STATE:**
    The state consists all the information needed for the aircraft to choose an optimal action:
    position, velocity, speed, heading, goal position, of each aircraft:
    (x, y, v_x, v_y, speed, \psi, goal_x, goal_y) for each aircraft.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 for the change of heading angle of each aircraft.
    More specifically, the action is a dictionary in form {id: action, id: action, ...}
    """

    def __init__(self, sd=2, debug=False, decentralized=False):
        self.load_config()  # load parameters for the simulator
        self.load_vertiport()  # load config for the vertiports
        self.state = None
        self.viewer = None

        # build observation space and action space
        self.observation_space = self.build_observation_space()  # observation space deprecated, not in use for MCTS
        self.position_range = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height]),
            dtype=np.float32)  # position range is the length and width of airspace
        self.action_space = spaces.Tuple((spaces.Discrete(3),) * self.num_aircraft)

        self.total_timesteps = 0

        self.conflicts = 0
        self.seed(sd)

        self.centralized_controller = Controller(self)

        self.decentralized = decentralized

        self.debug = debug

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        # self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_config(self):
        # input dim
        self.window_width = Config.window_width
        self.window_height = Config.window_height  # dimension of the airspace
        self.num_aircraft = Config.num_aircraft
        self.EPISODES = Config.EPISODES
        # self.tick = Config.tick
        self.scale = Config.scale  # 1 meter = ? pixels, set to 60 here
        self.minimum_separation = Config.minimum_separation
        self.NMAC_dist = Config.NMAC_dist
        # self.horizon_dist = Config.horizon_dist
        self.initial_min_dist = Config.initial_min_dist  # when aircraft generated, is shouldn't be too close to others
        self.goal_radius = Config.goal_radius
        self.init_speed = Config.init_speed
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed

    def load_vertiport(self):
        self.vertiport_list = []
        # read the vertiport location from config file
        for i in range(Config.vertiport_loc.shape[0]):
            self.vertiport_list.append(VertiPort(id=i, position=Config.vertiport_loc[i]))

    def reset(self):
        # aircraft is stored in this dict
        self.aircraft_dict = AircraftDict()
        self.id_tracker = 0  # assign id to newly generated aircraft, increase by one after generating aircraft.

        # keep track of number of conflicts, goals, and NMACs.
        self.conflicts = 0
        self.goals = 0
        self.NMACs = 0

        return self._get_ob()

    # deprecated
    def pressure_reset(self):
        self.conflicts = 0
        # aircraft is stored in this list
        self.aircraft_list = []

        for id in range(self.num_aircraft):
            theta = 2 * id * math.pi / self.num_aircraft
            r = self.window_width / 2 - 10
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            position = (self.window_width / 2 + x, self.window_height / 2 + y)
            goal_pos = (self.window_width / 2 - x, self.window_height / 2 - y)

            aircraft = Aircraft(
                id=id,
                position=position,
                speed=self.init_speed,
                heading=theta + math.pi,
                goal_pos=goal_pos
            )

            self.aircraft_list.append(aircraft)

        return self._get_ob()

    def _get_ob(self):
        s = []
        id = []
        # loop all the aircraft
        # return the information of each aircraft and their respective id
        # s is in shape [number_aircraft, 8], id is list of length number_aircraft
        for key, aircraft in self.aircraft_dict.ac_dict.items():
            # (x, y, vx, vy, speed, heading, gx, gy)
            s.append(aircraft.position[0])
            s.append(aircraft.position[1])
            s.append(aircraft.velocity[0])
            s.append(aircraft.velocity[1])
            s.append(aircraft.speed)
            s.append(aircraft.heading)
            s.append(aircraft.goal.position[0])
            s.append(aircraft.goal.position[1])

            id.append(key)

        return np.reshape(s, (-1, 8)), id

    def _get_normalized_ob(self):
        # state contains pos, vel for all intruder aircraft
        # pos, vel, speed, heading for ownship
        # goal pos
        def normalize_velocity(velocity):
            translation = velocity + self.max_speed
            return translation / (self.max_speed * 2)

        s = []
        id = []
        # loop all the aircraft
        # return the information of each aircraft and their respective id
        # s is in shape [number_aircraft, 8], id is list of length number_aircraft
        for key, aircraft in self.aircraft_dict.ac_dict.items():
            # (x, y, vx, vy, speed, heading, gx, gy)
            s.append(aircraft.position[0] / Config.window_width)
            s.append(aircraft.position[1] / Config.window_height)
            s.append(normalize_velocity(aircraft.velocity[0]))
            s.append(normalize_velocity(aircraft.velocity[1]))
            s.append((aircraft.speed - Config.min_speed) / (Config.max_speed - Config.min_speed))
            s.append(aircraft.heading / (2 * math.pi))
            s.append(aircraft.goal.position[0] / Config.window_width)
            s.append(aircraft.goal.position[1] / Config.window_height)

            id.append(key)

        return np.reshape(s, (-1, 8)), id

    def step(self, a, near_end=False):
        # a is a dictionary: {id: action, id: action, ...}
        # since MCTS is used every 5 seconds, there may be new aircraft generated during the 5 time step interval, which
        # MCTS algorithm doesn't generate an action for it. In this case we let it fly straight.
        assigned_actions = a
        for id, aircraft in self.aircraft_dict.ac_dict.items():
            try:
                aircraft.step(a[id])
            except KeyError:
                aircraft.step()
                assigned_actions[id] = 1

        # record actions in the controller for predictions under communication loss
        self.centralized_controller.collect_actions(assigned_actions)

        for vertiport in self.vertiport_list:
            vertiport.step()  # increase the clock of vertiport by 1
            # generate new aircraft if the clock pass the interval
            if vertiport.clock_counter >= vertiport.time_next_aircraft and not near_end:
                goal_vertiport_id = random.choice([e for e in range(len(self.vertiport_list)) if not e == vertiport.id])
                # generate new aircraft and prepare to add it the dict
                aircraft = Aircraft(
                    id=self.id_tracker,
                    position=vertiport.position,
                    speed=self.init_speed,
                    heading=self.random_heading(),
                    goal_pos=self.vertiport_list[goal_vertiport_id].position
                )
                # calc its dist to all the other aircraft
                dist_array, id_array = self.dist_to_all_aircraft(aircraft)
                min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
                # add it to dict only if it's far from others
                if min_dist > 5 * self.minimum_separation:  # and self.aircraft_dict.num_aircraft < 10:
                    self.aircraft_dict.add(aircraft)
                    self.id_tracker += 1  # increase id_tracker

                    vertiport.generate_interval()  # reset clock for this vertiport and generate a new time interval

        # return the reward, done, and info
        reward, terminal, info = self._terminal_reward()

        self.total_timesteps += self.aircraft_dict.num_aircraft

        return self.centralized_controller.get_ob(), reward, terminal, info
        return self._get_ob(), reward, terminal, info

    def _terminal_reward(self):
        """
        determine the reward and terminal for the current transition, and use info. Main idea:
        1. for each aircraft:
          a. if there a conflict, return a penalty for it
          b. if there is NMAC, assign a penalty to it and prepare to remove this aircraft from dict
          b. elif it is out of map, assign its reward as Config.wall_penalty, prepare to remove it
          c. elif if it reaches goal, assign its reward to Config.goal_reward, prepare to remove it
          d. else assign its reward as Config.step_penalty.
        3. remove out-of-map aircraft and goal-aircraft

        """
        reward = 0
        # info = {'n': [], 'c': [], 'w': [], 'g': []}
        info_dist_list = []
        aircraft_to_remove = []  # add goal-aircraft and out-of-map aircraft to this list

        for id, aircraft in self.aircraft_dict.ac_dict.items():
            # calculate min_dist and dist_goal for checking terminal
            dist_array, id_array = self.dist_to_all_aircraft(aircraft)
            min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
            info_dist_list.append(min_dist)
            aircraft.min_dist = min_dist

            conflict = False
            # set the conflict flag to false for aircraft
            # elif conflict, set penalty reward and conflict flag but do NOT remove the aircraft from list
            for id2, dist in zip(id_array, dist_array):
                if dist >= self.minimum_separation:  # safe
                    aircraft.conflict_id_set.discard(id2)  # discarding element not in the set won't raise error

                else:  # conflict!!
                    if self.debug:
                        self.render()
                        import ipdb
                        ipdb.set_trace()
                    conflict = True
                    if id2 not in aircraft.conflict_id_set:  # and dist < self.minimum_separation:  # use original min separation
                        self.conflicts += 1
                        aircraft.conflict_id_set.add(id2)
                        # info['c'].append('%d and %d' % (aircraft.id, id))
                    aircraft.reward = Config.conflict_penalty

            # if NMAC, set penalty reward and prepare to remove the aircraft from list
            if min_dist < self.NMAC_dist:
                if self.debug:
                    self.render()
                    import ipdb
                    ipdb.set_trace()
                # info['n'].append('%d and %d' % (aircraft.id, close_id))
                aircraft.reward = Config.NMAC_penalty
                aircraft_to_remove.append(aircraft)
                self.NMACs += 1
                # aircraft_to_remove.append(self.aircraft_dict.get_aircraft_by_id(close_id))

            # give out-of-map aircraft a penalty, and prepare to remove it
            elif not self.position_range.contains(np.array(aircraft.position)):
                aircraft.reward = Config.wall_penalty
                # info['w'].append(aircraft.id)
                if aircraft not in aircraft_to_remove:
                    aircraft_to_remove.append(aircraft)

            # set goal-aircraft reward according to simulator, prepare to remove it
            elif aircraft.distance_goal < self.goal_radius:
                aircraft.reward = Config.goal_reward
                # info['g'].append(aircraft.id)
                self.goals += 1
                if aircraft not in aircraft_to_remove:
                    aircraft_to_remove.append(aircraft)

            # for aircraft without NMAC, conflict, out-of-map, goal, set its reward as simulator
            elif not conflict:
                aircraft.reward = Config.step_penalty

            # accumulates reward
            reward += aircraft.reward

        # remove all the out-of-map aircraft and goal-aircraft
        removed_id = []
        for aircraft in aircraft_to_remove:
            self.aircraft_dict.remove(aircraft)
            removed_id.append(aircraft.id)
        # reward = [e.reward for e in self.aircraft_dict]

        # report removed aircraft to controller
        self.centralized_controller.collect_removed(removed_id)

        # info_dist_list is the min_dist to other aircraft for each aircraft.
        return reward, False, info_dist_list

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        from colour import Color
        red = Color('red')
        colors = list(red.range_to(Color('green'), self.num_aircraft))

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_width, self.window_height)
            self.viewer.set_bounds(0, self.window_width, 0, self.window_height)

        import os
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # draw all the aircraft
        for id, aircraft in self.aircraft_dict.ac_dict.items():
            aircraft_img = rendering.Image(os.path.join(__location__, 'images/aircraft.png'), 32, 32)
            jtransform = rendering.Transform(rotation=aircraft.heading - math.pi / 2, translation=aircraft.position)
            aircraft_img.add_attr(jtransform)
            r, g, b = colors[aircraft.id % self.num_aircraft].get_rgb()
            aircraft_img.set_color(r, g, b)
            self.viewer.onetime_geoms.append(aircraft_img)

            goal_img = rendering.Image(os.path.join(__location__, 'images/goal.png'), 32, 32)
            jtransform = rendering.Transform(rotation=0, translation=aircraft.goal.position)
            goal_img.add_attr(jtransform)
            goal_img.set_color(r, g, b)
            self.viewer.onetime_geoms.append(goal_img)

            circle_img = rendering.make_circle(radius=aircraft.minimum_separation / 2, res=30, filled=False)
            jtransform = rendering.Transform(rotation=0, translation=aircraft.position)
            circle_img.add_attr(jtransform)
            self.viewer.onetime_geoms.append(circle_img)

            if aircraft.communication_loss and self.decentralized:
                point = self.viewer.draw_polygon(Config.point)
                pos = aircraft.position
                jtransform = rendering.Transform(rotation=0, translation=pos)
                point.add_attr(jtransform)
                self.viewer.onetime_geoms.append(point)

        # draw all the vertiports
        for veriport in self.vertiport_list:
            vertiport_img = rendering.Image(os.path.join(__location__, 'images/verti.png'), 32, 32)
            jtransform = rendering.Transform(rotation=0, translation=veriport.position)
            vertiport_img.add_attr(jtransform)
            self.viewer.onetime_geoms.append(vertiport_img)

        for aircraft_id in self.centralized_controller.missing_aircraft:
            point = self.viewer.draw_polygon(Config.point)
            pos = self.centralized_controller.information_center[aircraft_id][:2]
            jtransform = rendering.Transform(rotation=0, translation=pos)
            point.add_attr(jtransform)
            self.viewer.onetime_geoms.append(point)

        return self.viewer.render(return_rgb_array=False)

    def draw_point(self, point):
        # for debug
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_width, self.window_height)
            self.viewer.set_bounds(0, self.window_width, 0, self.window_height)

        img = self.viewer.draw_polygon(Config.point)
        jtransform = rendering.Transform(rotation=0, translation=point)
        img.add_attr(jtransform)
        self.viewer.onetime_geoms.append(img)

        import os
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        for veriport in self.vertiport_list:
            vertiport_img = rendering.Image(os.path.join(__location__, 'images/verti.png'), 32, 32)
            jtransform = rendering.Transform(rotation=0, translation=veriport.position)
            vertiport_img.add_attr(jtransform)
            self.viewer.onetime_geoms.append(vertiport_img)

        return self.viewer.render(return_rgb_array=False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # dist to all the aircraft
    def dist_to_all_aircraft(self, aircraft):
        id_list = []
        dist_list = []
        for id, intruder in self.aircraft_dict.ac_dict.items():
            if id != aircraft.id:
                id_list.append(id)
                dist_list.append(self.metric(aircraft.position, intruder.position))

        return np.array(dist_list), np.array(id_list)

    def dist_goal(self, aircraft):
        return self.metric(aircraft.position, aircraft.goal.position)

    @staticmethod
    def metric(pos1, pos2):
        # the distance between two points
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    # def dist(self, pos1, pos2):
    #     return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def random_pos(self):
        return np.random.uniform(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height])
        )

    def random_speed(self):
        return np.random.uniform(low=self.min_speed, high=self.max_speed)

    def random_heading(self):
        return np.random.uniform(low=0, high=2 * math.pi)

    def build_observation_space(self):
        s = spaces.Dict({
            'pos_x': spaces.Box(low=0, high=self.window_width, shape=(1,), dtype=np.float32),
            'pos_y': spaces.Box(low=0, high=self.window_height, shape=(1,), dtype=np.float32),
            'vel_x': spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
            'vel_y': spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
            'speed': spaces.Box(low=self.min_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
            'heading': spaces.Box(low=0, high=2 * math.pi, shape=(1,), dtype=np.float32),
            'goal_x': spaces.Box(low=0, high=self.window_width, shape=(1,), dtype=np.float32),
            'goal_y': spaces.Box(low=0, high=self.window_height, shape=(1,), dtype=np.float32),
        })

        return spaces.Tuple((s,) * self.num_aircraft)


class AircraftDict:
    def __init__(self):
        self.ac_dict = OrderedDict()

    # how many aircraft currently en route
    @property
    def num_aircraft(self):
        return len(self.ac_dict)

    # add aircraft to dict
    def add(self, aircraft):
        # id should always be different
        assert aircraft.id not in self.ac_dict.keys(), 'aircraft id %d already in dict' % aircraft.id
        self.ac_dict[aircraft.id] = aircraft

    # remove aircraft from dict
    def remove(self, aircraft):
        try:
            del self.ac_dict[aircraft.id]
        except KeyError:
            pass

    # get aircraft by its id
    def get_aircraft_by_id(self, aircraft_id):
        return self.ac_dict[aircraft_id]


# class AircraftList:
#     def __init__(self):
#         self.ac_list = []
#         self.id_list = []
#
#     @property
#     def num_aircraft(self):
#         return len(self.ac_list)
#
#     def add(self, aircraft):
#         self.ac_list.append(aircraft)
#         self.id_list.append(aircraft.id)
#         assert len(self.ac_list) == len(self.id_list)
#
#         unique, count = np.unique(np.array(self.id_list), return_counts=True)
#         assert np.all(count < 2), 'ununique id added to list'
#
#     def remove(self, aircraft):
#         try:
#             self.ac_list.remove(aircraft)
#             self.id_list.remove(aircraft.id)
#             assert len(self.ac_list) == len(self.id_list)
#         except ValueError:
#             pass
#
#     def get_aircraft_by_id(self, aircraft_id):
#         index = np.where(np.array(self.id_list) == aircraft_id)[0]
#         assert index.shape[0] == 1, 'find multi aircraft with id %d' % aircraft_id
#         return self.ac_list[int(index)]
#
#         for aircraft in self.buffer_list:
#             if aircraft.id == aircraft_id:
#                 return aircraft


class Goal:
    def __init__(self, position):
        self.position = position

    def __repr__(self):
        s = 'pos: %s' % self.position
        return s


class Aircraft:
    def __init__(self, id, position, speed, heading, goal_pos):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.speed = speed
        self.heading = heading  # rad
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy], dtype=np.float32)

        self.reward = 0
        self.goal = Goal(goal_pos)
        dx, dy = self.goal.position - self.position
        self.heading = math.atan2(dy, dx)  # set its initial heading point to its goal

        self.load_config()

        self.conflict_id_set = set()  # store the id of all aircraft currently in conflict

        self.communication_loss = False  # self awareness of communication loss
        self.prob_lost = 0.1  # probability of communication loss
        self.steps = 0  # prevent communication loss right after take-off
        self.distance_goal = self.dist_goal()  # distance to goal

        self.lost_steps = 0  # duration of the current loss
        self.minimum_separation = Config.minimum_separation
        self.loss_happened = False

        self.information_center = {}
        self.action_center = {}
        self.action = 1
        self.visible_aircraft = OrderedDict()
        self.min_dist = np.inf
        self.min_dist_id = None
        self.state = None
        self.idx = None
        self.miss_ids = []
        self.miss_ac = []

    def load_config(self):
        self.G = Config.G
        self.scale = Config.scale
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed
        self.speed_sigma = Config.speed_sigma
        # self.position_sigma = Config.position_sigma
        self.d_heading = Config.d_heading
        self.power = Config.aircraft_power

    def step(self, a=1):
        self.speed = max(self.min_speed, min(self.speed, self.max_speed))  # project to range
        self.speed += np.random.normal(0, self.speed_sigma)
        self.heading += (a - 1) * self.d_heading + np.random.normal(0, Config.heading_sigma)
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy])

        self.position += self.velocity

        self.distance_goal = self.dist_goal()
        self.steps += 1

    def send_info_to(self, controller=None, to_aircraft=False):
        if not to_aircraft:
            dx = controller.position[0] - self.position[0]
            dy = controller.position[1] - self.position[1]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            power = dist
            self.power -= power

        # check probability of communication loss
        # has to be at least one step after take-off to have communication loss
        # for debugging, every 10 planes have one plane with loss

        # communication_loss = randdd < self.prob_lost and self.steps > 80 and not to_aircraft
        if not self.communication_loss:

            randdd = np.random.rand(1)
            if self.steps > 80 and not to_aircraft and self.dist_goal() > 5 * Config.goal_radius and randdd < self.prob_lost:
                self.steps_to_lost = np.random.uniform(2, 10)

                self.communication_loss = True
                self.loss_happened = True
                self.lost_steps += 1

                self.minimum_separation = Config.minimum_separation + 0.21 * self.lost_steps

            else:
                self.lost_steps = 0
                self.minimum_separation = Config.minimum_separation

                if not to_aircraft:
                    self.communication_loss = False

                s = []
                s.append(self.position[0])
                s.append(self.position[1])
                s.append(self.velocity[0])
                s.append(self.velocity[1])
                s.append(self.speed)
                s.append(self.heading)
                s.append(self.goal.position[0])
                s.append(self.goal.position[1])
                s.append(self.minimum_separation)
                if controller:
                    controller.information_center[self.id] = s

                if to_aircraft:
                    return s

        elif self.communication_loss:
            self.loss_happened = True
            self.lost_steps += 1

            self.minimum_separation = Config.minimum_separation + 0.21 * self.lost_steps

            if self.lost_steps >= self.steps_to_lost:
                self.communication_loss = False
                self.lost_steps = 0

        # if not communication_loss or self.dist_goal() < 5 * Config.goal_radius:
        #     self.lost_steps = 0
        #
        #     # self.minimum_separation = Config.minimum_separation if not self.loss_happened else 2 * Config.minimum_separation
        #     self.minimum_separation = Config.minimum_separation
        #
        #     if not to_aircraft:
        #         self.communication_loss = False
        #     # self.send_state_to(controller.information_center['state'])
        #     # self.send_id_to(controller.information_center['id'])
        #     s = []
        #     s.append(self.position[0])
        #     s.append(self.position[1])
        #     s.append(self.velocity[0])
        #     s.append(self.velocity[1])
        #     s.append(self.speed)
        #     s.append(self.heading)
        #     s.append(self.goal.position[0])
        #     s.append(self.goal.position[1])
        #     s.append(self.minimum_separation)
        #     if controller:
        #         controller.information_center[self.id] = s
        #
        #     if to_aircraft:
        #         return s
        #
        # else:
        #     self.communication_loss = True
        #     self.loss_happened = True
        #     self.lost_steps += 1
        #
        #     # self.minimum_separation = np.clip(np.exp(self.lost_steps / 15) + 1, 2, 3.5) * Config.minimum_separation
        #     # self.minimum_separation = np.clip(np.exp(self.lost_steps / 30), 1, 2) * Config.minimum_separation
        #     self.minimum_separation = Config.minimum_separation + 0.21 * self.lost_steps

    def send_state_to(self, lst):
        lst.append(self.position[0])
        lst.append(self.position[1])
        lst.append(self.velocity[0])
        lst.append(self.velocity[1])
        lst.append(self.speed)
        lst.append(self.heading)
        lst.append(self.goal.position[0])
        lst.append(self.goal.position[1])
        lst.append(self.minimum_separation)

    def send_id_to(self, lst):
        lst.append(self.id)

    def dist_goal(self):
        return MultiAircraftEnv.metric(self.goal.position, self.position)

    def dist_min_max(self, ac_dict):
        self.miss_ids = []
        for ac_id, aircraft in ac_dict.items():
            if ac_id != self.id:
                distance = MultiAircraftEnv.metric(self.position, aircraft.position)
                if distance < self.min_dist:
                    self.min_dist = distance
                    self.min_dist_id = ac_id
                prob_loss = np.clip(distance / (15 * Config.minimum_separation) - 1, 0, 0.5)
                rn = np.random.rand(1)
                if rn < prob_loss and len(self.miss_ids) < 3:
                    self.miss_ids.append(ac_id)

    def get_aircraft_info(self, ac_dict):
        self.information_center = {}
        self.dist_min_max(ac_dict)
        state = []
        ac_copy = ac_dict.copy()
        for lost in self.miss_ids:
            self.miss_ac.append(ac_copy.pop(lost))
        self.visible_aircraft = ac_copy

        for i, (ac_id, aircraft) in enumerate(ac_copy.items()):
            if ac_id == self.id:
                self.idx = i
            self.information_center[ac_id] = aircraft.send_info_to(None, True)
            state += self.information_center[ac_id]

        self.state = np.reshape(state, (-1, 9))
        assert self.idx is not None
        return self

    def make_decision(self):
        self.action = None
        action = []
        for ac_id, aircraft in self.visible_aircraft.items():
            try:
                action.append(self.action_center[ac_id])
            except KeyError:
                action.append(1)

        state = MultiAircraftState(state=self.state, index=self.idx, init_action=action)
        root = MultiAircraftNode(state=state)
        mcts = MCTS(root)
        # if aircraft if close to another aircraft, build a larger tree, else build smaller tree
        if self.min_dist < 3 * Config.minimum_separation:
            best_node = mcts.best_action(Config.no_simulations, Config.search_depth)
        else:
            best_node = mcts.best_action(Config.no_simulations_lite, Config.search_depth_lite)
        self.action = best_node.state.prev_action[self.idx]
        assert self.action is not None
        return self

    def broadcast_action(self, ac_dict):
        for ac_id in self.visible_aircraft.keys():
            ac_dict[ac_id].action_center[self.id] = self.action
        return ac_dict

    @staticmethod
    def move_toward_goal(aircraft):
        dx, dy = aircraft.goal.position - aircraft.position
        goal_heading = math.atan2(dy, dx)
        adjust_angle = aircraft.heading - goal_heading
        if -0.06 <= adjust_angle <= 0.06:  # within five degrees no adjustment
            action = 1
        elif 0 <= adjust_angle <= math.pi:
            action = 0
        else:
            action = 2
        return action

    # def default_step(self, aircraft_id, action):
    #     last_state = self.information_center_last[aircraft_id]
    #     p_x, p_y, v_x, v_y, speed, heading, g_x, g_y, min_seq = last_state
    #     speed = max(self.min_speed, min(speed, self.max_speed))  # project to range
    #     # speed += np.random.normal(0, self.speed_sigma)
    #     heading += (action - 1) * self.d_heading  # + np.random.normal(0, Config.heading_sigma)
    #     v_x = speed * math.cos(heading)
    #     v_y = speed * math.sin(heading)
    #     p_x += v_x
    #     p_y += v_y
    #     min_seq = Config.minimum_separation
    #     predicted_state = [p_x, p_y, v_x, v_y, speed, heading, g_x, g_y, min_seq]
    #     self.information_center[aircraft_id] = predicted_state
    #     return predicted_state

    def __repr__(self):
        s = 'id: %d, pos: %.2f,%.2f, speed: %.2f, heading: %.2f goal: %.2f,%.2f' \
            % (self.id,
               self.position[0],
               self.position[1],
               self.speed,
               math.degrees(self.heading),
               self.goal.position[0],
               self.goal.position[1],
               )
        return s


class VertiPort:
    def __init__(self, id, position):
        self.id = id
        self.position = np.array(position)  # position of vertiport
        self.clock_counter = 0
        self.time_next_aircraft = np.random.uniform(0, 60)

    # when the next aircraft will take off
    def generate_interval(self):
        # time interval to generate next aircraft
        self.time_next_aircraft = np.random.uniform(Config.time_interval_lower, Config.time_interval_upper)
        self.clock_counter = 0

    # add the clock counter by 1
    def step(self):
        self.clock_counter += 1


class Controller:
    def __init__(self, env):
        self.position = np.array([400, 400])
        self.env = env
        # self.information_center = {'state': [], 'id': []}
        # self.information_center_last = {'state': [], 'id': []}
        self.information_center = {}
        self.information_center_last = {}
        self.removed_aircraft_id = []
        self.missing_aircraft = []
        self.missing_aircraft_last = []
        self.missing_duration = {}
        self.last_actions = {}

        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed
        self.speed_sigma = Config.speed_sigma
        self.d_heading = Config.d_heading

    def get_ob(self):
        self.information_center_last = self.information_center
        self.missing_aircraft_last = self.missing_aircraft
        # self.information_center = {'state': [], 'id': []}
        self.information_center = {}
        for key, aircraft in self.env.aircraft_dict.ac_dict.items():
            aircraft.send_info_to(self, self.env.decentralized)

        self.missing_aircraft = [aircraft for aircraft in self.information_center_last.keys() if aircraft not in
                                 self.information_center.keys() and aircraft not in self.removed_aircraft_id]
        self.duration_update()
        # import ipdb; ipdb.set_trace()
        for aircraft_id in self.missing_aircraft:
            self.default_step(aircraft_id)

        state = []
        for _, s in self.information_center.items():
            state += s
        # state = np.concatenate([s for _, s in self.information_center.items()])

        return self.process_state(np.reshape(state, (-1, 9))), list(self.information_center.keys())
        return self.process_state(np.reshape(self.information_center['state'], (-1, 9))), self.information_center['id']

    def process_state(self, state):
        return state

    # get normally removed aircraft in situations such as goal, NMAC
    def collect_removed(self, removed):
        self.removed_aircraft_id = removed

    # record previously assigned actions
    def collect_actions(self, actions):
        self.last_actions = actions

    # fill missing id and state using previously assigned action
    # def default_step(self, aircraft_id):
    #     last_state_idx = self.information_center_last['id'].index(aircraft_id) * 8
    #     last_state = self.information_center_last['state'][last_state_idx: last_state_idx + 8]
    #     predicted_state = self._step(last_state, self.last_actions[aircraft_id])
    #     self.information_center['state'].extend(predicted_state)
    #     self.information_center['id'].append(aircraft_id)

    # fill missing id and state using previously assigned action
    def default_step(self, aircraft_id):
        last_state = self.information_center_last[aircraft_id]
        predicted_state = self._step(last_state, self.last_actions[aircraft_id], aircraft_id)
        self.information_center[aircraft_id] = predicted_state

    # based on aircraft class step method
    # predict current location based on previously assigned action
    def _step(self, last_state, action, aircraft_id):
        # import ipdb; ipdb.set_trace()
        p_x, p_y, v_x, v_y, speed, heading, g_x, g_y, min_seq = last_state
        speed = max(self.min_speed, min(speed, self.max_speed))  # project to range
        # speed += np.random.normal(0, self.speed_sigma)
        heading += (action - 1) * self.d_heading  # + np.random.normal(0, Config.heading_sigma)
        v_x = speed * math.cos(heading)
        v_y = speed * math.sin(heading)
        p_x += v_x
        p_y += v_y
        # min_seq = np.clip(np.exp(self.missing_duration[aircraft_id] / 30), 1, 2) * Config.minimum_separation
        min_seq = Config.minimum_separation + 0.21 * self.missing_duration[aircraft_id]
        return [p_x, p_y, v_x, v_y, speed, heading, g_x, g_y, min_seq]

    # update duration for aircrafts
    def duration_update(self):
        last = set(self.missing_aircraft_last)
        current = set(self.missing_aircraft)
        removed = last - current
        new = current - last
        continued = last & current

        for key in removed:
            self.missing_duration.pop(key)

        for key in continued:
            self.missing_duration[key] += 1

        for key in new:
            self.missing_duration[key] = 1
