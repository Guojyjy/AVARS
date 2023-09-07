"""Environment for training TLC in a sumo net template."""

import numpy as np
from gym.spaces import Box
from gym.spaces.discrete import Discrete
from flow.envs.multiagent.base import MultiEnv

import math

ADDITIONAL_ENV_PARAMS_UAV = {
    "controlled_intersections": [],
}


class UAVEnvAVARS(MultiEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_UAV.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.controlled_tl = env_params.additional_params.get("controlled_intersections")
        # network, edge
        self.mapping_inc, self.num_in_edges_max, self.mapping_out, self.num_out_edges_max = \
            network.node_mapping_choose(self.controlled_tl)
        self.lanes_related = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
        for each in self.mapping_out.values():
            self.lanes_related.extend(each)
        self.lanes_related = list(set(self.lanes_related))
        # vehicle
        self.num_traffic_lights = len(self.mapping_inc.keys())
        self.state_tl = network.get_states_choose(self.controlled_tl)
        # obs
        self.observation_info = {}
        # used during visualization
        self.observed_ids = []

    @property
    def action_space(self):
        return Discrete(2)

    @property
    def observation_space(self):
        return Box(low=0., high=1, shape=((self.num_in_edges_max + self.num_out_edges_max)*2 + 1,))

    def full_name_edge_lane(self, veh_id):
        edge_id = self.k.vehicle.get_edge(veh_id)
        lane_id = self.k.vehicle.get_lane(veh_id)
        return edge_id + '_' + str(lane_id)

    def get_state(self, **kwargs):
        obs = {}

        max_speed = self.k.network.max_speed()

        if self.time_counter < 2700:
            veh_lane_pair = {each: [] for each in self.lanes_related}
            for each_veh in self.k.vehicle.get_ids():
                if self.full_name_edge_lane(each_veh) in self.lanes_related:
                    veh_lane_pair[self.full_name_edge_lane(each_veh)].append(each_veh)
            veh_num_per_edge = {}  # key: name of each edge in the road network
            avg_speed_per_edge = {}
            for lane, veh_list in veh_lane_pair.items():
                w_nor = math.ceil(self.k.network.edge_length(lane.split('_')[0]) / 7.5)
                veh_num_per_edge.update({lane: len(veh_list) / w_nor})

                speed_list = []
                for each_veh in veh_list:
                    speed_this_veh = self.k.vehicle.get_speed(each_veh)
                    speed_list.append(speed_this_veh)
                if len(veh_list) != 0:
                    avg_speed_per_edge.update({lane: np.mean(speed_list) / max_speed})
                else:
                    avg_speed_per_edge.update({lane: 0})

            # Traffic light information
            for tl_id in self.controlled_tl:
                local_edges = self.mapping_inc[tl_id]
                local_edges_out = self.mapping_out[tl_id]

                veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
                veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]
                avg_speed_per_in = [avg_speed_per_edge[each] for each in local_edges]
                avg_speed_per_out = [avg_speed_per_edge[each] for each in local_edges_out]

                # not 4-leg intersection
                if len(local_edges) < self.num_in_edges_max:
                    diff = self.num_in_edges_max - len(local_edges)
                    veh_num_per_in.extend([0] * diff)
                    avg_speed_per_in.extend([0] * diff)
                if len(local_edges_out) < self.num_out_edges_max:
                    diff = self.num_out_edges_max - len(local_edges_out)
                    veh_num_per_out.extend([0] * diff)
                    avg_speed_per_out.extend([0] * diff)

                states = self.state_tl[tl_id]
                now_state = self.k.traffic_light.get_state(tl_id)
                state_index = states.index(now_state)

                con = [round(i, 8) for i in np.concatenate(
                    [veh_num_per_in, veh_num_per_out, avg_speed_per_in, avg_speed_per_out, [state_index / len(states)]])]

                observation = np.array(con)
                obs.update({tl_id: observation})

        self.observation_info = obs
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        if self.time_counter < 2700:
            for rl_id in self.controlled_tl:
                obs = self.observation_info[rl_id]
                reward[rl_id] = -max(obs[0:self.num_in_edges_max])

        return reward

    def reset(self, **kwargs):
        self.observation_info = {}
        return super().reset()

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            action = rl_action > 0.0

            states = self.state_tl[rl_id]
            now_state = self.k.traffic_light.get_state(rl_id)
            state_index = states.index(now_state)
            if action and 'G' in now_state and self.time_counter < 2700:
                # 10min:1500; 20min:2100; 30min:2700; 40min:3300
                self.k.traffic_light.set_state_specific(node_id=rl_id, index=state_index + 1)


class UAVEnvIntelliLight(MultiEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_UAV.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.controlled_tl = env_params.additional_params.get("controlled_intersections")
        # network, edge
        self.mapping_inc, self.num_in_edges_max, self.mapping_out, self.num_out_edges_max = \
            network.node_mapping_choose(self.controlled_tl)
        self.lanes_related = []
        self.incoming_lanes = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
            self.incoming_lanes.extend(each)
        for each in self.mapping_out.values():
            self.lanes_related.extend(each)
        self.lanes_related = list(set(self.lanes_related))

        self.state_tl = network.get_states_choose(self.controlled_tl)

        edge_length = []
        edge_length.extend([self.k.network.edge_length(edge) for edge in self.k.network.get_edge_list()])
        self.max_length = max(edge_length)

        self.max_number_vehicles_lane = int(np.ceil(230 / 7.5))  # max road length / (veh length + min gap)
        # updated value per timestep
        self.custom_timestep = 0

        # related to vehicle speed
        self.waiting_veh_lane = {each: 0 for each in self.incoming_lanes}
        self.delay_lane = {each: 0 for each in self.incoming_lanes}
        # vehs list for the lanes around the selected intersections in the network
        self.veh_lane_pair = {each: [] for each in self.lanes_related}
        self.veh_passing_lane = {each: [] for each in self.incoming_lanes}
        # travel time that vehicles spent on approaching lanes, 0 if passing the intersection
        self.travel_time_veh = {}

        self.sum_waiting_time = {each: 0 for each in self.controlled_tl}

    @property
    def action_space(self):
        return Discrete(2)

    @property
    def observation_space(self):
        return Box(low=0., high=np.inf,
                   shape=((self.num_in_edges_max + self.num_out_edges_max) * (3 + self.max_number_vehicles_lane) + 2,))

    def full_name_edge_lane(self, veh_id):
        edge_id = self.k.vehicle.get_edge(veh_id)
        lane_id = self.k.vehicle.get_lane(veh_id)
        return edge_id + '_' + str(lane_id)

    def get_state(self, **kwargs):
        """
        For each lane at this intersection
        - queue length -> last veh position
        - number of vehicles
        - updated waiting time of vehicles

        - vehicle position
        - current phase and next phase
        """
        self.waiting_veh_lane = {each: 0 for each in self.incoming_lanes}
        self.delay_lane = {each: 0 for each in self.incoming_lanes}
        self.sum_waiting_time = {each: 0 for each in self.controlled_tl}
        self.veh_passing_lane = {each: [] for each in self.lanes_related}

        waiting_time_veh = {}

        obs = {}
        if self.time_counter <= 2700:
            pre_veh_lane_pair = self.veh_lane_pair
            self.veh_lane_pair = {each: [] for each in self.lanes_related}
            for each_veh in self.k.vehicle.get_ids():
                if self.full_name_edge_lane(each_veh) in self.lanes_related:
                    self.veh_lane_pair[self.full_name_edge_lane(each_veh)].append(each_veh)

            for each_lane in self.incoming_lanes:
                for veh in pre_veh_lane_pair[each_lane]:
                    if veh not in self.veh_lane_pair[each_lane]:
                        self.veh_passing_lane[each_lane].append(veh)

            veh_num_per_edge = {}  # key: name of each edge in the road network
            queue_per_lane = {}
            position_per_lane = {}
            for lane, veh_list in self.veh_lane_pair.items():
                speed_list = []
                position_list = []
                for each_veh in veh_list:
                    if lane in self.incoming_lanes:
                        if each_veh not in self.travel_time_veh.keys():
                            self.travel_time_veh.update({each_veh: 0})
                        else:
                            if each_veh not in self.veh_passing_lane[lane]:
                                self.travel_time_veh[each_veh] += 1
                    speed_this_veh = self.k.vehicle.get_speed(each_veh)
                    speed_list.append(speed_this_veh)
                    position_list.append(self.k.vehicle.get_position(each_veh))
                    if speed_this_veh < 0.1:
                        if lane in self.incoming_lanes:
                            self.waiting_veh_lane[lane] += 1
                        if each_veh in waiting_time_veh.keys():
                            waiting_time_veh[each_veh] += 1
                        else:
                            waiting_time_veh.update({each_veh: 1})
                    else:
                        waiting_time_veh[each_veh] = 0
                if lane in self.incoming_lanes:
                    self.delay_lane[lane] = np.mean(speed_list) / self.k.network.speed_limit(lane.split('_')[0]) \
                        if speed_list else 0
                if not position_list:
                    queue_per_lane.update({lane: 0})
                else:
                    queue_per_lane.update({lane: self.k.network.edge_length(lane.split('_')[0]) - np.min(position_list)})
                position_per_lane.update({lane: position_list})
                veh_num_per_edge.update({lane: len(veh_list)})

            # Traffic light information
            for tl_id in self.controlled_tl:
                local_edges = self.mapping_inc[tl_id]
                local_edges_out = self.mapping_out[tl_id]

                queue_per_lane_in = [queue_per_lane[each] for each in local_edges]
                queue_per_lane_out = [queue_per_lane[each] for each in local_edges_out]
                veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
                veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]

                position_inter = []

                waiting_time_lane_in = []
                for each_lane in local_edges:
                    sum_waiting = 0
                    for each_veh in self.veh_lane_pair[each_lane]:
                        sum_waiting += waiting_time_veh[each_veh]
                    position_inter.extend(position_per_lane[each_lane])
                    position_inter.extend([0] * (self.max_number_vehicles_lane - len(position_per_lane[each_lane])))
                    waiting_time_lane_in.append(sum_waiting)
                self.sum_waiting_time[tl_id] = np.sum(waiting_time_lane_in)

                # not 4-leg intersection
                if len(local_edges) < self.num_in_edges_max:
                    diff = self.num_in_edges_max - len(local_edges)
                    queue_per_lane_in.extend([0] * diff)
                    veh_num_per_in.extend([0] * diff)
                    waiting_time_lane_in.extend([0] * diff)
                    position_inter.extend([0] * diff * self.max_number_vehicles_lane)

                waiting_time_lane_out = []
                for each_lane in local_edges_out:
                    sum_waiting = 0
                    for each_veh in self.veh_lane_pair[each_lane]:
                        sum_waiting += waiting_time_veh[each_veh]
                    position_inter.extend(position_per_lane[each_lane])
                    position_inter.extend(
                            [0] * (self.max_number_vehicles_lane - len(position_per_lane[each_lane])))
                    waiting_time_lane_out.append(sum_waiting)

                if len(local_edges_out) < self.num_out_edges_max:
                    diff = self.num_out_edges_max - len(local_edges_out)
                    queue_per_lane_out.extend([0] * diff)
                    veh_num_per_out.extend([0] * diff)
                    waiting_time_lane_out.extend([0] * diff)
                    position_inter.extend([0] * diff * self.max_number_vehicles_lane)

                states = self.state_tl[tl_id]
                now_state = self.k.traffic_light.get_state(tl_id)
                state_index = states.index(now_state)
                next_state = state_index + 1
                if next_state > 3:
                    next_state = 0

                con = [round(i, 8) for i in np.concatenate(
                    [queue_per_lane_in, queue_per_lane_out, veh_num_per_in, veh_num_per_out, waiting_time_lane_in,
                     waiting_time_lane_out, position_inter, [state_index, next_state]])] #
                observation = np.array(con)
                obs.update({tl_id: observation})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        if self.time_counter <= 2700:
            for rl_id in self.controlled_tl:
                local_edges = self.mapping_inc[rl_id]

                sum_waiting_vehs = 0
                sum_delay = 0
                sum_passing_vehs = 0
                sum_travel_time = 0
                for each_lane in local_edges:
                    sum_waiting_vehs += self.waiting_veh_lane[each_lane]
                    sum_delay += (1 - self.delay_lane[each_lane])
                    sum_passing_vehs += len(self.veh_passing_lane[each_lane])
                    for veh in self.veh_passing_lane[each_lane]:
                        sum_travel_time += self.travel_time_veh[veh]
                        # del self.travel_time_veh[veh]
                        self.travel_time_veh[veh] = 0

                reward[rl_id] = -0.25 * sum_waiting_vehs - 0.25 * sum_delay - 0.25 * self.sum_waiting_time[rl_id] \
                                - 5 * rl_actions[rl_id] + sum_passing_vehs + sum_travel_time / 60

        return reward

    def reset(self, **kwargs):
        self.custom_timestep = 0
        self.waiting_veh_lane = {each: 0 for each in self.incoming_lanes}
        self.delay_lane = {each: 0 for each in self.incoming_lanes}
        self.veh_lane_pair = {each: [] for each in self.lanes_related}
        self.veh_passing_lane = {each: [] for each in self.incoming_lanes}
        self.travel_time_veh = {}
        self.sum_waiting_time = {each: 0 for each in self.controlled_tl}
        return super().reset()

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            action = rl_action > 0.0

            states = self.state_tl[rl_id]
            now_state = self.k.traffic_light.get_state(rl_id)
            state_index = states.index(now_state)

            if action and 'G' in now_state and self.time_counter <= 2700:
                # 10min:1500; 20min:2100; 30min:2700; 40min:3300
                self.k.traffic_light.set_state_specific(node_id=rl_id, index=state_index + 1)
