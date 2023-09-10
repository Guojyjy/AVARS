import os, sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import sumolib


def get_e1(tl_id, e1_add_file):
    """
    Input: the controlled tl list
    ---
    Return e1s, dict <lane:e1.id>
    """
    # get the list of lanes which are controlled by the named tl (incoming lanes)
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    e1s = {}
    for e1 in sumolib.output.parse(e1_add_file, 'e1Detector'):
        # example of e1: <[('file', 'e1output.xml'), ('freq', '120'), ('friendlyPos', 'x'), ('id', 'e1det_-gneE19_0'),
        # ('lane', '-gneE19_0'), ('pos', '183.6')],child_dict={}>
        if e1.lane in lanes:
            e1s[e1.lane] = e1.id
    # some instances in e1s: {'-gneE19_0': 'e1det_-gneE19_0', '-gneE25_0': 'e1det_-gneE25_0',
    # '-gneE29_0': 'e1det_-gneE29_0', 'gneE15_0': 'e1det_gneE15_0'}
    return e1s


def program_split(each_tl):
    splits_of_programs = {}
    duration_of_programs = {}
    tlLogic = traci.trafficlight.getAllProgramLogics(each_tl)
    for logic in tlLogic:
        dura = []
        cycle_dura = 0
        for phase in logic.phases:
            dura.append(phase.duration)
            cycle_dura += phase.duration
        splits_of_programs.update({logic.programID: [dura[i] / cycle_dura for i in range(len(dura))]})
        duration_of_programs.update({logic.programID: cycle_dura})
    return splits_of_programs, duration_of_programs


def occu_info(occu_each_lane, e1s, tl_id):
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    avoid_lane_repeated = ""
    for lane in lanes:
        if avoid_lane_repeated != lane:
            occu_each_lane[lane] += (traci.inductionloop.getLastStepOccupancy(e1s[lane]) / 100)  # percentage to float
        # accumulate for the phase
        avoid_lane_repeated = lane
    return occu_each_lane


def construct_DS(tl_ids):
    DS_each_phase = {}
    for each_tl in tl_ids:
        DS_each_phase.update({each_tl: {}})
        tlLogic = traci.trafficlight.getAllProgramLogics(each_tl)
        # (Logic(programID='0', type=0, currentPhaseIndex=0, phases=(Phase(duration=42.0, state='GGgrrrGGgrrr',
        # minDur=42.0, maxDur=42.0, next=()), Phase(duration=3.0, state='yyyrrryyyrrr', minDur=3.0, maxDur=3.0,
        # next=()), Phase(duration=42.0, state='rrrGGgrrrGGg', minDur=42.0, maxDur=42.0, next=()), Phase(duration=3.0,
        # state='rrryyyrrryyy', minDur=3.0, maxDur=3.0, next=())), subParameter={}),)
        i = 0  # add index to phase
        for logic in tlLogic:
            if logic.programID == traci.trafficlight.getProgram(each_tl):
                for phase in logic.phases:
                    DS_each_phase[each_tl][i] = 0
            i += 1
    return DS_each_phase


def is_a_cycle(phase_count, tl_id):
    tlLogic = traci.trafficlight.getAllProgramLogics(tl_id)
    for logic in tlLogic:
        if logic.programID == traci.trafficlight.getProgram(tl_id):
            if len(logic.phases) == phase_count:
                return True
    return False


def infer_all_candidate(tl_id, DS_each_phase, splits_of_programs, duration_of_programs):
    DS_this_cycle = sum(DS_each_phase)
    this_program = traci.trafficlight.getProgram(tl_id)
    tlLogic = traci.trafficlight.getAllProgramLogics(tl_id)
    next_program = this_program  # initial
    DS_next_cycle = DS_this_cycle  # initial
    for split_id, split_dura in splits_of_programs.items():
        DS_temp = DS_next_cycle
        if split_id != this_program:
            DS_temp = 0
            for logic in tlLogic:
                if logic.programID == split_id:
                    for i in range(len(logic.phases)):
                        DS_temp += DS_each_phase[i] * splits_of_programs[this_program][i] * duration_of_programs[
                            this_program] / (splits_of_programs[split_id][i] * duration_of_programs[split_id])

        if DS_next_cycle > DS_temp:
            DS_next_cycle = DS_temp
            next_program = split_id
    return next_program


def run(tl_ids, e1_add_file):
    """Get predefined information
    1. the incoming lane (E1 detector) <lane:e1.id> -> tl_id
    2. the phase split, duration for all predefined signal plans -> infer DS
    """
    tl_lane_pair = {}
    for each_tl in tl_ids:
        e1s = get_e1(each_tl, e1_add_file)
        tl_lane_pair.update({each_tl: e1s})

    tl_program_split = {}
    tl_program_dura = {}
    for each_tl in tl_ids:
        splits_of_programs, duration_of_programs = program_split(each_tl)
        tl_program_split.update({each_tl: splits_of_programs})
        tl_program_dura.update({each_tl: duration_of_programs})

    """Store information for all controlled tls within a cycle
    1. DS for each phase in a cycle
    2. Phase count for each tl
    3. occupancy initial setting
    4. last switch time for each tl
    """

    # initial cycle
    DS_each_phase = construct_DS(tl_ids)  # dict, {<tl_id: <phase_index: 0>}
    Phase_count = {each_tl: 0 for each_tl in tl_ids}
    last_switch_time = {each_tl: 0 for each_tl in tl_ids}

    occu_info_all = {}
    for each_tl in tl_ids:
        traci.trafficlight.setProgram(each_tl, "0")
        occu_info_all.update({each_tl: {lane: 0 for lane in tl_lane_pair[each_tl].keys()}})
    while traci.simulation.getMinExpectedNumber() > 0:  # terminate until all vehicles complete their trips
        traci.simulationStep()  # Make a simulation step and simulate up to the given sim time (in seconds).
        current_time = traci.simulation.getTime()

        for each_tl in tl_ids:
            occu_each_lane = occu_info_all[each_tl]
            # next phase, absolute time counting from simulation start
            next_switch_time = traci.trafficlight.getNextSwitch(each_tl)
            # just phase switch, this is the first timestep for the current phase
            # or initial time
            if last_switch_time[each_tl] + 1 == current_time or current_time == 0:
                # green_start_time = current_time
                occu_each_lane = {lane: 0 for lane in tl_lane_pair[each_tl].keys()}
                Phase_count[each_tl] += 1

            # get traffic info for this timestep and accumulate for the phase
            occu_each_lane = occu_info(occu_each_lane, tl_lane_pair[each_tl], each_tl)

            occu_info_all[each_tl] = occu_each_lane
            if next_switch_time == current_time:
                last_switch_time[each_tl] = next_switch_time
                this_phase_dura = traci.trafficlight.getPhaseDuration(each_tl)
                DS_each_phase[each_tl][Phase_count[each_tl]] = 1 - max(occu_each_lane.values()) / this_phase_dura

                if is_a_cycle(Phase_count[each_tl], each_tl):
                    Phase_count[each_tl] = 0
                    # if current_time > 20700:
                    next_program = infer_all_candidate(each_tl, DS_each_phase[each_tl], tl_program_split[each_tl],
                                                       tl_program_dura[each_tl])
                    print(current_time, each_tl, traci.trafficlight.getProgram(each_tl), "-->", next_program)
                    print('--------')
                    traci.trafficlight.setProgram(each_tl, next_program)


import optparse


def get_options():
    optParse = optparse.OptionParser()
    optParse.add_option("--scen", type=str, default="center10",
                        help="scenario name, e.g., center10, center10_closing, center10_SCATS")
    optParse.add_option("--scen_path", type=str, default="../scenarios/UAV/", help="scenario directory")
    optParse.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    optParse.add_option("--noscats", action="store_true", default=False, help="set to implement scats")
    optParse.add_option("--run_num", type=int, default=1,
                        help="run how many times of simulation, and generate the index for output files, emission and tripinfo")
    options, args = optParse.parse_args()
    return options


import errno


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


from sumolib import checkBinary
import time

if __name__ == "__main__":
    options = get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    n_run = options.run_num
    scen = options.scen
    scen_path = options.scen_path
    output_path = f"../output/{scen}/"
    ensure_dir(output_path)

    for n in range(n_run):
        if not options.noscats:
            traci.start([sumoBinary, "-c", scen_path + scen + ".sumocfg", "--emission-output",
                         output_path + str(n) + "-emission.xml", "--tripinfo-output",
                         output_path + str(n) + "-tripinfo.xml"], numRetries=10, label=str(time.time()))

            controlled_tls = ['389279', '659784', 'cluster_389280_434149497', 'cluster_26868380_305313534', '389357',
                              '12639664']
            run(controlled_tls, "../scenarios/UAV/SCATS/e1.add.xml")

            traci.close()
        else:
            traci.start([sumoBinary, "-c", scen_path + scen + ".sumocfg", "--emission-output",
                         output_path + str(n) + "-emission.xml", "--tripinfo-output",
                         output_path + str(n) + "-tripinfo.xml"], numRetries=10, label=str(time.time()))

            while traci.simulation.getMinExpectedNumber() > 0:  # terminate until all vehicles complete their trips
                traci.simulationStep()  # Make a simulation step and simulate up to the given sim time (in seconds).

            traci.close()

