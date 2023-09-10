import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib


def choose_files(path, scenario, avg_num, recover_bool):
    """
    Choose the corresponding output files for multiple simulation episodes, tripinfo files included as default

    Parameters
    ----------
    path: str, path to scenario directory
    scenario: str, scenario name
    avg_num: int, how many episodes to get the average, default is 8
    recover_bool: bool, whether to calibrate output file format

    Returns
    -------
    file_dict: dict, {file_type: [file_name*avg_num]}
    """
    file_num = 0
    related_file_list = []  # except for tripinfo file, such as ['-emission.xml', '-queue.xml', '_emission.csv']
    file_dict = {'-tripinfo.xml': []}
    scenarios_list = os.listdir(path)
    if scenario in scenarios_list:
        dir_path = os.path.join(path, scenario)
        for file in os.listdir(dir_path):
            if file.endswith('tripinfo.xml'):
                if recover_bool:
                    recover_xml(dir_path + '/' + file)
                file_dict['-tripinfo.xml'].append(file)
                file_num += 1

                file_suffix = file.split('-tripinfo')[0]
                for each in related_file_list:
                    file_name = file_suffix + each
                    if recover_bool:
                        recover_xml(dir_path + '/' + file)
                    file_dict[each].append(file_name)
            if file_num == avg_num:
                break
    print('Evaluated files:', file_dict['-tripinfo.xml'])
    return file_dict


def recover_xml(file_path):
    """ Recovers incomplete output xml files written during the simulation,
    closing tag element is probably missing when simulation terminates """
    cmd = "xmllint --valid " + file_path + " --recover --output " + file_path
    print()
    print('-----')
    print('Recover the xml file')
    os.system(cmd)


def get_main_metric(tripinfo_files, path, speed_limit):
    print()
    print('---------')
    print(path.split('/')[-1])
    fuel = [0] * len(tripinfo_files)
    CO2 = [0] * len(tripinfo_files)
    duration = [0] * len(tripinfo_files)
    all_trip_length = []
    finished_trip_num = []
    tt_veh = {}  # {veh_id: [travel_time_i]}
    for i in range(0, len(tripinfo_files)):
        trip_length = {}  # {trip length(m): vehicle count}
        for trip in sumolib.output.parse(path + '/' + tripinfo_files[i], ['tripinfo']):
            duration[i] += float(trip.duration)  # sum of travel time
            # travel time for individual vehicles
            if trip.id not in tt_veh.keys():
                tt_veh.update({trip.id: [float(trip.duration)]})
            else:
                tt_veh[trip.id].append(float(trip.duration))

            fuel[i] += float(trip['emissions'][0].fuel_abs)
            CO2[i] += float(trip['emissions'][0].CO2_abs)

            triplength = float(trip.routeLength)
            if trip_length.get(triplength):
                trip_length[triplength] += 1
            else:
                trip_length.update({triplength: 1})
        all_trip_length.append(sum(triplength * num for triplength, num in trip_length.items()))
        finished_trip_num.append(sum(num for num in trip_length.values()))  # only finished trip in tripinfo file

    # get average value and convert unit
    avg_all_trip_length = np.mean(all_trip_length)
    avg_finished_trip_num = np.mean(finished_trip_num)
    print('Average finished trip number in this scenario: ', avg_finished_trip_num)
    print('Average vehicle trip length (m): ', avg_all_trip_length / avg_finished_trip_num)
    fuel = np.mean(fuel) / (avg_all_trip_length / 100)  # ml/m -> l/100km
    CO2 = np.mean(CO2) / avg_all_trip_length  # mg/m -> g/km
    duration = np.mean(duration) / avg_finished_trip_num

    return fuel, CO2, duration, tt_veh


def have_df_metric(scen_file_dict, path_dir, speed_limit, scenarios_legend):
    avg_fuel = []
    avg_CO2 = []
    avg_dura = []

    travelt_all = []
    traveltime_df = pd.DataFrame()
    scen_tag = []
    for scenario in scen_file_dict.keys():
        fuel, CO2, duration, travelt = get_main_metric(scen_file_dict[scenario]['-tripinfo.xml'],
                                                              path_dir + '/' + scenario, speed_limit)
        avg_fuel.append(fuel)
        avg_CO2.append(CO2)
        avg_dura.append(duration)

        traveltime_avg_each_vehicle = []
        for tt_list in travelt.values():
            traveltime_avg_each_vehicle.append(np.mean(tt_list))
        travelt_all.extend(traveltime_avg_each_vehicle)
        if scenarios_legend:
            scen_tag += [scenarios_legend[list(scen_file_dict.keys()).index(scenario)]] \
                * len(traveltime_avg_each_vehicle)
            traveltime_df.insert(len(traveltime_df.columns),
                                 scenarios_legend[list(scen_file_dict.keys()).index(scenario)],
                                 traveltime_avg_each_vehicle)
        else:
            scen_tag += [scenario] * len(traveltime_avg_each_vehicle)
            traveltime_df.insert(len(traveltime_df.columns), scenario, traveltime_avg_each_vehicle)


    dataframe = pd.DataFrame(
        {'Scenario': scen_file_dict.keys(), 'Fuel consumption l/100km': avg_fuel, 'CO2 emission g/1km': avg_CO2,
         'Average travel time': avg_dura})
    print()
    print("Main traffic statistics:")
    print(dataframe)
    print("Statistics of travel time (sec)")
    print(traveltime_df.describe())
    return travelt_all, scen_tag


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when evaluating an experiment result.",
        epilog="python outputFilesProcessing.py")

    # necessary
    parser.add_argument(
        '--scen', nargs='+',
        help='The list of several scenario names.')

    # optional
    parser.add_argument(
        '--output_dir', type=str, default="../output/",
        help='The directory of output files contains multiple scenarios.')

    parser.add_argument(
        '--scen_legend', nargs='+',
        help='The list of several scenario names in figure legend.')

    parser.add_argument(
        '--speed_limit', type=int, default=15,
        help='Speed limit.')

    parser.add_argument(
        '--avg_num', type=int, default=18,
        help='How many scenarios to get the average result.')

    parser.add_argument(
        '--recover', type=bool, default=False,
        help='About xml.etree.ElementTree.ParseError, set to True')

    return parser.parse_known_args(args)[0]


def main(args):
    flags = parse_args(args)
    if flags.scen:
        path_dir = flags.output_dir
        scenarios_list = flags.scen
        scenarios_legend = flags.scen_legend
        speed_limit = flags.speed_limit
        avg_num = flags.avg_num
        recover_bool = flags.recover
        # choose some scenarios for assessment
        scen_file_dict = {}
        for scenario in scenarios_list:
            scen_file_dict.update({scenario: choose_files(path_dir, scenario, avg_num, recover_bool)})
        # show the main metric results, CO2 emissions, fuel consumption, and travel time
        have_df_metric(scen_file_dict, path_dir, speed_limit, scenarios_legend)


    else:
        raise ValueError("Unable to find necessary options: --scen.")


if __name__ == '__main__':
    main(sys.argv[1:])
