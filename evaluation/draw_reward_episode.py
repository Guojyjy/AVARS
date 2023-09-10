import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import numpy as np


def toFloat(list_):
	list = []
	for item in list_:
		item = float(item)
		list.append(item)
	return list


def read_file(file, agent_type, legend_tag):
	interation = []
	avg = []
	tag_this = []

	data = pd.read_csv(file)
	tt = toFloat(list(data['training_iteration']))
	interation += tt * 3

	avg += toFloat(list(data['policy_reward_mean/'+agent_type]))
	avg_values = np.array(toFloat(list(data['policy_reward_mean/'+agent_type])))
	std_values = np.array(toFloat(list(data['policy_reward_std/'+agent_type])))
	avg += list(avg_values + std_values)
	avg += list(avg_values - std_values)
	tag_this += [legend_tag]*len(avg_values)*3
	return interation, avg, tag_this


def draw_plot(interation_num, avg_reward, tagg):

	dataframe = pd.DataFrame({'Training iteration': interation_num, 'Average reward': avg_reward, ' ':tagg})
	print(dataframe)

	fig, ax = plt.subplots()
	sns.lineplot(x=dataframe["Training iteration"], y=dataframe["Average reward"], ax=ax , hue=dataframe[" "])
	sns.color_palette('bright')
	plt.legend(title='', loc='lower right', fontsize='13')
	plt.xticks(fontsize=13)
	plt.yticks(fontsize=13)
	plt.xlim(-5, 150)
	ax.set_xlabel("Training Iteration", fontsize=14)
	ax.set_ylabel("Average Episode Reward", fontsize=14)
	ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=-1400, decimals=2))
	plt.grid()
	plt.show()


def UAV():
	scenario_list = ['IntelliLight_ppo/PPO_UAVEnvIntelliLight-v1/']
	scenario_legend = ['IntelliLight(PPO)']

	scenario_list = ['AVARS_dqn/DQN_UAVEnvAVARS-v1/']
	scenario_legend = ['AVARS(DQN)']

	return scenario_list, scenario_legend


def main_uav():
	path = '~/ray_results/'
	prog = '/progress.csv'
	scen_list, scen_legend = UAV()

	interation_num = []
	avg_reward = []  # include std range
	tagg = []

	for each in scen_list:
		interation, avg, tag_this = read_file(path+each+prog, 'uav', scen_legend[scen_list.index(each)])
		interation_num += interation
		avg_reward += avg
		tagg += tag_this
	draw_plot(interation_num, avg_reward, tagg)


if __name__ == '__main__':
	main_uav()
