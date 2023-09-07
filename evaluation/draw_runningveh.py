import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def toFloat(list_):
	list = []
	for item in list_:
		item = float(item)
		list.append(item)
	return list


def main(hue_list):
	# csv file to store number of running veh for compared methods
	df = pd.read_csv('./compare_UAVTime.csv')  # example here

	print(df)

	timestep = []
	running_vehicles = []
	exp_id = []

	for each in hue_list:
		data = toFloat(df[each].tolist())
		timestep += [i for i in range(len(data))]
		running_vehicles += data
		exp_id.extend([each] * len(data))

	print(len(timestep), len(running_vehicles), len(exp_id))
	draw_df = pd.DataFrame({'Timestep (sec)': timestep, '#Running Vehicles': running_vehicles, ' ':exp_id})
	color_list = ['goldenrod', 'red', 'teal', 'olive', 'dodger blue', 'mauve', 'steel blue', 'green']
	palette = sns.xkcd_palette(color_list[0:len(hue_list)])  # , 'mauve', 'steel blue', 'green'
	fig, ax = plt.subplots()
	sns.lineplot(x=draw_df['Timestep (sec)'], y=draw_df['#Running Vehicles'], ax=ax, hue=draw_df[' '], palette=palette)
	plt.legend(title='', loc='best', fontsize='11')
	plt.xticks([i*300 for i in range(10)],fontsize=13)
	plt.yticks(fontsize=13)
	ax.set_xlabel('Timestep (sec)', fontsize=13)
	ax.set_ylabel('#Running Vehicles', fontsize=13)
	# ax.axvline(x=600, color='#e9df01', ls='dashed', linewidth=2)
	# ax.axvline(x=2400, color='#e9df01', ls='dashed', linewidth=2)
	ax.axvline(x=900, color='grey', ls='solid', linewidth=1)
	ax.axvline(x=1500, color='grey', ls='solid', linewidth=1)
	ax.axvline(x=2100, color='grey', ls='solid', linewidth=1)
	ax.axvline(x=2700, color='grey', ls='solid', linewidth=1)
	ax.axvspan(600, 2400, alpha=0.1, color='orange')
	# ax.axvspan(900, 2700, alpha=0.1, color='green')
	ax.text(y=230, x=1350, s='Road Closure', fontsize=12, color='grey')
	ax.arrow(y=235, x=1000, dx=-300, dy=0, width=0.01, head_width=5, head_length=100, fc='grey', ec='grey', shape='full')
	ax.arrow(y=235, x=2000, dx=300, dy=0, width=0.01, head_width=5, head_length=100, fc='grey', ec='grey', shape='full')
	# ax.text(y=200, x=1700, s='UAV Control', fontsize=11, color='grey')
	# ax.arrow(y=205, x=1400, dx=-400, dy=0, width=0.01, head_width=5, head_length=100, fc='grey', ec='grey', shape='full')
	# ax.arrow(y=205, x=2350, dx=250, dy=0, width=0.01, head_width=5, head_length=100, fc='grey', ec='grey', shape='full')
	ax.text(y=15, x=1000, s='UAVs start to control', fontsize=12, color='grey')
	ax.arrow(y=13, x=990, dx=-50, dy=-7.5, width=0.01, head_width=5, head_length=40, fc='grey', ec='grey', shape='full')
	plt.xlim(0, 2700)
	plt.ylim(0, 260)
	plt.grid(axis='y')
	plt.show()


if __name__ == '__main__':
	hue_list = ['Original', 'Congestion', 'SCATS', 'IntelliLight', 'AVARS']
	hue_list = ['Original', 'Congestion', 'AVARS-10min', 'AVARS-20min', 'AVARS-30min']

	main(hue_list)



