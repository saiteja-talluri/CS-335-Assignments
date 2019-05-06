import os,sys,math
import numpy as np
import matplotlib.pyplot as plt

def preprocess() :
	pro_act = {}
	probabilty = []
	actions = []
	for filepath in os.listdir("test"):
		try:
			fd = open("test/" + filepath, "r")
		except IOError:
			print('file not present')
			sys.exit(1)
		else:
			lines = fd.readlines()
			prob = float(lines[0])
			length = len(lines[1])/2
			pro_act[prob] = length
			fd.close()
			# os.system("rm " + "test/" + filepath)
	
	for i in pro_act.keys():
		probabilty.append(i)
	probabilty.sort()
	for i in probabilty:
		actions.append(pro_act[i])

	return actions,probabilty

def plot_stochastic((actions, probabilty)):
	print(probabilty)
	print(actions)
	plt.plot(probabilty, actions)
	plt.ylabel('Number of actions to reach the exit')
	plt.xlabel('p')
	plt.show()

if __name__ == "__main__":

	if len(sys.argv) != 3:
		print('Usage: python plot.py gridfilesize no_of_samples')
		sys.exit(1)

	gridfilesize = int(sys.argv[1])
	no_of_samples = int(sys.argv[2])

	for i in range(no_of_samples + 1):
		prob = i*(1.0/no_of_samples)
		cmd = "./test.sh test" + " " + str(gridfilesize) + " " + str(prob)
		os.system(cmd)

	plot_stochastic(preprocess())