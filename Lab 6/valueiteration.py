import sys, math
import numpy as np

def preprocess(filepath) :
	try:
		fd = open(filepath, "r")
	except IOError:
		print('mdpFile not present')
		sys.exit(1)
	else:
		lines = fd.readlines()
		numStates,numActions,start,gamma = 0,0,0,0.0
		end  = []
		reward_prob, actions = {},None

		for i in range(len(lines)):
			tokens = lines[i].rstrip('\n').split()
			if(tokens[0] == "numStates"):
				numStates = int(tokens[1])
				actions = [{} for _ in range(numStates)]
			elif(tokens[0] == "numActions"):
				numActions = int(tokens[1])
			elif(tokens[0] == "start"):
				start = int(tokens[1])
			elif(tokens[0] == "end"):
				if(int(tokens[1]) != -1):
					for i in range(1, len(tokens)):
						end.append(int(tokens[i]))
			elif(tokens[0] == "transition"):
				reward_prob[(int(tokens[1]),int(tokens[2]),int(tokens[3]))] = (float(tokens[4]),float(tokens[5]))
				if int(tokens[2]) not in actions[int(tokens[1])].keys():
					actions[int(tokens[1])][int(tokens[2])] = [int(tokens[3])]
				else:
					actions[int(tokens[1])][int(tokens[2])].append(int(tokens[3]))
			elif(tokens[0] == "discount"):
				gamma = float(tokens[1])
			else:
				print('mdpFile Corrupted')
				sys.exit(1)
			fd.close()

	return numStates,numActions,start,end,actions,reward_prob,gamma

def value_iteration_policy((numStates,numActions,start,end,actions,reward_prob,gamma)):
	values = [0 for _ in range(numStates)]
	values_updated = [0 for _ in range(numStates)]
	optimal_action = {}
	t = 0
	is_converged = False

	while((t == 0) or not(is_converged)):
		is_converged = True
		values = list(values_updated)
		for i in range(numStates):
			optimal_action[i] = -1
			if i in end:
				values_updated[i] = 0
			elif(len(actions[i]) == 0):
				values_updated[i] = 0
			else:
				ans = -10**40
				for j in actions[i].keys():
					val = 0
					for k in actions[i][j]:
						val += reward_prob[(i,j,k)][1]*(reward_prob[(i,j,k)][0] + gamma*values[k])
					if(val > ans):
						optimal_action[i] = j
					ans = max(ans, val)
				values_updated[i] = ans
				if(abs(values_updated[i] - values[i]) > 10**-16):
					is_converged = False
		t += 1

	for i in range(numStates):
		print(str(values_updated[i]) + " " + str(optimal_action[i]))
	print("iterations " + str(t))

if __name__ == "__main__":

	if len(sys.argv) < 2:
		print('Usage: python valueiteration.py mdpFileName\n')
		sys.exit(1)

	value_iteration_policy(preprocess(sys.argv[1]))


