import sys, math
import numpy as np

randomSeed = 0
np.random.seed(randomSeed)

def preprocess(grid_file_path, value_policy_file_path, probability) :
	try:
		fd = open(grid_file_path, "r")
	except IOError:
		print('gridfile not present')
		sys.exit(1)
	else:
		lines = fd.readlines()
		height = len(lines)
		width = len(lines[0])/2
		start = None
		end = []
		maze = {}
		for i in range(height):
			for j in range(width):
				maze[(i,j)] = int(lines[i][2*j])
				if(maze[(i,j)] == 2):
					start = width*i + j
				elif(maze[(i,j)] == 3):
					end.append(width*i + j)
		fd.close()

	try:
		fd = open(value_policy_file_path, "r")
	except IOError:
		print('value_and_policy_file not present')
		sys.exit(1)
	else:
		value_policy = {}
		lines = fd.readlines()
		for i in range(width*height):
			tokens = lines[i].split()
			value_policy[i] = (float(tokens[0]),float(tokens[1]))
		fd.close()
	
	return width, height, start, end, maze, value_policy, probability

def print_path(path):
	ans = ""
	for i in range(len(path)):
		a = path[i]
		if(a == 0):
			ans = ans + "N"
		elif(a == 1):
			ans = ans + "W"
		elif(a == 2):
			ans = ans + "E"
		else:
			ans = ans + "S"
		if i!=len(path)-1:
			ans += " "
	print(ans)
	return None

def decode((width, height, start, end, maze, value_policy, probability)):
	p = float(probability)
	state = start
	path = []

	while(state not in end):
		i = state/width
		j = state%width
		
		a = i > 0 and maze[(i-1,j)] != 1
		b = j > 0 and maze[(i,j-1)] != 1
		c = j < (width-1) and maze[(i,j+1)] != 1
		d = i < (height-1) and maze[(i+1,j)] != 1

		total = int(a) + int(b) + int(c) + int(d)
		p_rand = (1-p)/total
		p_actual = p + p_rand

		x = value_policy[state][1]
		choice = 0
		if(x == 0):
			choice = np.random.choice(4, 1, p=[p_actual*int(a),p_rand*int(b),p_rand*int(c),p_rand*int(d)])[0]
		elif(x == 1):
			choice = np.random.choice(4, 1, p=[p_rand*int(a),p_actual*int(b),p_rand*int(c),p_rand*int(d)])[0]
		elif(x == 2):
			choice = np.random.choice(4, 1, p=[p_rand*int(a),p_rand*int(b),p_actual*int(c),p_rand*int(d)])[0]
		else:
			choice = np.random.choice(4, 1, p=[p_rand*int(a),p_rand*int(b),p_rand*int(c),p_actual*int(d)])[0]

		if(choice == 0):
			i -= 1
		elif(choice == 1):
			j -= 1
		elif(choice == 2):
			j += 1
		else:
			i += 1
		path.append(choice)
		state = i*width+j

	print_path(path)
	return None

if __name__ == "__main__":

	if len(sys.argv) != 4:
		print('Usage: python encoder.py gridfile value_and_policy_file probability\n')
		sys.exit(1)

	decode(preprocess(sys.argv[1], sys.argv[2], sys.argv[3]))