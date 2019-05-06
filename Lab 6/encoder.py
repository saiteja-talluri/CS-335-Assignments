import sys, math
import numpy as np

def preprocess(filepath, probabilty) :
	try:
		fd = open(filepath, "r")
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

	return width, height, start, end, maze, probabilty


def print_transitions(i,j,height,width,maze,p):
	state, top_state, right_state, bottom_state, left_state= i*width+j, (i-1)*width+j, i*width+j+1, (i+1)*width+j, i*width+j-1
	if(maze[(i,j)] == 3 or maze[(i,j)] == 1):
		return None
	else:
		a = i > 0 and maze[(i-1,j)] != 1
		b = j > 0 and maze[(i,j-1)] != 1
		c = j < (width-1) and maze[(i,j+1)] != 1
		d = i < (height-1) and maze[(i+1,j)] != 1

		total = int(a) + int(b) + int(c) + int(d)
		p_rand = (1-p)/total
		p_actual = p + p_rand

		if(a):
			if(a and p_actual!=0):
				print "transition",state,0,top_state,-1,p_actual
			if(b and p_rand!=0):
				print "transition",state,0,left_state,-1,p_rand
			if(c and p_rand!=0):
				print "transition",state,0,right_state,-1,p_rand
			if(d and p_rand!=0):
				print "transition",state,0,bottom_state,-1,p_rand
		if(b):
			if(a and p_rand!=0):
				print "transition",state,1,top_state,-1,p_rand
			if(b and p_actual!=0):
				print "transition",state,1,left_state,-1,p_actual
			if(c and p_rand!=0):
				print "transition",state,1,right_state,-1,p_rand
			if(d and p_rand!=0):
				print "transition",state,1,bottom_state,-1,p_rand
		if(c):
			if(a and p_rand!=0):
				print "transition",state,2,top_state,-1,p_rand
			if(b and p_rand!=0):
				print "transition",state,2,left_state,-1,p_rand
			if(c and p_actual!=0):
				print "transition",state,2,right_state,-1,p_actual
			if(d and p_rand!=0):
				print "transition",state,2,bottom_state,-1,p_rand
		if(d):
			if(a and p_rand!=0):
				print "transition",state,3,top_state,-1,p_rand
			if(b and p_rand!=0):
				print "transition",state,3,left_state,-1,p_rand
			if(c and p_rand!=0):
				print "transition",state,3,right_state,-1,p_rand
			if(d and p_actual!=0):
				print "transition",state,3,bottom_state,-1,p_actual
	return None

def encode((width,height,start,end,maze,probabilty)):
	S = width*height
	A = 4
	gamma = 1.0
	print "numStates",S
	print "numActions",A
	print "start",start
	if(len(end) == 0):
		print "end", -1
	else:
		print "end",' '.join(map(str,end))
	for i in range(height):
		for j in range(width):
			print_transitions(i,j,height,width,maze,probabilty)
	print "discount ",gamma
	return None

if __name__ == "__main__":

	if len(sys.argv) != 3:
		print('Usage: python encoder.py gridfile probabilty\n')
		sys.exit(1)
	
	encode(preprocess(sys.argv[1], float(sys.argv[2])))