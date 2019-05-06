import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    ## YOUR CODE HERE
    start_node = Node(problem.getStartState(),None,0,None,0)
    frontier = util.Stack()
    frontier.push(start_node)
    visited = util.Counter()

    while(not frontier.isEmpty()):
        cur_node = frontier.pop()
        state_hash = convertStateToHash(cur_node.state)
        visited[state_hash] = 1
        if problem.isGoalState(cur_node.state):
            return cur_node.state
        for next_state in problem.getSuccessors(cur_node.state):
            next_state_hash = convertStateToHash(next_state[0])
            if visited[next_state_hash] == 0:
                next_node = Node(next_state[0],next_state[1],cur_node.path_cost + next_state[2], cur_node, cur_node.depth + 1)
                frontier.push(next_node)

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(node, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.

    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """
    return util.points2distance([(problem.G.node[node.state]['x'],0,0),(problem.G.node[node.state]['y'],0,0)], [(problem.G.node[problem.end_node]['x'],0,0),(problem.G.node[problem.end_node]['y'],0,0)])

def AStar_search(problem, heuristic=nullHeuristic):

    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """

    def get_priority(state, problem):
        priority = state.path_cost + heuristic(state, problem)
        return priority

    start_node = Node(problem.getStartState(),None,0,-1,0)
    frontier = util.PriorityQueue()
    frontier.push(start_node, get_priority(start_node, problem))
    visited = util.Counter()
    end_node = None

    while(not frontier.isEmpty()):
        cur_node = frontier.pop()
        if visited[cur_node.state] == 0:
            visited[cur_node.state] = 1
            if problem.isGoalState(cur_node.state):
                end_node = cur_node
                break
            for next_state in problem.getSuccessors(cur_node.state):
                if visited[next_state[0]] == 0:
                    next_node = Node(next_state[0],next_state[1],cur_node.path_cost + next_state[2], cur_node, cur_node.depth + 1)
                    frontier.push(next_node, get_priority(next_node, problem))


    route = []
    present_node = end_node
    while(present_node != -1):
        route.append(present_node.state)
        present_node = present_node.parent_node
    
    route.reverse()
    return route