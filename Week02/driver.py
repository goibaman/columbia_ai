import Queue

import time

import resource

import sys

import math

# The Class that Represents the Puzzle


class PuzzleState(object):

    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n*n != len(config) or n < 2:

            raise Exception("the length of config is not correct!")

        self.n = n

        self.cost = cost

        self.parent = parent

        self.action = action

        self.dimension = n

        self.config = config

        self.children = []

        for i, item in enumerate(self.config):

            if item == 0:

                self.blank_row = i / self.n

                self.blank_col = i % self.n

                break

    def display(self):

        for i in range(self.n):

            line = []

            offset = i * self.n

            for j in range(self.n):

                line.append(self.config[offset + j])

            print line

    def move_left(self):

        if self.blank_col == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):

        if self.blank_col == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):

        if self.blank_row == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):

        if self.blank_row == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):

        """expand the node"""

        # add child nodes in order of UDLR

        if len(self.children) == 0:

            up_child = self.move_up()

            if up_child is not None:

                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:

                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:

                self.children.append(left_child)

            right_child = self.move_right()

            if right_child is not None:

                self.children.append(right_child)

        return self.children


def write_output(path_to_goal, cost_of_path, nodes_expanded, search_depth, max_search_depth, running_time, max_ram_usage):

    # path_to_goal: the sequence of moves taken to reach the goal
    output = "path_to_goal: " + str(path_to_goal) + "\n"

    # cost_of_path: the number of moves taken to reach the goal
    output += "cost_of_path: " + str(cost_of_path) + "\n"

    # nodes_expanded: the number of nodes that have been expanded
    output += "nodes_expanded: " + str(nodes_expanded) + "\n"

    # search_depth: the depth within the search tree when the goal node is found
    output += "search_depth: " + str(search_depth) + "\n"

    # max_search_depth:  the maximum depth of the search tree in the lifetime of the algorithm
    output += "max_search_depth: " + str(max_search_depth) + "\n"

    # running_time: the total running time of the search instance, reported in seconds
    output += "running_time: " + str(round(running_time,8)) + "\n"

    # max_ram_usage: the maximum RAM usage in the lifetime of the process as measured by the ru_maxrss attribute in the resource module, reported in megabytes
    output += "max_ram_usage: " + str(round(max_ram_usage,8)) + "\n"

    # Open output.txt and write file contents.
    with open('output.txt', 'w') as output_stream:
        output_stream.write(output)

    print(output)
    return


def bfs_search(initial_state):

    """BFS search"""

    # Initialization
    frontier = list([initial_state])
    explored = set()
    explored.add(initial_state.config)
    search_depth = 0
    max_search_depth = 0

    # Start clock
    start_time = time.time()

    # Process Loop
    while len(frontier) != 0:

        # Step 01 - Remove a node from the frontier set.
        current_state = frontier.pop(0)

        # Step 02 - Check the state against the goal state to determine if a solution has been found.
        if test_goal(current_state):

            # Post Process
            state = current_state
            path_to_goal = []

            while state.parent is not None:
                path_to_goal.insert(0, state.action)
                state = state.parent

            # Output
            write_output(path_to_goal, current_state.cost, search_depth, len(path_to_goal), max_search_depth, time.time() - start_time, float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024)
            return

        # Step 03 - If the result of the check is negative, we then expand the node
        else:
            children = current_state.expand()
            search_depth += 1

            if len(children) > 0:
                for state in children:
                    if state.config not in explored:
                        frontier.append(state)
                        explored.add(state.config)
                        if state.cost > max_search_depth:
                            max_search_depth = state.cost
    return


def dfs_search(initial_state):

    """DFS search"""

    # Initialization
    frontier = list([initial_state])
    explored = set()
    explored.add(initial_state.config)
    search_depth = 0
    max_search_depth = 0

    # Start clock
    start_time = time.time()

    # Process Loop
    while len(frontier) != 0:

        # Step 01 - Remove a node from the frontier set.
        current_state = frontier.pop()

        # Step 02 - Check the state against the goal state to determine if a solution has been found.
        if test_goal(current_state):

            # Post Process
            state = current_state
            path_to_goal = []

            while state.parent is not None:
                path_to_goal.insert(0, state.action)
                state = state.parent

            # Output
            write_output(path_to_goal, current_state.cost, search_depth, len(path_to_goal), max_search_depth, time.time() - start_time, float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024)
            return

        # Step 03 - If the result of the check is negative, we then expand the node
        else:
            children = current_state.expand()
            search_depth += 1
            children.reverse()

            if len(children) > 0:
                for state in children:
                    if state.config not in explored:
                        frontier.append(state)
                        explored.add(state.config)
                        if state.cost > max_search_depth:
                            max_search_depth = state.cost
    return


def A_star_search(initial_state):

    """A * search"""

    # Initialization
    frontier = Queue.PriorityQueue()
    frontier.put((0, initial_state))
    explored = set()
    explored.add(initial_state.config)
    search_depth = 0
    max_search_depth = 0

    # Start clock
    start_time = time.time()

    # Process Loop
    while frontier.qsize() != 0:

        # Step 01 - Remove a node from the frontier set.
        current_state = frontier.get()[1]

        # Step 02 - Check the state against the goal state to determine if a solution has been found.
        if test_goal(current_state):

            # Post Process
            state = current_state
            path_to_goal = []

            while state.parent is not None:
                path_to_goal.insert(0, state.action)
                state = state.parent

            # Output
            write_output(path_to_goal, current_state.cost, search_depth, len(path_to_goal), max_search_depth,
                         time.time() - start_time, float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024)
            return

        # Step 03 - If the result of the check is negative, we then expand the node
        else:
            children = current_state.expand()
            search_depth += 1

            if len(children) > 0:
                for state in children:
                    if state.config not in explored:
                        total_distance = calculate_total_cost(state) + calculate_total_manhattan_dist(state)
                        frontier.put((total_distance, state))
                        explored.add(state.config)
                        if state.cost > max_search_depth:
                            max_search_depth = state.cost
    return


def calculate_total_cost(state):

    cost = 0
    current_state = state
    while current_state.parent != None:
        cost += 1
        current_state = current_state.parent

    return cost

def calculate_total_manhattan_dist(state):

    total = 0
    for n in range(0, state.n**2):
        total += calculate_manhattan_dist(n, state.config[n], state.n)

    return total


def calculate_manhattan_dist(idx, value, n):

    """calculate the manhattan distance of a tile"""
    return abs((idx % n) - (value % n)) + abs((idx // n) - (value // n))



def test_goal(puzzle_state):

    """test the state is the goal state or not"""
    for i in range(0, (puzzle_state.n ** 2) - 1):
        if puzzle_state.config[i] != i:
            return False

    return True

# Main Function that reads in Input and Runs corresponding Algorithm


def main():

    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)

    if sm == "bfs":

        bfs_search(hard_state)

    elif sm == "dfs":

        dfs_search(hard_state)

    elif sm == "ast":

        A_star_search(hard_state)

    else:

        print("Enter valid command arguments !")


if __name__ == '__main__':

    main()
