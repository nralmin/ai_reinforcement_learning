import sys
import random
import copy


def find_action_with_max_U(U, x, y):
    max_U = float('-inf')
    if y < DIM_Y - 1 and U[x][y + 1] > max_U and [x + 1, y + 2] not in OBSTACLE_COORD:
        action, max_U = '0', U[x][y + 1]  # '0' for North
    if x < DIM_X - 1 and U[x + 1][y] > max_U and [x + 2, y + 1] not in OBSTACLE_COORD:
        action, max_U = '1', U[x + 1][y]  # '1' for East
    if y > 0 and U[x][y - 1] > max_U and [x + 1, y] not in OBSTACLE_COORD:
        action, max_U = '2', U[x][y - 1]  # '2' for South
    if x > 0 and U[x - 1][y] > max_U and [x, y + 1] not in OBSTACLE_COORD:
        action, max_U = '3', U[x - 1][y]  # '3' for West
    return action, max_U


def value_iteration(R_s):
    U_iter = [[0 for y in range(DIM_Y)] for x in range(DIM_X)]
    delta = 0

    while True:
        U = copy.deepcopy(U_iter)
        delta = 0
        for x in range(DIM_X):
            for y in range(DIM_Y):
                _, max_neighbor_U = find_action_with_max_U(U, x, y)
                U_iter[x][y] = R_s[x][y] + GAMMA * max_neighbor_U
                change_in_U = abs(U_iter[x][y] - U[x][y])
                if change_in_U > delta:
                    delta = change_in_U
        if delta < THETA:
            break
    return U


def find_action_with_max_Q(Q, index):
    max_Q = max(Q[index])
    action = Q[index].index(max_Q)
    return action, max_Q


def go_North(s_index, R_s):
    x = s_index / DIM_Y
    y = s_index % DIM_Y
    if not (s_index + 1) % DIM_Y:  # hitting a wall
        s_p_index = s_index
        r = R_O
    elif [x + 1, y + 2] in OBSTACLE_COORD:
        s_p_index = s_index
        r = R_O
    else:
        s_p_index = s_index + 1
        r = R_s[x][y + 1]
    return (s_p_index, r)


def go_East(s_index, R_s):
    x = s_index / DIM_Y
    y = s_index % DIM_Y
    if s_index / DIM_Y >= DIM_X - 1:  # hitting a wall
        s_p_index = s_index
        r = R_O
    elif [x + 2, y + 1] in OBSTACLE_COORD:
        s_p_index = s_index
        r = R_O
    else:
        s_p_index = s_index + DIM_Y
        r = R_s[x + 1][y]
    return (s_p_index, r)


def go_South(s_index, R_s):
    x = s_index / DIM_Y
    y = s_index % DIM_Y
    if not s_index % DIM_Y:  # hitting a wall
        s_p_index = s_index
        r = R_O
    elif [x + 1, y] in OBSTACLE_COORD:
        s_p_index = s_index
        r = R_O
    else:
        s_p_index = s_index - 1
        r = R_s[x][y - 1]
    return (s_p_index, r)


def go_West(s_index, R_s):
    x = s_index / DIM_Y
    y = s_index % DIM_Y
    if not s_index / DIM_Y:  # hitting a wall
        s_p_index = s_index
        r = R_O
    elif [x, y + 1] in OBSTACLE_COORD:
        s_p_index = s_index
        r = R_O
    else:
        s_p_index = s_index - DIM_Y
        r = R_s[x - 1][y]
    return (s_p_index, r)


def q_learning(R_s, regular_cells):
    # initialize Q(s,a) table
    Q = [[0 for i in range(4)] for j in range(DIM_X * DIM_Y)]
    g_index = (GOAL_COORD[0] - 1) * DIM_Y + GOAL_COORD[1] - 1

    for _ in range(NUM_OF_EPISODES):
        # start from a random regular state
        s_i = random.randrange(len(regular_cells))
        x, y = regular_cells[s_i]
        s_index = x * DIM_Y + y  # index in Q(s,a) table

        while s_index != g_index:  # until the agent reaches the goal state

            randx = random.uniform(0, 1)
            if randx < EPSILON:  # exploration: choose a random action
                action = random.randrange(4)
            else:  # exploitation: choose the greediest action
                action, _ = find_action_with_max_Q(Q, s_index)

            # take the action, find out the new state and reward
            action_list = [go_North, go_East, go_South, go_West]
            s_p_index, r = action_list[action](s_index, R_s)

            # update the Q-value for the state
            _, max_Q = find_action_with_max_Q(Q, s_p_index)
            Q[s_index][action] = (1 - ALPHA) * Q[s_index][action] + ALPHA * (r + GAMMA * max_Q)

            s_index = s_p_index
    return Q


def construct_policy():
    R_s = build_R_2D_list()  # rewards
    if METHOD == 'V':
        U = value_iteration(R_s)
        construct_policy_V(U)
    elif METHOD == 'Q':
        regular_cells = find_regular_cell_indexes()
        Q = q_learning(R_s, regular_cells)
        construct_policy_Q(Q)


def construct_policy_V(U):
    with open(sys.argv[2], 'w') as f:
        for x in range(DIM_X):
            for y in range(DIM_Y):
                action, _ = find_action_with_max_U(U, x, y)
                print >>f, str(x + 1) + ' ' + str(y + 1) + ' ' + action


def construct_policy_Q(Q):
    with open(sys.argv[2], 'w') as f:
        for s_i in range(len(Q)):
            x = s_i / DIM_Y
            y = s_i % DIM_Y
            action, _ = find_action_with_max_Q(Q, s_i)
            print >>f, str(x + 1) + ' ' + str(y + 1) + ' ' + str(action)


def build_R_2D_list():
    R_s = [[R_D for y in range(DIM_Y)] for x in range(DIM_X)]
    for x,y in OBSTACLE_COORD:
        R_s[x - 1][y - 1] = R_O
    for x,y in PITFALL_COORD:
        R_s[x - 1][y - 1] = R_P
    x,y = GOAL_COORD
    R_s[x - 1][y - 1] = R_G

    return R_s


def find_regular_cell_indexes():
    indices = [[x,y] for x in range(DIM_X) for y in range(DIM_Y)]
    indices.remove([GOAL_COORD[0] - 1, GOAL_COORD[1] - 1])
    [indices.remove([x - 1, y - 1]) for [x, y] in OBSTACLE_COORD]
    [indices.remove([x - 1, y - 1]) for [x, y] in PITFALL_COORD]
    return indices


def print_2D_list(config):
    for y in range(DIM_Y):
        for x in range(DIM_X):
            config_ud = config[x][::-1]
            print str(config_ud[y]) + ' ',
        print '\n'


if __name__ == "__main__":
    # read task parameters from input file
    # their values remain constant throughout the program
    with open(sys.argv[1]) as f:
        content = f.read().splitlines()
        METHOD = content[0]
        offset = 0

        if METHOD == 'V':
            THETA = float(content[1])
            GAMMA = float(content[2])

        elif METHOD == 'Q':
            NUM_OF_EPISODES = int(content[1])
            ALPHA = float(content[2])
            GAMMA = float(content[3])
            EPSILON = float(content[4])
            offset = 2

        DIM_Y, DIM_X = map(int, content[offset + 3].split(' '))
        NUM_OF_OBSTACLES = int(content[offset + 4])

        OBSTACLE_COORD = [map(int, c.split(' '))
            for c in content[offset + 5:offset + 5 + NUM_OF_OBSTACLES]]

        NUM_OF_PITFALLS = int(content[offset + 5 + NUM_OF_OBSTACLES])

        PITFALL_COORD = [map(int, c.split(' ')) for c in content[offset + 6 +
            NUM_OF_OBSTACLES:offset + 6 + NUM_OF_OBSTACLES + NUM_OF_PITFALLS]]

        GOAL_COORD = map(int,
            content[offset + 6 + NUM_OF_OBSTACLES + NUM_OF_PITFALLS].split(' '))

        rewards = content[offset + 7 +
                          NUM_OF_OBSTACLES + NUM_OF_PITFALLS].split(' ')
        # rewards for regular cell, obstacle, pitfall, and goal
        R_D, R_O, R_P, R_G = map(float, rewards)

        construct_policy()
