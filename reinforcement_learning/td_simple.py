from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, row=3, col=1):
        self.row = row
        self.col = col
        self.goal = (0, 0)
        self.reward_goal = 1
        
    def get_reward(self, state):
        if state == self.goal:
            return self.reward_goal
        else:
            return 0
    # up: 0, right: 1, down: 2, left: 3
    def get_next_state(self, state, action):
        if action == 0:
            # print("up")
            next_state = (state[0] - 1, state[1])
        elif action == 1:
            # print("right")
            next_state = (state[0], state[1] + 1)
        elif action == 2:
            # print("down")
            next_state = (state[0] + 1, state[1])
        elif action == 3:
            # print("left")
            next_state = (state[0], state[1] - 1)
        else:
            print("Error: action is not defined")
            return state
        if next_state[0] < 0 or next_state[0] >= self.row or next_state[1] < 0 or next_state[1] >= self.col:
            # print("Error: next_state is over grid world")
            return state
        return next_state

class TdAgent:
    def __init__(self, alpha=0.1, gamma=0):
        self.alpha = alpha
        self.gamma = gamma
        self.actions = [0, 1, 2, 3]
        self.V = defaultdict(lambda: 0)
        # すべての行動に対する確率 ここでは↑ のみに固定
        self.pi = defaultdict(lambda: [1, 0, 0, 0])
        # グラフの可視化用
        self.episode = []
        self.V_10= []
        self.V_20= []
    def get_action(self, state) :
        return np.random.choice(self.actions, p=self.pi[state])
    
    def eval(self, state, reward, next_state) :
        target = reward + self.gamma * self.V[next_state]
        self.V[state] += self.alpha * (target - self.V[state])
        
grid = GridWorld()
agent = TdAgent()

max_episode = 100

for episode in range(max_episode):
    # 初期位置
    state = (2, 0)
    while True:
        action = agent.get_action(state)
        next_state = grid.get_next_state(state, action)
        reward = grid.get_reward(next_state)
        agent.eval(state, reward, next_state)
        state = next_state
        agent.episode.append(episode)
        agent.V_10.append(agent.V[(1, 0)])
        agent.V_20.append(agent.V[(2, 0)])
        
        if state == grid.goal:
            # print(agent.V[(1, 0)], agent.V[(2, 0)])
            break    

# agent.V をいい感じに表示する
for i in range(grid.row):
    for j in range(grid.col):
        print("{:.2f}".format(agent.V[(i, j)]), end=" ")
    print("")

# agent.episode と agent.V_10, agent.V_20 をグラフにする 軸の名前も表示
plt.xlabel("episode")
plt.ylabel("V", rotation=0)
plt.plot(agent.episode, agent.V_10, label="V(1, 0)")
plt.plot(agent.episode, agent.V_20, label="V(2, 0)")
plt.legend()
plt.show()
