import matplotlib.pyplot as plt
import numpy as np

Goal = np.random.randint(0, 100, size=(1, 2)) / 100
X_goal, y_goal = Goal[0][0], Goal[0][1]

R = np.random.randint(0, 100, size=(4, 2)) / 100
R1X, R1y = R[0]
R2X, R2y = R[1]

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def update_positions(R1X, R1y, R2X, R2y, X_goal, y_goal):
    dist_R1 = distance(R1X, R1y, X_goal, y_goal)
    if dist_R1 > 0:
        step_size = min(0.01, dist_R1)
        R1X += step_size * (X_goal - R1X) / dist_R1
        R1y += step_size * (y_goal - R1y) / dist_R1

    dist_R2 = distance(R2X, R2y, X_goal, y_goal)
    if dist_R2 > 0:
        step_size = min(0.01, dist_R2)
        R2X += step_size * (X_goal - R2X) / dist_R2
        R2y += step_size * (y_goal - R2y) / dist_R2

    return R1X, R1y, R2X, R2y

def update_positions_random(R1X, R1y, R2X, R2y):
    R1X += (np.random.rand() - 0.5) * 0.05
    R1y += (np.random.rand() - 0.5) * 0.05

    R2X += (np.random.rand() - 0.5) * 0.05
    R2y += (np.random.rand() - 0.5) * 0.05

    return R1X, R1y, R2X, R2y

iterations = 100

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Signal Tracker')

ax1.scatter(X_goal, y_goal, color='red', label='Goal')
R1X_random = R1X.copy()
R1y_random = R1y.copy()
R2X_random = R2X.copy()
R2y_random = R2y.copy()

for _ in range(iterations):
    R1X, R1y, R2X, R2y = update_positions(R1X, R1y, R2X, R2y, X_goal, y_goal)

    R1X_random, R1y_random, R2X_random, R2y_random = update_positions_random(R1X_random,
                                                                          R1y_random,
                                                                          R2X_random,
                                                                          R2y_random)

    if distance(R1X, R1y, X_goal, y_goal) < 0.01 or distance(R2X, R2y, X_goal, y_goal) < 0.01:
        print("arrived")
        break

    ax2.clear()
    ax2.scatter(X_goal, y_goal, color='red', label='Goal')
    ax2.scatter(R1X, R1y, color='blue', label='Targeting')
    ax2.scatter(R2X, R2y, color='blue', label='Targeting')

    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    ax2.legend()
    ax2.set_title("Machine View")

    ax1.clear()
    ax1.scatter(X_goal, y_goal, color='red', label='Goal')
    ax1.scatter(R1X_random, R1y_random,
                color='blue', label='Random')
    ax1.scatter(R2X_random,
                R2y_random,
                color='blue', label='Random')

    ax1.set_xlim(-0.05,
                 1.05)
    ax1.set_ylim(-0.05,
                 1.05)

    ax1.legend()
    ax1.set_title("Random View")
    
    plt.pause(0.001)

plt.show()
