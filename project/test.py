import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

traj = np.load("trajectory.npy")
x = traj[:, 0]
y = traj[:, 1]

plt.figure(figsize=(12, 7))
plt.subplot(2,1,1)
plt.plot(x, color='#2E86AB', linewidth=1.5)
plt.title('X 坐标')
plt.ylabel('像素')

plt.subplot(2,1,2)
plt.plot(y, color='#E84855', linewidth=1.5)
plt.title('Y 坐标')
plt.ylabel('像素')
plt.xlabel('帧')

plt.tight_layout()
plt.savefig('check_xy.png', dpi=150)
plt.show()