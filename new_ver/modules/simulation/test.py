from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# plt.ion()

# plt = plt

fig = plt.figure(figsize=(12, 9))
ax = Axes3D(fig)

fig.suptitle("3DoF Arm")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# print(f'{arm.base=}')
ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
ax.set_zlim([-1, 2])

x, = ax.plot([1],[2],[1],'ro')

# while True:
#     a=1
    # plt.pause(0.01)

plt.show()
