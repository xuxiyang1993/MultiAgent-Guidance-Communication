import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
speed = 190 / 3.6 / 60

t = 2.33 * np.arange(-2, 2, 0.2)
v = 2.33 * np.arange(-5, 5, 0.2) / 60 + speed

x = []
y = []

for _ in range(100000):
    t = np.random.normal(loc=0, scale=2)
    v = np.random.normal(loc=190 / 3.6, scale=5)
    x.append(v * np.cos(np.radians(t)))
    y.append(v * np.sin(np.radians(t)))

# for i in t:
#     # print(i)
#     x.append(v * np.cos(np.radians(i)))
#     y.append(v * np.sin(np.radians(i)))

f_pointx = (speed + 2.33 * 5 / 60) * np.cos(np.radians(2.33 * 2))
f_pointy = (speed + 2.33 * 5 / 60) * np.sin(np.radians(2.33 * 2))

dist = np.sqrt((speed - f_pointx) ** 2 + f_pointy ** 2)
print(dist)

plt.scatter(x, y, s=0.5)
plt.scatter(speed * 60, 0, c='r')
plt.scatter(0, 0)

# circle = plt.Circle((speed * 60, 0), 12)
# plt.gcf().gca().add_artist(circle)

# plt.scatter(f_pointx, f_pointy, c='r')
plt.xlabel('x-position (m)')
plt.ylabel('y-position (m)')
# plt.ylim([-40, 40])
# plt.legend(['distribution', 'noise'])
plt.show()
