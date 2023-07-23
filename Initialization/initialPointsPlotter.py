import matplotlib.pyplot as plt

with open("initial_points.txt", "r") as file:
    lines = file.readlines()
    points = [eval(line.strip()) for line in lines]

x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]
z_coords = [point[2] for point in points]

with open("real_centroids.txt", "r") as file:
    lines = file.readlines()
    additional_points = [eval(line.strip()) for line in lines]
additional_x_coords = [point[0] for point in additional_points]
additional_y_coords = [point[1] for point in additional_points]
additional_z_coords = [point[2] for point in additional_points]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, color='blue', s=0.05)
ax.scatter(additional_x_coords, additional_y_coords, additional_z_coords, color='red', s=25)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Initial points and real clusters')

plt.show()