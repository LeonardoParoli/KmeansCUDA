import matplotlib.pyplot as plt

with open('clustered_points.txt', 'r') as file:
    lines = file.readlines()

clusters = {}
centroids = []
for line in lines:
    line = line.strip()
    if line.startswith('Cluster'):
        cluster_num = int(line.split()[1].strip('{'))
        clusters[cluster_num] = []
    elif line.startswith('['):
        centroid = eval(line)
        centroids.append(centroid)
    elif line.startswith('('):
        point = eval(line)
        clusters[cluster_num].append(point)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for cluster_num, points in clusters.items():
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    z_coords = [point[2] for point in points]
    ax.scatter(x_coords, y_coords, z_coords, s=0.05)

centroids_x_coords = [centroid[0] for centroid in centroids]
centroids_y_coords = [centroid[1] for centroid in centroids]
centroids_z_coords = [centroid[2] for centroid in centroids]
ax.scatter(centroids_x_coords, centroids_y_coords, centroids_z_coords,s=25)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Kmeans clustered points and centroids')
plt.show()