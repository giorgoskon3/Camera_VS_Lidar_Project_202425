import numpy as np
import open3d as o3d

def load_lidar_bin(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # x, y, z

def save_lidar_bin(file_path, points):
    N = points.shape[0]
    points_with_intensity = np.hstack([points, np.zeros((N, 1), dtype=np.float32)])
    points_with_intensity.astype(np.float32).tofile(file_path)

def add_wall(points, wall_x=10, y_range=(-5, 5), z_range=(-1, 2), resolution=0.2):
    y_vals = np.arange(y_range[0], y_range[1], resolution)
    z_vals = np.arange(z_range[0], z_range[1], resolution)
    wall_points = np.array([[wall_x, y, z] for y in y_vals for z in z_vals])
    return np.vstack((points, wall_points))

def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.visualization.draw_geometries([pcd])

def main():
    i = "um_000046"
    original_file = f"training/velodyne/{i}.bin"
    output_file = f"training/velodyne/{i}_with_wall.bin"

    lidar_points = load_lidar_bin(original_file)
    lidar_with_wall = add_wall(lidar_points, wall_x=15, y_range=(-4, 16), z_range=(-1.5, 1.5), resolution=0.2)
    save_lidar_bin(output_file, lidar_with_wall)
    visualize_point_cloud(lidar_with_wall)

if __name__ == "__main__":
    main()
