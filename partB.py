import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from collections import Counter
import open3d as o3d


def load_lidar_points(path):
    scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]


def load_calibration_file(filepath):
    calib = {}
    with open(filepath) as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                calib[key] = np.array([float(x) for x in value.strip().split()])
    P2 = calib["P2"].reshape(3, 4)
    R0 = calib["R0_rect"].reshape(3, 3)
    Tr = calib["Tr_velo_to_cam"].reshape(3, 4)
    return P2, R0, Tr


def project_points_to_image(points, P2, R0, Tr, return_mask=False):
    N = points.shape[0]
    points_hom = np.hstack((points, np.ones((N, 1))))
    cam_points = Tr @ points_hom.T
    cam_points = R0 @ cam_points[:3, :]
    valid = cam_points[2, :] > 0
    cam_points = cam_points[:, valid]
    proj = P2 @ np.vstack((cam_points, np.ones((1, cam_points.shape[1]))))
    proj = proj[:2, :] / proj[2, :]

    if return_mask:
        return proj.T.astype(int), valid
    else:
        return proj.T.astype(int)


def compute_saliency(points, k=30):
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points)
    )  # Convert points to Open3D PointCloud
    tree = o3d.geometry.KDTreeFlann(
        pcd
    )  # Create KD-Tree for efficient nearest neighbor search
    saliency = np.zeros(len(points))  # Initialize saliency scores with zeros
    for i in range(len(points)):  # Iterate through each point
        _, idxs, _ = tree.search_knn_vector_3d(points[i], k)  # Find k nearest neighbors
        if len(idxs) < 3:  # If fewer than 3 neighbors, assign high saliency
            saliency[i] = 1.0
            continue
        local_pts = points[idxs]  # Get local points around the current point
        pca = PCA(n_components=3)  # Initialize PCA for dimensionality reduction
        try:
            pca.fit(local_pts)  # Fit PCA to local points
            eigvals = (
                pca.explained_variance_
            )  # Get the explained variance (eigenvalues)
            saliency[i] = (
                eigvals[2] / np.sum(eigvals) if np.sum(eigvals) != 0 else 1.0
            )  # Normalize saliency by the sum of eigenvalues
        except:
            saliency[i] = 1.0
    return saliency


def filter_ground_points(points, z_range=(-2.2, -1.3), y_limit=10.0, x_range=(0, 25)):
    return points[
        (points[:, 0] > x_range[0])
        & (
            points[:, 0] < x_range[1]
        )  # Filter points based on x-coordinate (x_start is the minimum x value)
        & (points[:, 1] > -y_limit)
        & (
            points[:, 1] < y_limit
        )  # Filter points based on y-coordinate (y_limit is the maximum y value)
        & (points[:, 2] > z_range[0])
        & (
            points[:, 2] < z_range[1]
        )  # Filter points based on z-coordinate (z_range defines the vertical range)
    ]


def filter_by_saliency(points, k=50, percentile=20):
    saliency = compute_saliency(points, k=k)  # Compute saliency scores for the points
    threshold = np.percentile(
        saliency, percentile
    )  # Determine the threshold for filtering based on the specified percentile
    return points[
        saliency < threshold
    ]  # Return points with saliency below the threshold


def select_cluster_with_seed(points, labels, seed_point):
    if len(labels) == 0 or np.all(labels == -1):
        return np.empty((0, 3))  # Return empty array if no clusters found

    distances = np.linalg.norm(
        points[:, :2] - np.array(seed_point), axis=1
    )  # Compute distances from the seed point to all points
    seed_idx = np.argmin(
        distances
    )  # Find the index of the point closest to the seed point
    seed_label = labels[
        seed_idx
    ]  # Get the label of the closest point to the seed point
    return points[
        labels == seed_label
    ]  # Return all points that belong to the same cluster as the seed point


def detect_road_with_saliency(
    filtered_points,
    P2,
    R0,
    Tr,
    saliency_k=25,
    saliency_thresh=30,
    eps=20.0,
    min_samples=10,
    seed_pixel=(550, 320),
):
    # Filter points based on saliency
    road_candidates = filter_by_saliency(
        filtered_points, k=saliency_k, percentile=saliency_thresh
    )

    # Extract a valid mask for the projected points
    proj_pixels, valid_mask = project_points_to_image(
        road_candidates, P2, R0, Tr, return_mask=True
    )

    # Take valid road candidates only
    road_candidates = road_candidates[valid_mask]
    # Project the valid points to image coordinates
    proj_pixels = project_points_to_image(road_candidates, P2, R0, Tr)

    # Perform DBSCAN clustering on the projected pixels
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Extact DBSCAN's labels
    labels = dbscan.fit_predict(proj_pixels)

    # Return the cluster that contains the seed pixel
    return select_pixel_cluster(road_candidates, labels, proj_pixels, seed_pixel)


def select_pixel_cluster(points, labels, pixel_coords, seed_pixel):
    if len(labels) == 0 or np.all(labels == -1):
        return np.empty((0, 3))

    # Χρησιμοποίησε μόνο τις 2 πρώτες στήλες των pixels (u, v)
    distances = np.linalg.norm(pixel_coords - np.array(seed_pixel), axis=1)
    seed_idx = np.argmin(distances)
    seed_label = labels[seed_idx]
    return points[labels == seed_label]


def visualize_projection(image, proj_points, color=(0, 255, 0), alpha=0.6):
    overlay = image.copy()
    h, w = image.shape[:2]
    for x, y in proj_points:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), 2, color, -1)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def detect_obstacles_on_road(lidar_points, road_points, z_threshold=0.2):
    x_min, x_max = np.min(road_points[:, 0]), np.max(road_points[:, 0])
    y_min, y_max = np.min(road_points[:, 1]), np.max(road_points[:, 1])
    z_road = np.median(road_points[:, 2])
    mask = (
        (lidar_points[:, 0] >= x_min)
        & (lidar_points[:, 0] <= x_max)
        & (lidar_points[:, 1] >= y_min)
        & (lidar_points[:, 1] <= y_max)
        & (lidar_points[:, 2] > z_road + z_threshold)
    )
    # Return the points that are obstacles on the road
    return lidar_points[mask]


def estimate_motion_vector(road_points):
    pca = PCA(n_components=3)
    pca.fit(road_points)
    direction = pca.components_[0]
    return direction / np.linalg.norm(direction)


def draw_obstacle_boxes(overlay, obstacles, P2, R0, Tr, eps=1.0, min_samples=5):
    if len(obstacles) == 0:
        return overlay

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(obstacles)
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue

        cluster_pts = obstacles[labels == label]
        proj = project_points_to_image(cluster_pts, P2, R0, Tr)

        x_min, y_min = np.min(proj, axis=0)
        x_max, y_max = np.max(proj, axis=0)

        top_left = (int(x_min), int(y_min))
        bottom_right = (int(x_max), int(y_max))

        cv2.rectangle(overlay, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(overlay, "Obstacle", (top_left[0], top_left[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return overlay


def main():
    # Define paths to the image, LiDAR data, and calibration file
    img_path = "image_2/um_000000.png"
    lidar_path = "training/velodyne/um_000000_with_wall.bin"
    calib_path = "calib/um_000000.txt"

    image = cv2.imread(img_path)  # BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    lidar_points = load_lidar_points(lidar_path)  # Load LiDAR points from binary file
    P2, R0, Tr = load_calibration_file(
        calib_path
    )  # Load camera calibration parameters (P2, R0, Tr)

    ground_points = filter_ground_points(
        lidar_points
    )  # Filter points that are likely to be on the ground
    road_points = detect_road_with_saliency(
        ground_points, P2, R0, Tr, seed_pixel=(580, 300)
    )  # Detect road points using saliency and clustering

    proj_road = project_points_to_image(
        road_points, P2, R0, Tr
    )  # Project road points to image coordinates using camera calibration parameters
    overlay = visualize_projection(
        image, proj_road, color=(0, 255, 0)
    )  # Visualize the projected road points on the image

    obstacles = detect_obstacles_on_road(
        lidar_points, road_points
    )  # Detect obstacles on the road based on LiDAR points and road points
    
    proj_obstacles = project_points_to_image(
        obstacles, P2, R0, Tr
    )  # Project detected obstacles to image coordinates
    
    overlay = visualize_projection(
        overlay, proj_obstacles, color=(255, 165, 0)
    )  # Visualize the projected obstacles on the image
    
    overlay = draw_obstacle_boxes(overlay, obstacles, P2, R0, Tr)
    
    direction = estimate_motion_vector(road_points)

    if len(obstacles) > 0:
        h, w = overlay.shape[:2]
        center = (w // 2, h // 2)

        # Σχεδίαση κόκκινου κύκλου στο κέντρο
        cv2.circle(overlay, center, 30, (0, 0, 255), thickness=3)

        # Προειδοποιητικό μήνυμα πάνω από τον κύκλο
        cv2.putText(
            overlay,
            "Obstacle Ahead",
            (center[0] - 50, center[1] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        # Οπτικοποίηση του διανύσματος κατεύθυνσης
        origin = np.mean(road_points, axis=0)
        end_point = origin + direction * 5.0  # π.χ. 5 μέτρα εμπρός
        vector_points = np.vstack([origin, end_point])
        proj_vector = project_points_to_image(vector_points, P2, R0, Tr)

        pt1 = tuple(proj_vector[0])
        pt2 = tuple(proj_vector[1])
        cv2.arrowedLine(overlay, pt1, pt2, (0, 0, 255), 4, tipLength=0.2)


    plt.figure(figsize=(12, 6))
    plt.imshow(overlay)
    plt.title("Road and Obstacles Detection using LiDAR and Camera Data")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
