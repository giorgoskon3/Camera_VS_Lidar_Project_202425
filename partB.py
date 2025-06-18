import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import open3d as o3d


class RoadSegmenter:
    def __init__(self, img_path, lidar_path, calib_path):
        self.image = self.load_image(img_path)
        self.lidar_points = self.load_lidar_points(lidar_path)
        self.P2, self.R0, self.Tr = self.load_calibration_file(calib_path)

    @staticmethod
    def load_image(path):
        image = cv2.imread(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_lidar_points(path):
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return scan[:, :3]

    @staticmethod
    def load_calibration_file(filepath):
        calib = {}
        with open(filepath) as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    calib[key] = np.array([float(x) for x in value.strip().split()])
        return calib["P2"].reshape(3, 4), calib["R0_rect"].reshape(3, 3), calib["Tr_velo_to_cam"].reshape(3, 4)

    def project_points_to_image(self, points, return_mask=False):
        N = points.shape[0]
        points_hom = np.hstack((points, np.ones((N, 1))))
        cam_points = self.Tr @ points_hom.T
        cam_points = self.R0 @ cam_points[:3, :]
        valid = cam_points[2, :] > 0
        cam_points = cam_points[:, valid]
        proj = self.P2 @ np.vstack((cam_points, np.ones((1, cam_points.shape[1]))))
        proj = proj[:2, :] / proj[2, :]
        return (proj.T.astype(int), valid) if return_mask else proj.T.astype(int)

    @staticmethod
    def compute_saliency(points, k=30):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        tree = o3d.geometry.KDTreeFlann(pcd)
        saliency = np.zeros(len(points))
        for i in range(len(points)):
            _, idxs, _ = tree.search_knn_vector_3d(points[i], k)
            if len(idxs) < 3:
                saliency[i] = 1.0
                continue
            local_pts = points[idxs]
            pca = PCA(n_components=3)
            try:
                pca.fit(local_pts)
                eigvals = pca.explained_variance_
                saliency[i] = eigvals[2] / np.sum(eigvals) if np.sum(eigvals) != 0 else 1.0
            except:
                saliency[i] = 1.0
        return saliency

    @staticmethod
    def filter_ground_points(points, z_range=(-2.2, -1.3), y_limit=10.0, x_range=(0, 25)):
        return points[
            (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) &
            (points[:, 1] > -y_limit) & (points[:, 1] < y_limit) &
            (points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1])
        ]

    def filter_by_saliency(self, points, k=50, percentile=20):
        saliency = self.compute_saliency(points, k=k)
        threshold = np.percentile(saliency, percentile)
        return points[saliency < threshold]

    @staticmethod
    def select_cluster_with_seed(points, labels, seed_point):
        if len(labels) == 0 or np.all(labels == -1):
            return np.empty((0, 3))
        distances = np.linalg.norm(points[:, :2] - np.array(seed_point), axis=1)
        seed_label = labels[np.argmin(distances)]
        return points[labels == seed_label]

    @staticmethod
    def select_pixel_cluster(points, labels, pixel_coords, seed_pixel):
        if len(labels) == 0 or np.all(labels == -1):
            return np.empty((0, 3))
        distances = np.linalg.norm(pixel_coords - np.array(seed_pixel), axis=1)
        seed_label = labels[np.argmin(distances)]
        return points[labels == seed_label]

    def detect_road(self, ground_points, seed_pixel, saliency_k=25, saliency_thresh=30, eps=25.0, min_samples=10):
        road_candidates = self.filter_by_saliency(ground_points, k=saliency_k, percentile=saliency_thresh)
        proj_pixels, valid_mask = self.project_points_to_image(road_candidates, return_mask=True)
        road_candidates = road_candidates[valid_mask]
        proj_pixels = self.project_points_to_image(road_candidates)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(proj_pixels)
        return self.select_pixel_cluster(road_candidates, labels, proj_pixels, seed_pixel)

    @staticmethod
    def detect_obstacles(lidar_points, road_points, z_threshold=0.2):
        x_min, x_max = np.min(road_points[:, 0]), np.max(road_points[:, 0])
        y_min, y_max = np.min(road_points[:, 1]), np.max(road_points[:, 1])
        z_road = np.median(road_points[:, 2])
        mask = (
            (lidar_points[:, 0] >= x_min) & (lidar_points[:, 0] <= x_max) &
            (lidar_points[:, 1] >= y_min) & (lidar_points[:, 1] <= y_max) &
            (lidar_points[:, 2] > z_road + z_threshold)
        )
        return lidar_points[mask]

    @staticmethod
    def estimate_motion_vector(road_points):
        pca = PCA(n_components=3)
        pca.fit(road_points)
        direction = pca.components_[0]
        return direction / np.linalg.norm(direction)

    def draw_obstacle_boxes(self, overlay, obstacles, eps=1.0, min_samples=5):
        if len(obstacles) == 0:
            return overlay
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(obstacles)
        for label in set(labels):
            if label == -1:
                continue
            cluster_pts = obstacles[labels == label]
            proj = self.project_points_to_image(cluster_pts)
            x_min, y_min = np.min(proj, axis=0)
            x_max, y_max = np.max(proj, axis=0)
            cv2.rectangle(overlay, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            cv2.putText(overlay, "Obstacle", (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return overlay

    def visualize_projection(self, image, proj_points, color=(0, 255, 0), alpha=0.2):
        overlay = image.copy()
        h, w = image.shape[:2]

        # Φιλτράρισμα σημείων εντός εικόνας
        proj_points = np.array([pt for pt in proj_points if 0 <= pt[0] < w and 0 <= pt[1] < h])
        if len(proj_points) < 3:
            return overlay  # Δεν υπάρχει αρκετό υλικό για convex hull

        # Υπολογισμός κυρτού περιβλήματος
        hull = cv2.convexHull(proj_points.astype(np.int32))

        # Δημιουργία μάσκας με γεμισμένο κυρτό περίβλημα
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, color)

        # Συνδυασμός με εικόνα (80% εικόνα, 20% χρώμα)
        blended = cv2.addWeighted(mask, alpha, overlay, 1 - alpha, 0)

        return blended


    def run(self, seed_pixel=(580, 300)):
        ground_points = self.filter_ground_points(self.lidar_points)
        road_points = self.detect_road(ground_points, seed_pixel)
        proj_road = self.project_points_to_image(road_points)
        overlay = self.visualize_projection(self.image, proj_road, color=(0, 255, 0))

        obstacles = self.detect_obstacles(self.lidar_points, road_points)
        proj_obstacles = self.project_points_to_image(obstacles)
        overlay = self.visualize_projection(overlay, proj_obstacles, color=(255, 165, 0))
        overlay = self.draw_obstacle_boxes(overlay, obstacles)

        if len(obstacles) > 0:
            h, w = overlay.shape[:2]
            center = (w // 2, h // 2)
            cv2.circle(overlay, center, 30, (0, 0, 255), thickness=3)
            cv2.putText(overlay, "Obstacle Ahead", (center[0] - 50, center[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            direction = self.estimate_motion_vector(road_points)
            origin = np.mean(road_points, axis=0)
            end_point = origin + direction * 5.0
            vector_points = np.vstack([origin, end_point])
            proj_vector = self.project_points_to_image(vector_points)
            cv2.arrowedLine(overlay, tuple(proj_vector[0]), tuple(proj_vector[1]), (0, 0, 255), 4, tipLength=0.2)

        plt.figure(figsize=(12, 6))
        plt.imshow(overlay)
        plt.title("Road and Obstacles Detection")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    segmenter_lidar = RoadSegmenter(
        img_path="image_2/um_000010.png",
        lidar_path="training/velodyne/um_000010.bin",
        calib_path="calib/um_000010.txt"
    )
    segmenter_lidar.run()

