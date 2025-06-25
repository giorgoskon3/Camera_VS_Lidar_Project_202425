import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree


class RoadSegmenter:
    def __init__(self, img_path, lidar_path, calib_path):
        self.image = self.load_image(img_path)
        self.lidar_points = self.load_lidar_points(lidar_path)
        self.road_points = np.empty((0, 3))
        self.P2, self.R0, self.Tr = self.load_calibration_file(calib_path)

    def load_image(self, path):
        image = cv2.imread(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_lidar_points(self, path):
        scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return scan[:, :3]

    def load_calibration_file(self, filepath):
        calib = {}
        with open(filepath) as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    calib[key] = np.array([float(x) for x in value.strip().split()])
        return (
            calib["P2"].reshape(3, 4),
            calib["R0_rect"].reshape(3, 3),
            calib["Tr_velo_to_cam"].reshape(3, 4),
        )

    def project_points_to_image(self, points, return_mask=False):
        N = points.shape[0]
        points_hom = np.hstack((points, np.ones((N, 1))))
        cam_points = self.Tr @ points_hom.T
        cam_points = self.R0 @ cam_points[:3, :]
        valid = cam_points[2, :] > 0
        cam_points = cam_points[:, valid]
        proj = self.P2 @ np.vstack((cam_points, np.ones((1, cam_points.shape[1]))))
        proj = proj[:2, :] / proj[2, :]
        if return_mask:
            return proj.T.astype(int), valid
        else:
            return proj.T.astype(int)

    @staticmethod
    def compute_saliency(points, k=30):
        saliency = np.zeros(len(points)) # Initialize saliency array
        tree = KDTree(points) # Build KDTree for fast nearest neighbor search

        for i in range(len(points)): # Iterate through each point
            idxs = tree.query(points[i : i + 1], k=k, return_distance=False)[0] # Get k nearest neighbors
            local_pts = points[idxs] # Local points around the current point

            local_pts_centered = local_pts - np.mean(local_pts, axis=0) # Center the local points
            try: # Perform Singular Value Decomposition (SVD)
                _, s, _ = np.linalg.svd(local_pts_centered, full_matrices=False)
                saliency[i] = s[-1] / np.sum(s) if np.sum(s) != 0 else 1.0 # Normalize the smallest singular value
            except:
                saliency[i] = 1.0 # If SVD fails, assign maximum saliency

        return saliency

    @staticmethod
    def filter_ground_points(
        points, z_range=(-2.2, -1.3), y_limit=10.0, x_range=(0, 25)
    ):
        return points[
            (points[:, 0] > x_range[0])
            & (points[:, 0] < x_range[1])
            & (points[:, 1] > -y_limit)
            & (points[:, 1] < y_limit)
            & (points[:, 2] > z_range[0])
            & (points[:, 2] < z_range[1])
        ] # Filter points based on x, y, and z limits

    def filter_by_saliency(self, points, saliency, percentile=10):
        threshold = np.percentile(saliency, percentile) # Calculate the saliency threshold
        return points[saliency < threshold] # Filter points below the saliency threshold

    @staticmethod
    def select_pixel_cluster(points, labels, pixel_coords, seed_pixel):
        if len(labels) == 0 or np.all(labels == -1):
            return np.empty((0, 3))
        distances = np.linalg.norm(pixel_coords - np.array(seed_pixel), axis=1)
        seed_label = labels[np.argmin(distances)]
        return points[labels == seed_label]

    def detect_road(
        self,
        ground_points,
        seed_pixel,
        saliency,
        saliency_thresh=25,
        eps=15.0,
        min_samples=10,
    ):
        road_candidates = self.filter_by_saliency(
            ground_points, saliency, percentile=saliency_thresh
        )
        proj_pixels, valid_mask = self.project_points_to_image(
            road_candidates, return_mask=True
        )
        road_candidates = road_candidates[valid_mask]
        proj_pixels = self.project_points_to_image(road_candidates)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(proj_pixels)
        return self.select_pixel_cluster(
            road_candidates, labels, proj_pixels, seed_pixel
        )

    @staticmethod
    def estimate_motion_vector(road_points):
        pca = PCA(n_components=3)
        pca.fit(road_points)
        direction = pca.components_[0]
        return direction / np.linalg.norm(direction)

    def visualize_projection(self, image, proj_points, color=(0, 255, 0), alpha=0.2):
        overlay = image.copy()
        h, w = image.shape[:2]

        proj_points = np.array(
            [pt for pt in proj_points if 0 <= pt[0] < w and 0 <= pt[1] < h]
        )
        if len(proj_points) < 3:
            return overlay

        mask = np.zeros((h, w), dtype=np.uint8)
        for x, y in proj_points.astype(np.int32):
            cv2.circle(mask, (x, y), radius=5, color=255, thickness=-1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = [
            cv2.approxPolyDP(cnt, epsilon=2.0, closed=True) for cnt in contours
        ]

        filled_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(
            filled_mask, simplified_contours, -1, color, thickness=cv2.FILLED
        )

        blended = cv2.addWeighted(filled_mask, alpha, overlay, 1, 0)

        self.road_mask = mask

        return blended

    def points_within_road(self, lidar_points, road_points, margin=0.5):
        x_min, x_max = np.min(road_points[:, 0]), np.max(road_points[:, 0])
        y_min, y_max = np.min(road_points[:, 1]), np.max(road_points[:, 1])

        return lidar_points[
            (lidar_points[:, 0] >= x_min - margin) &
            (lidar_points[:, 0] <= x_max + margin) &
            (lidar_points[:, 1] >= y_min - margin) &
            (lidar_points[:, 1] <= y_max + margin)
        ]


    def detect_obstacles(self, lidar_points, road_points,
                                xy_margin=0.1, z_threshold=0.2):
        if len(road_points) == 0:
            return np.empty((0, 3))

        x_min, x_max = np.min(road_points[:, 0]) - xy_margin, np.max(road_points[:, 0]) + xy_margin
        y_min, y_max = np.min(road_points[:, 1]) - xy_margin, np.max(road_points[:, 1]) + xy_margin

        in_road_xy = lidar_points[
            (lidar_points[:, 0] >= x_min) & (lidar_points[:, 0] <= x_max) &
            (lidar_points[:, 1] >= y_min) & (lidar_points[:, 1] <= y_max)
        ]

        if len(in_road_xy) == 0:
            return np.empty((0, 3))

        road_z_mean = np.mean(road_points[:, 2])

        mask = in_road_xy[:, 2] > road_z_mean + z_threshold
        return in_road_xy[mask]
    
    
    def draw_obstacles(self, image, obstacles, color=(255, 0, 0)):
        proj, valid = self.project_points_to_image(obstacles, return_mask=True)
        proj = proj[valid]
        for pt in proj:
            cv2.circle(image, tuple(pt), 3, color, -1)
        return image

    def run_road_segmentation(self, seed_pixel=(580, 300)):
        ground_points = self.filter_ground_points(self.lidar_points) # Filter ground points
        saliency = self.compute_saliency(ground_points, k=20) # Compute saliency of points
        self.road_points = self.detect_road(ground_points, seed_pixel, saliency) # Detect road points
        proj_road = self.project_points_to_image(self.road_points) # Project road points to image
        self.overlay = self.visualize_projection(self.image.copy(), proj_road, color=(0, 255, 0)) # Visualize road points on the image

    def run_obstacle_detection(self, xy_margin=0.1, z_threshold=0.3):
        obstacle_candidates = self.detect_obstacles(self.lidar_points, self.road_points,
                                                    xy_margin=xy_margin, z_threshold=z_threshold)

        if len(obstacle_candidates) > 0:
            labels = DBSCAN(eps=1.0, min_samples=10).fit_predict(obstacle_candidates[:, :3])
            self.obstacles = obstacle_candidates[labels != -1]
        else:
            self.obstacles = np.empty((0, 3))

        self.overlay = self.draw_obstacles(self.overlay, self.obstacles)

    def run_motion_estimation(self):
        if self.overlay is None:
            raise RuntimeError("Overlay image has not been generated.")

        if len(self.obstacles) > 0:
            h, w = self.overlay.shape[:2]
            center = (w // 2, h // 2)
            cv2.circle(self.overlay, center, 30, (0, 0, 255), thickness=3)
            cv2.putText(
                self.overlay,
                "Obstacle Ahead",
                (center[0] - 50, center[1] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        else:
            direction = self.estimate_motion_vector(self.road_points)
            origin = np.mean(self.road_points, axis=0)
            end_point = origin + direction * 5.0
            vector_points = np.vstack([origin, end_point])
            proj_vector = self.project_points_to_image(vector_points)
            cv2.arrowedLine(
                self.overlay,
                tuple(proj_vector[0]),
                tuple(proj_vector[1]),
                (0, 0, 255),
                4,
                tipLength=0.2,
            )

    def show_overlay(self, title="Results"):
        if self.overlay is None:
            print("No overlay image exists.")
            return
        plt.figure(figsize=(12, 6))
        plt.imshow(self.overlay)
        plt.title(title)
        plt.axis("off")
        plt.show()


    def save_results(self, path):
        save_dir = "saved_images/partB"
        os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.basename(path)
        save_path = os.path.join(save_dir, base_name)
        
        bgr_overlay = cv2.cvtColor(self.overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr_overlay)
        print(f"Image saved as: {save_path}")

if __name__ == "__main__":
    i = "um_000047"
    j = f"{i}_with_wall"
    path=f"image_2/{i}.png"
    segmenter_lidar = RoadSegmenter(
        img_path=f"image_2/{i}.png",
        lidar_path=f"training/velodyne/{i}.bin",
        calib_path=f"calib/{i}.txt",
    )
    segmenter_lidar.run_road_segmentation(seed_pixel=(580, 300))
    segmenter_lidar.show_overlay("Road Detection")
        
    segmenter_lidar.run_obstacle_detection()
    segmenter_lidar.show_overlay("Obstacle Detection")
    
    segmenter_lidar.run_motion_estimation()
    segmenter_lidar.show_overlay("Final Result")

    segmenter_lidar.save_results(path)
