import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.decomposition import PCA

class RoadSegmenter_CAMERA:
    def __init__(self, path):
        self.path = path
        self.image = None
        self.gray = None
        self.result = None

    def load_image(self):
        self.image = cv2.imread(self.path) # Load the image
        if self.image is None: # Check if the image was loaded successfully
            raise ValueError("Could not load the image.")
    
    def preprocessing(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
        self.gray = clahe.apply(self.gray) # Apply Clahe filter
    
    def show_preprocessed_image(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.gray, cmap='gray')
        plt.title("Preprocessed Image", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()
        
    def region_grow(self):
        height, width = self.gray.shape # Get the dimensions of the grayscale image
        seed_point = (width // 2, height - 30) # Define a seed point for region growing
        # cv2.circle(self.image, seed_point, 5, (0, 255, 255), -1)  # Draw the seed point on the original image

        mask = np.zeros((height + 2, width + 2), np.uint8) # Create a mask for flood fill
        flooded = self.gray.copy()
        
        cutoff_row = int(height * 0.6) # Define a cutoff row for the flood fill
        flooded[:cutoff_row, :] = 0 # Set the upper half of the image to zero

        cv2.floodFill(flooded, mask, seed_point, 125, loDiff=5, upDiff=5, flags=8) # Perform flood fill to segment the road region
        self.region = mask[1:-1, 1:-1] # Remove the border added by flood fill
        
        self.filled_region = cv2.morphologyEx(self.region, cv2.MORPH_CLOSE, np.ones((12, 12), np.uint8)) # Close small holes in the filled region

    def show_region_of_interest(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.filled_region, cmap='gray')
        plt.title("Region Of Interest", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()
    
        
    def colorize_lanes(self, alpha=0.2):
        road_mask = self.filled_region.astype(np.uint8) * 255 # Convert the filled region το 0/255 mask
        height, _ = road_mask.shape # Get the dimensions of the road mask
        midline = np.zeros_like(road_mask) # Create an empty mask for the midline
        left_mask = np.zeros_like(road_mask) # Create an empty mask for the left lane
        right_mask = np.zeros_like(road_mask) # Create an empty mask for the right lane

        mid_x_vals = [] # List to store midline x-coordinates

        for y in range(height): # Iterate through each column of the road mask
            x_coords = np.where(road_mask[y] == 255)[0] # Get the x-coordinates of the road pixels in the current row
            if len(x_coords) > 1:
                x_left = x_coords[0] # Leftmost x-coordinate
                x_right = x_coords[-1] # Rightmost x-coordinate
                x_mid = (x_left + x_right) // 2  # Midpoint x-coordinate
                mid_x_vals.append(x_mid) # Store the midline x-coordinate
            else:
                mid_x_vals.append(-1) # If no road pixels, append -1

        mid_x_vals = np.array(mid_x_vals) # Convert to numpy array for processing
        self.mid_x_vals = mid_x_vals # Store midline x-coordinates in the instance variable
        
        for i, x in enumerate(mid_x_vals):
            if x != -1:
                x = int(x)
                midline[i, x] = 255
                for j in np.where(road_mask[i] == 255)[0]:
                    if j < x:
                        left_mask[i, j] = 1
                    elif j > x:
                        right_mask[i, j] = 1

        # Overlay image
        overlay = np.zeros_like(self.image, dtype=np.uint8)

        # Color the lanes
        overlay[road_mask == 0] = [255, 0, 0]
        overlay[(road_mask == 255) & (left_mask == 1)] = [0, 255, 0]
        overlay[(road_mask == 255) & (right_mask == 1)] = [0, 0, 255]
        overlay[midline == 255] = [255, 255, 255]

        self.result = cv2.addWeighted(self.image, 1 - alpha, overlay, alpha, 0)

    def detect_obstacles(self, model_path="yolov8n.pt", conf_threshold=0.3):
        # Load YOLO model
        model = YOLO(model_path)

        # Detect objects in the image
        results = model(self.image)[0]
        self.obstacles = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf)
            if conf < conf_threshold:
                continue

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.filled_region.shape[1] - 1, x2)
            y2 = min(self.filled_region.shape[0] - 1, y2)

            obj_mask = self.filled_region[y1:y2, x1:x2]
            if np.any(obj_mask > 0):
                self.obstacles.append((x1, y1, x2, y2))

                cv2.rectangle(self.result, (x1, y1), (x2, y2), (0, 255, 255), 2)

        print(f"Found {len(self.obstacles)} obstacles intersecting the road.")


    def compute_motion_vector_pca(self):
        mid_points = []
        for y, x in enumerate(self.mid_x_vals):
            if x != -1:
                mid_points.append([x, y])
        # PCA
        pca = PCA(n_components=2)
        pca.fit(mid_points)
        direction = pca.components_[0]

        if direction[1] > 0:
            direction = -direction

        center = np.mean(mid_points, axis=0).astype(int)
        
        if len(self.obstacles) == 0:
            scale = 100
            end_point = (int(center[0] + scale * direction[0]), int(center[1] + scale * direction[1]))
            cv2.arrowedLine(self.result, tuple(center), end_point, (255, 0, 0), 3, tipLength=0.2)
        else:
            cv2.circle(self.result, tuple(center), 30, (0, 0, 255), thickness=3)
            cv2.putText(self.result, "Obstacle Ahead", (center[0] - 50, center[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        print(f"Midline PCA vector: {direction}")

    def show_original_image(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()

    def show_region_result(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
        plt.title("Road Detection with Region Growing", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()


    def show_final_result(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
        if len(self.obstacles) > 0:
            title = "Obstacles on the road"
        else:
            title = "Motion Vector using PCA"
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()
        
        
    def save_results(self, path):
        # Save Result
        save_dir = "saved_images/partA/road_with_wall"
        os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.basename(path)
        save_path = os.path.join(save_dir, base_name)

        cv2.imwrite(save_path, self.result)
        print(f"Image saved as: {save_path}")

    def run(self):
        # Load Both Left and Right Images and convert to grayscale
        self.load_image()
        self.preprocessing()
        self.show_original_image()
        # a) Using growing region
        self.region_grow()
        self.colorize_lanes()
        self.show_region_result()
        
        # b) Detect obstacles using YOLO
        self.detect_obstacles()
        
        # c) Compute motion vector using PCA
        self.compute_motion_vector_pca()
        self.show_final_result()
    
if __name__ == "__main__":
    path = "road_with_wall/um_000003.png"
    segmenter = RoadSegmenter_CAMERA(path)
    segmenter.run()
    segmenter.save_results(path)
