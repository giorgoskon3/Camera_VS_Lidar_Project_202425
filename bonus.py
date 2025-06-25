import imageio.v2 as imageio
import cv2
import os

def create_gif_from_images(image_paths, output_path="road_animation.gif", fps=1):
    frames = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Image not found: {path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)
    
    if frames:
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"GIF saved as: {output_path}")
    else:
        print("No image loaded.")


if __name__ == "__main__":
    image_paths_camera = [
        "saved_images/partA/road_with_walls/um_000000.png",
        "saved_images/partA/road_with_walls/um_000001.png",
        "saved_images/partA/road_with_walls/um_000002.png",
        "saved_images/partA/road_with_walls/um_000003.png"
    ]
    
    image_paths_lidar = [
        "saved_images/partA/road_with_walls/um_000000.png",
        "saved_images/partA/road_with_walls/um_000001.png",
        "saved_images/partA/road_with_walls/um_000002.png",
        "saved_images/partA/road_with_walls/um_000003.png"
    ]

    # Create GIF from the specified images - CAMERA
    create_gif_from_images(image_paths_camera, output_path="road_animation_with_walls.gif", fps=1.0)

    # Create GIF from the specified images - LIDAR
    create_gif_from_images(image_paths_lidar, output_path="road_animation_with_walls.gif", fps=1.0)