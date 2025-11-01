import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pycocotools import mask as maskUtils

from utils import DataProcessor

class Visualizer:
    """Handles the visualization of images and their corresponding masks."""

    def __init__(self, image_path_dir: Path, img_fmt: str = "{image_id}.jpg"):
        self.image_path_dir = image_path_dir
        self.img_fmt = img_fmt

    def load_image(self, image_id: int):
        """Loads a single image."""
        img_filename = self.image_path_dir / self.img_fmt.format(image_id=image_id)
        if not img_filename.exists():
            print(f"Warning: Image not found at {img_filename}")
            return None
        return Image.open(img_filename).convert('RGB')

    @staticmethod
    def rle_to_polygon(rle):
        """Converts RLE encoded masks to polygon coordinates."""
        mask = maskUtils.decode(rle).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # A polygon needs at least 3 points
                polygons.append(contour)
        return polygons

    def visualize_image_with_masks(self, image, annotations, fig_title=""):
        """Visualizes an image with its corresponding mask annotations."""
        if image is None or not annotations:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        for ann in annotations:
            rle = ann['segmentation']
            polygons = self.rle_to_polygon(rle)

            for polygon in polygons:
                x_coords = polygon[0::2]
                y_coords = polygon[1::2]
                ax.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'r-', linewidth=2, alpha=0.7)
                ax.fill(x_coords, y_coords, 'red', alpha=0.3)

            x, y, _, _ = ann['bbox']
            ax.text(x, y - 5, f"Score {ann['score']:.2f}", color='blue', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.5, pad=0))

        ax.set_title(fig_title)
        ax.axis('off')
        return fig

def run_visualization(visualizer: Visualizer, processor: DataProcessor, filtered_data, batch_size: int):
    """Executes the visualization logic."""
    print(f"\nStarting visualization for top {batch_size} images...")
    image_groups = processor.group_by_image(filtered_data)

    sorted_images = sorted(image_groups.items(), key=lambda item: len(item[1]), reverse=True)

    if not sorted_images:
        print("No images to visualize.")
        return

    for i, (image_id, annotations) in enumerate(sorted_images[:batch_size]):
        image = visualizer.load_image(image_id)
        if image:
            annotations_sorted = sorted(annotations, key=lambda x: x['score'], reverse=True)
            top_score = annotations_sorted[0]['score']

            fig = visualizer.visualize_image_with_masks(
                image,
                annotations_sorted,
                fig_title=f"Image {image_id} ({len(annotations)} detections) | Top Score: {top_score:.3f}"
            )
            if fig:
                plt.show()

def main(args):
    """Main function to run the visualization script."""
    processor = DataProcessor(threshold=args.threshold)
    processor.load_annotations(Path(args.input))
    processor.print_statistics(processor.annotations, args.top_k, title="Original Data")

    if args.statistics:
        print("\nStatistics mode finished.")
        return

    processor.filter_by_score()

    if args.output:
        processor.save_annotations(Path(args.output), processor.filtered_annotations)

    processor.print_statistics(processor.filtered_annotations, args.top_k, title="Filtered Data")

    if args.visualize:
        if not args.image_dir:
            raise ValueError("--image_dir is required for visualization.")
        visualizer = Visualizer(Path(args.image_dir), args.img_fmt)
        run_visualization(visualizer, processor, processor.filtered_annotations, args.batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and visualize annotation data.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path.")
    parser.add_argument("--output", type=str, help="Output JSON file path.")
    parser.add_argument("--image_dir", type=str, help="Image directory path for visualization.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Score threshold for filtering.")
    parser.add_argument("--img_fmt", type=str, default="{image_id}.jpg", help="Image filename format.")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization.")
    parser.add_argument("--batch", type=int, default=4, help="Number of images to visualize.")
    parser.add_argument("--top_k", type=int, default=5, help="Top k scores for statistics.")
    parser.add_argument("--statistics", action="store_true", help="Run in statistics-only mode.")

    args = parser.parse_args()
    main(args)
