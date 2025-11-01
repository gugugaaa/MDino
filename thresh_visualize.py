import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
from collections import Counter
import cv2


class DataProcessor:
    """处理JSON标注数据的类"""
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.annotations = []  # 原始标注
        self.filtered_annotations = [] # 过滤后的标注
        self.image_stats = {} # 过滤后，每张图片包含的检测数量
    
    def load_annotations(self, json_path):
        """加载JSON标注文件"""
        print(f"正在从 {json_path} 加载标注...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        print(f"加载完成，共 {len(self.annotations)} 条标注。")
    
    def filter_by_score(self):
        """
        根据阈值过滤 self.annotations
        """
        if not self.annotations:
            print("未加载标注，请先调用 load_annotations()")
            return

        print(f"正在以阈值 {self.threshold} 进行过滤...")
        self.filtered_annotations = []
        self.image_stats = {}
        
        for item in self.annotations:
            if item.get('score', 0) >= self.threshold:
                self.filtered_annotations.append(item)
                image_id = item.get('image_id')
                self.image_stats[image_id] = self.image_stats.get(image_id, 0) + 1
        
        print(f"过滤完成，剩余 {len(self.filtered_annotations)} 条标注。")
    
    def save_annotations(self, output_path, data):
        """保存处理后的JSON文件"""
        print(f"正在将数据保存到 {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("保存完毕。")
    
    def print_statistics(self, data, top_k=5, title=""):
        """打印得分统计"""
        if not data:
            print(f"{title} 统计: 没有数据。")
            return
        
        scores = [ann.get('score', 0) for ann in data]
        
        # 按得分排序，取top_k进行统计
        sorted_data = sorted(data, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
        top_scores = [ann.get('score', 0) for ann in sorted_data]
        
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        top_avg_score = np.mean(top_scores) if top_scores else 0.0
        
        print(f"--- {title} 统计 (共 {len(scores)} 条) ---")
        print(f"{'平均得分':<12} {'最高分':<12} {'最低分':<12} {'Top{top_k}平均分':<12}")
        print(f"{avg_score:<12.4f} {max_score:<12.4f} {min_score:<12.4f} {top_avg_score:<12.4f}")
        print("-" * (12 * 4 + 3))
    
    def group_by_image(self, annotations):
        """按图片ID分组标注"""
        image_groups = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in image_groups:
                image_groups[img_id] = []
            image_groups[img_id].append(ann)
        return image_groups


class Visualizer:
    """可视化图像和掩码的类"""
    
    def __init__(self, image_path_dir, img_fmt="{image_id}.jpg"):
        self.image_path_dir = image_path_dir
        self.img_fmt = img_fmt
    
    def load_image(self, image_id):
        """加载单张图片"""
        img_filename = os.path.join(self.image_path_dir, self.img_fmt.format(image_id=image_id))
        if not os.path.exists(img_filename):
            print(f"警告：未找到图片 {img_filename}")
            return None
        return Image.open(img_filename).convert('RGB')
    
    @staticmethod
    def rle_to_polygon(rle):
        """
        将RLE编码转换为多边形坐标
        Args:
            rle: RLE编码的mask
        Returns:
            多边形坐标列表 [[x1, y1, x2, y2, ...], ...]
        """
        mask = maskUtils.decode(rle).astype(np.uint8)
        # 查找轮廓
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        polygons = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # 至少3个点
                polygons.append(contour)
        return polygons
    
    def visualize_image_with_masks(self, image, annotations, fig_title=""):
        """
        可视化图像和对应的掩码标注
        Args:
            image: PIL Image对象
            annotations: 标注列表
            fig_title: 图表标题
        """
        if image is None or not annotations:
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8)) # 稍微调大画布
        ax.imshow(image)
        
        # 叠加所有mask
        for ann in annotations:
            rle = ann['segmentation']
            # 使用静态方法调用
            polygons = Visualizer.rle_to_polygon(rle)
            
            # 绘制多边形
            for polygon in polygons:
                x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
                y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
                ax.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'r-', linewidth=2, alpha=0.7)
                ax.fill(x_coords, y_coords, 'red', alpha=0.3)
            
            # 画得分
            x, y, w, h = ann['bbox']
            ax.text(x, y-5, f"Score {ann['score']:.2f}", color='blue', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.5, pad=0))
        
        ax.set_title(fig_title)
        ax.axis('off')
        return fig


def run_visualization(visualizer, processor, filtered_data, batch_size):
    """执行可视化逻辑"""
    print(f"\n开始可视化，批次大小 {batch_size}...")
    image_groups = processor.group_by_image(filtered_data)
    
    # 按图像中的检测数量排序，可视化检测数最多的图像
    sorted_images = sorted(image_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    if not sorted_images:
        print("没有可供可视化的图像。")
        return
        
    print(f"将显示 {min(len(sorted_images), batch_size)} 张检测数量最多的图像。")
    
    for idx, (image_id, annotations) in enumerate(sorted_images[:batch_size]):
        image = visualizer.load_image(image_id)
        if image is not None:
            # 按得分排序，确保图例标题显示最高分
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
    """主执行函数"""
    
    # 1. 初始化处理器
    processor = DataProcessor(threshold=args.threshold)
    
    # 2. 加载数据
    processor.load_annotations(args.input)
    
    # 3. 打印原始数据统计
    processor.print_statistics(processor.annotations, args.top_k, title="原始数据")
    
    # 4. 如果是仅统计模式，则退出
    if args.statistics:
        print("\n仅进行统计模式，已完成。")
        return
    
    # 5. 过滤数据
    processor.filter_by_score()
    
    # 6. 保存过滤后的数据
    processor.save_annotations(args.output, processor.filtered_annotations)
    
    # 7. 打印过滤后的数据统计
    processor.print_statistics(processor.filtered_annotations, args.top_k, title="过滤后数据")
    
    # 8. 可视化
    if args.visualize:
        visualizer = Visualizer(args.image_dir, args.img_fmt)
        run_visualization(visualizer, processor, processor.filtered_annotations, args.batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理和可视化标注数据")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output", type=str, help="输出JSON文件路径 (如果非仅统计模式，则为必需)")
    parser.add_argument("--image_dir", type=str, help="图像目录路径 (如果可视化，则为必需)")
    parser.add_argument("--threshold", type=float, default=0.8, help="得分阈值")
    parser.add_argument("--img_fmt", type=str, default="{image_id}.jpg", help="图像文件名格式")
    parser.add_argument("--visualize", action="store_true", help="是否可视化结果")
    parser.add_argument("--batch", type=int, default=4, help="一次可视化的图像批次数量")
    parser.add_argument("--top_k", type=int, default=5, help="统计时使用的top_k值")
    parser.add_argument("--statistics", action="store_true", help="仅进行统计，不进行其他操作")
    
    args = parser.parse_args()
    
    # 简单的参数校验
    if not args.statistics:
        if not args.output:
            parser.error("--output 是必需的 (除非使用 --statistics)")
        if args.visualize and not args.image_dir:
            parser.error("--image_dir 是必需的 (当使用 --visualize 时)")
            
    main(args)