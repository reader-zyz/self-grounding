from lang_sam import LangSAM
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import find_objects, center_of_mass
import cv2

def visualize(image_pil, masks, boxes, mode=["blend_masks", "blend_boxes", "number_masks", "outline_masks"], output_path="visualized_image.png"):
    """
    可视化函数：将 masks 和 boxes blend 到原图像上，并保存结果。
    
    参数:
        image_pil (PIL.Image): 原始图像。
        masks (np.ndarray): N x H x W 的 masks 数组。
        boxes (np.ndarray): N x 4 的 boxes 数组，每行为 (x1, y1, x2, y2)。
        mode (str): 模式，可选 "blend_masks", "blend_boxes", "blend_masks_boxes", "number_masks", "outline_masks"。
        output_path (str): 保存路径。
    """
    # 转换原图像为 RGBA 格式（支持透明度）
    image = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))  # 创建透明图层
    
    if "blend_masks" in mode:
        # 遍历每个 mask，生成伪随机颜色并 blend 到 overlay 上
        for mask in masks:
            # 随机生成伪随机颜色
            color = tuple([random.randint(0, 255) for _ in range(3)] + [int(255 * 0.3)])  # 0.3 不透明度
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # 转换为 8 位灰度图
            mask_colored = Image.new("RGBA", image.size, color)  # 创建彩色 mask
            overlay.paste(mask_colored, (0, 0), mask_image)  # 使用 mask 作为 alpha 通道进行叠加
    
    if "blend_boxes" in mode:
        # 在 overlay 上绘制 boxes
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=2)  # 绿色边框，宽度为 2
    
    if "number_masks" in mode:
        # 在每个 mask 的中心添加编号
        draw = ImageDraw.Draw(overlay)
        for i, mask in enumerate(masks):
            # 计算 mask 的中心
            center_y, center_x = center_of_mass(mask)
            text = str(i + 1)
            font_size = 20
            # font = ImageFont.truetype("arial.ttf", font_size)  # 替换为系统可用字体
            try:
                font = ImageFont.truetype("arial.ttf", 20)  # 替换为具体字体路径
            except OSError:
                print("Custom font not found, using default font.")
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 添加黑色底的白色数字
            rect_start = (int(center_x - text_width / 2) - 2, int(center_y - text_height / 2) - 2)
            rect_end = (int(center_x + text_width / 2) + 2, int(center_y + text_height / 2) + 2)
            draw.rectangle([rect_start, rect_end], fill=(0, 0, 0, 255))
            draw.text((center_x - text_width / 2, center_y - text_height / 2), text, fill=(255, 255, 255, 255), font=font)
    
    if "outline_masks" in mode:
        # 绘制 mask 的边界
        draw = ImageDraw.Draw(overlay)
        for mask in masks:
            # 使用 OpenCV 提取轮廓
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # 转换轮廓为坐标列表
                points = [(int(point[0][0]), int(point[0][1])) for point in contour]
                # 绘制边界
                draw.line(points + [points[0]], fill=(0, 0, 0, 255), width=3)
    
    # 将 overlay 和原图像 blend
    blended = Image.alpha_composite(image, overlay)
    
    # 转换为 RGB 格式（去除 alpha 通道）并保存
    blended = blended.convert("RGB")
    blended.save(output_path)
    print(f"Visualized image saved to {output_path}")

if __name__ == '__main__':
    ckpt_path = '/root/autodl-tmp/weight'
    model = LangSAM(ckpt_path=ckpt_path)
    print('init finished')
    
    image_pil = Image.open("./tidy_scene_1.jpg").convert("RGB")
    text_prompt = "can. brown liquid."
    results = model.predict([image_pil], [text_prompt]) # results->list[dict{'scores', 'labels', 'boxes', 'masks', 'masks_score'}]

    # print(results)

    # 示例调用
    masks = results[0]['masks']
    boxes = results[0]['boxes']
    visualize(image_pil, masks, boxes, mode=["blend_masks", "blend_boxes", "number_masks", "outline_masks"], output_path="result.png")




