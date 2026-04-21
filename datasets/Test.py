import os
import cv2
from pycocotools.coco import COCO

# Ultralytics 颜色映射类
class Colors:
    def __init__(self):
        hexs = ('FF3838', '00D4BB', 'FF701F', 'FF37C7', '00C2FF', 'FF95C8')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# 实例化颜色类
colors = Colors()

# 设置 COCO JSON 文件和图片目录
json_path = r"E:\Underwater_Image_Dataset\19.目标检测\RUOD\RUOD数据集\RUOD_ANN/instances_test.json"
image_dir = r"E:\Underwater_Image_Dataset\19.目标检测\RUOD\RUOD数据集\RUOD_pic\Images\test/"
output_dir = r"E:\1.GoodLuck\1.DiffColor\4.TMM\实验结果\UROP2020\Detection\Reference/"

# 读取 COCO 标注文件
coco = COCO(json_path)

# 获取所有类别
categories = coco.loadCats(coco.getCatIds())
category_dict = {cat["id"]: cat["name"] for cat in categories}

# 处理所有图片
for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(image_dir, img_info["file_name"])

    # 读取图片
    image = cv2.imread(img_path)
    if image is None:
        print(f"无法读取图片: {img_path}")
        continue

    # 获取该图片的所有标注
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # 绘制所有检测框
    for ann in annotations:
        bbox = ann["bbox"]  # COCO 格式: [x, y, width, height]
        category_id = ann["category_id"]
        category_name = category_dict.get(category_id, "Unknown")

        # 获取固定颜色
        box_color = colors(category_id, bgr=True)  # BGR 格式
        box_thickness = 4  # 框粗细

        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), box_color, box_thickness)

        # 字体设置
        font_scale = 1.2  # 字体大小
        font_thickness = 4  # 字体粗细

        # 获取文本大小
        (text_w, text_h), _ = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # 画文本背景（提高可读性）
        cv2.rectangle(image, (x, y - text_h - 5), (x + text_w, y), box_color, -1)

        # 绘制类别名称（白色字体）
        cv2.putText(image, category_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, img_info["file_name"])
    cv2.imwrite(output_path, image)
    print(f"已保存可视化结果: {output_path}")

print("所有图片处理完成！")
