import os
import cv2
import xml.etree.ElementTree as ET

# Ultralytics 颜色映射类
class Colors:
    def __init__(self):
        hexs = ('FF3838', '00D4BB', 'FF701F', 'FF37C7', '00C2FF', 'FF95C8')  # 颜色列表
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c  # BGR 格式

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# 实例化颜色类
colors = Colors()

# 固定类别颜色索引
CLASS_INDEX_MAP = {
    "holothurian": 0,  # FF3838 (红色)
    "starfish": 1,     # 00D4BB (青色)
    "echinus": 2,      # FF701F (橙色)
    "scallop": 5       # FF95C8 (粉色)  (0018EC 不在默认颜色中，替换为 FF95C8)
}

# 设置 VOC 数据集路径
image_dir = r"E:\Underwater_Image_Dataset\Test\URPC2020\test\test_A_image"  # 图片路径
xml_dir = r"E:\Underwater_Image_Dataset\Test\URPC2020\test\test_A_box"  # XML 标注路径
output_dir = r"E:\1.GoodLuck\1.DiffColor\4.TMM\Reference"  # 结果保存路径

# 读取 XML 文件
os.makedirs(output_dir, exist_ok=True)
xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]

for xml_file in xml_files:
    xml_path = os.path.join(xml_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 直接用 XML 文件名替换 .xml 为 .jpg
    filename = os.path.splitext(xml_file)[0] + ".jpg"
    img_path = os.path.join(image_dir, filename)

    # 读取图片
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 无法读取图片: {img_path}")
        continue

    # 解析所有 <object>
    for obj in root.findall("object"):
        category_name = obj.find("name").text.strip()  # 获取类别名
        bbox = obj.find("bndbox")

        xmin, ymin, xmax, ymax = [int(bbox.find(tag).text) for tag in ("xmin", "ymin", "xmax", "ymax")]

        # 获取固定颜色索引（如果类别不在字典中，则按 hash 取默认颜色）
        color_index = CLASS_INDEX_MAP.get(category_name, hash(category_name) % colors.n)
        box_color = colors(color_index, bgr=True)

        # 绘制边界框
        box_thickness = 12
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, box_thickness)

        # 设置字体
        font_scale = 4
        font_thickness = 8

        # 获取文本大小
        (text_w, text_h), _ = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # 绘制文本背景（提升可读性）
        cv2.rectangle(image, (xmin, ymin - text_h - 5), (xmin + text_w, ymin), box_color, -1)

        # 绘制类别名称（白色字体）
        cv2.putText(image, category_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # 保存可视化结果
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    print(f"✅ 已保存可视化结果: {output_path}")

print("🎉 所有图片处理完成！")
