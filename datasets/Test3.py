import matplotlib.pyplot as plt
import colorsys

# --- args
args = {"light_factor": 1.2, "sat_factor": 1}

#hexs = ('FF3838', '5d1d6d', 'FF701F', '0777ff', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
        #'2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
hexs = ('FF3838', 'FF701F', '48F90A', '00D4BB', '0018EC','FF37C7')
# --- args end

def enhance_color(hex_color, light_factor=args["light_factor"], sat_factor=args["sat_factor"]):
    # 1. 解析 hex 为 rgb（0~1）
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    # 2. RGB -> HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # 3. 提升亮度和饱和度
    l = min(l * light_factor, 1.0)
    s = min(s * sat_factor, 1.0)

    # 4. HLS -> RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # 5. 转回 HEX
    return '%02X%02X%02X' % (int(r * 255), int(g * 255), int(b * 255))

brighter_hexs = [enhance_color(h) for h in hexs]
#brighter_hexs = [brighter_hexs[i] for i in [0, 2, 5, 9, 14, 19]]
# print(brighter_hexs)

# 打印对比效果
for i, (old, new) in enumerate(zip(hexs, brighter_hexs)):
    print(f"{i:2d}: {old} -> {new}")


# 将 HEX 转为 RGB 格式
colors_rgb = ['#' + h for h in brighter_hexs]

# 绘图
fig, ax = plt.subplots(figsize=(10, 2))
for i, color in enumerate(colors_rgb):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.text(i + 0.5, -0.3, str(i), ha='center', va='top', fontsize=10)

ax.set_xlim(0, len(colors_rgb))
ax.set_ylim(-1, 1)
ax.axis('off')
plt.title("Color Palette")
plt.show()
