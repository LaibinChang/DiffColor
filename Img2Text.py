import os

img_folder = "WaterDatasetsSmall/train/input/"
paths = []

for root, _, names in os.walk(img_folder):
    for name in names:
        path = os.path.join(root, name)
        path_gt = path.replace('input', 'target')
        if os.path.exists(path_gt):
            paths.append(path)
        else:
            paths.append(path)
            print(f"No corresponding file for {path}")

save_path = "WaterDatasetsLSUI_test.txt"
with open(save_path, 'w') as f:
    for p1 in paths:
        f.write(f"{p1}\n")
