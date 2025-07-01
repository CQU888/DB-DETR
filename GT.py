import cv2
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# 设置路径
dataDir = 'E://CZY/WBCDD'  # COCO数据集根目录
annFile = f'{dataDir}/annotations/instances_train2017.json'  # 标注文件路径
imgDir = f'{dataDir}/train2017'  # 图像文件夹路径

# 初始化COCO API
coco = COCO(annFile)

# 选择要可视化的图像ID（可以随机选择或指定）
# img_ids = coco.getImgIds([228,1,64,105,120])
img_ids = coco.getImgIds([65,105,120,334,335,336,407])

for img_id in img_ids:
# img_id = img_ids[0]  # 这里选择第一张图像

    # 加载图像
    img_info = coco.loadImgs(img_id)[0]
    img_path = f'{imgDir}/{img_info["file_name"]}'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR，转为RGB

    # 加载该图像的标注
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # 在图像上绘制GT框和类别标签
    for ann in annotations:
        # 获取框坐标（COCO格式是[x,y,width,height]）
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)

        # 绘制矩形框
        color = (0, 255, 0)  # 绿色框
        thickness = 10
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        # 添加类别标签
        cat_id = ann['category_id']
        cat_name = coco.loadCats(cat_id)[0]['name']
        cv2.putText(img, cat_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,0), thickness)

    # 显示图像
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("GT/{}".format(img_info["file_name"]), bbox_inches='tight',
            dpi=400)