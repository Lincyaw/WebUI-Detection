import cv2
import numpy as np
from PIL import Image

# 加载图像
image_path = "your image"
image = cv2.imread(image_path)

# 预处理图像
# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 二值化
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
# 去噪
kernel = np.ones((2,2), np.uint8)
denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 使用形态学操作来合并文字区域
# 定义一个长方形结构元素
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 10))
# 应用膨胀操作
dilation = cv2.dilate(denoised, rect_kernel, iterations=2)

# 寻找轮廓
contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓（可选，用于可视化）
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示图像（可选）
cv2.imshow('Text Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:  # 可以调整这个阈值来识别更大或更小的图像区域
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 用蓝色框标记图像区域

# 显示和保存结果
cv2.imshow('Text and Image Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('detected_text_and_images.png', image)
