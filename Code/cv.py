import cv2

# 创建摄像头对象
cap = cv2.VideoCapture(1)  # 0 表示默认摄像头，如果有多个摄像头，可以尝试不同的编号

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置摄像头分辨率（可选）
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480*2)

while True:
    # 从摄像头读取一帧图像
    ret, frame = cap.read()
    cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img1", 1000, 600)
    # 检查帧是否成功读取
    if not ret:
        print("无法读取帧")
        break
    # 获取图像的宽度和高度
    height, width, _ = frame.shape

    print("像素点范围（宽度 x 高度）：{} x {}".format(width, height))
    # 在窗口中显示图像
    cv2.imshow('img1', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()
