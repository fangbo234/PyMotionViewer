import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import sys
import os
import urllib.request

import config


# ==========================================
# 核心导入区
# ==========================================


# ==========================================
# 纯手动绘制工具 (不依赖 mp.solutions)
# ==========================================
class ManualDrawer:
    """
    自己实现的绘图类，完全绕过 mediapipe.solutions.drawing_utils
    防止因 protobuf 版本冲突导致的 AttributeError
    """
    # 身体连接关系 (MediaPipe 标准拓扑)
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
        (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32), (27, 31), (28, 32)
    ]

    @staticmethod
    def draw(image, detection_result):
        # 如果没有检测到人，直接返回
        if not detection_result.pose_landmarks:
            return image

        annotated_image = image.copy()
        h, w, _ = image.shape

        # 遍历每一个检测到的人
        for pose_landmarks in detection_result.pose_landmarks:
            # 1. 先画连接线 (骨骼)
            for start_idx, end_idx in ManualDrawer.POSE_CONNECTIONS:
                # 获取归一化坐标
                start_pt = pose_landmarks[start_idx]
                end_pt = pose_landmarks[end_idx]

                # 转换为像素坐标
                px_start = (int(start_pt.x * w), int(start_pt.y * h))
                px_end = (int(end_pt.x * w), int(end_pt.y * h))

                # 简单的可见性过滤
                if start_pt.visibility > 0.5 and end_pt.visibility > 0.5:
                    cv2.line(annotated_image, px_start, px_end, (255, 255, 255), 2)

            # 2. 再画关键点 (关节)
            for idx, landmark in enumerate(pose_landmarks):
                if landmark.visibility > 0.5:  # 只画可见度高的点
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # 画外圈
                    cv2.circle(annotated_image, (cx, cy), 4, (0, 0, 255), -1)
                    # 画内芯
                    cv2.circle(annotated_image, (cx, cy), 2, (255, 255, 255), -1)

        return annotated_image


# ==========================================
# 主检测类
# ==========================================
class PoseCheck:
    def __init__(self, model_name=config.MP_PATH):
        self.model_name = model_name
        self.ensure_model_exists()

        # 加载模型二进制数据 (比路径加载更稳定)
        with open(self.model_name, 'rb') as f:
            model_bytes = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            # 设为 True 可以输出分割掩码，如果不需可设 False 加速
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        print("✅ 检测器初始化成功！")

    def ensure_model_exists(self):
        """如果本地没有模型文件，自动去 Google 官网下载"""
        if not os.path.exists(self.model_name):
            print(f"⚠️ 本地未找到 {self.model_name}，正在自动下载 (约 30MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            try:
                urllib.request.urlretrieve(url, self.model_name)
                print("✅ 下载完成！")
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                print("请手动下载模型文件并放到脚本同级目录。")
                sys.exit(1)
        print(f"本地找到 {self.model_name}")

    def check(self, cv2_frame):
        # 转换颜色空间 BGR -> RGB
        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        start_time = time.perf_counter()
        result = self.detector.detect(mp_image)
        latency = time.perf_counter() - start_time

        return result, latency

    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 初始化检测器
    checker = PoseCheck()

    print("\n🚀 开始运行... 按 'ESC' 退出")

    try:
        frame =cv2.imread("./run.png")
        detection_result, latency = checker.check(frame)
        # 2. 绘制 (使用自定义的 ManualDrawer，不依赖官方库)
        annotated_frame = ManualDrawer.draw(frame, detection_result)
        cv2.imshow('MediaPipe Stable Pose', annotated_frame)

        if cv2.waitKey(0) & 0xFF == 27:
            pass

    except Exception as e:
        print(f"❌ 运行中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        checker.close()
        cv2.destroyAllWindows()