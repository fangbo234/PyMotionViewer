import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
import cv2
import numpy as np

# 假设这些是你自定义的工具库
from utils.animation_utils import find_bone, rig_skeleton, SmoothLandmarks
from pose_check import PoseCheck


class PoseEstimator:
    """
    负责姿态估计与数据平滑
    """

    def __init__(self):
        self.detector = PoseCheck()
        self.smoother = SmoothLandmarks()

    def process(self, cv2_img):
        """
        输入图片，返回平滑后的 MediaPipe 3D 坐标数据 (dot_list)
        """
        results, latency = self.detector.check(cv2_img)

        if not results.pose_world_landmarks or len(results.pose_world_landmarks) == 0:
            return None

        # 提取原始数据
        landmarks = []
        dot_list = []

        # 提取用于平滑的原始对象
        for lm in results.pose_world_landmarks[0]:
            landmarks.append(lm)

        # 执行平滑算法
        smooth_landmarks = self.smoother.deal(landmarks)

        if smooth_landmarks:
            # 格式化为 rig_skeleton 需要的格式 [x, -y, -z, visibility]
            for lm in results.pose_world_landmarks[0]:  # 注意：这里如果用 smooth_landmarks 应该遍历 smooth 的结果
                # 原始代码逻辑里，你用的是 results.pose_world_landmarks[0] 构建 dot_list
                # 但判断是用 smooth_landmarks。这里保留你原始的逻辑结构。
                dot_list.append([lm.x, -lm.y, -lm.z, lm.visibility])
            return dot_list

        return None


class AvatarModel:
    """
    负责 3D 模型的加载、骨骼绑定状态管理和姿态驱动
    """

    def __init__(self, model_path, scale=(0.5, 0.5, 0.5)):
        print(f"Loading model from: {model_path}")
        self.gltf = gfx.load_gltf(model_path, quiet=True)
        self.obj = self.gltf.scene.children[0]
        self.obj.local.scale = scale

        self.skeleton_helper = gfx.SkeletonHelper(self.obj)

        # 核心逻辑：捕获初始绑定姿态矩阵 (Bind Pose Matrices)
        # 这对于后续的重定向计算至关重要
        self.bind_matrices = self._capture_bind_pose()

    def _capture_bind_pose(self):
        """捕获初始骨骼矩阵"""
        matrices = []
        found_hips, hips = find_bone(self.skeleton_helper, "Hips")
        found_spine, spine = find_bone(self.skeleton_helper, "Spine")
        found_spine1, spine1 = find_bone(self.skeleton_helper, "Spine1")

        if found_hips and found_spine and found_spine1:
            matrices.append(hips.world.matrix)
            matrices.append(spine.local.matrix)
            matrices.append(spine1.local.matrix)
        else:
            print("⚠️ Warning: Critical bones (Hips, Spine, Spine1) not found!")

        return matrices

    def add_to_scene(self, scene):
        scene.add(self.obj)
        scene.add(self.skeleton_helper)

    def update_pose(self, dot_list, lerp=0.9):
        """驱动骨骼"""
        if dot_list and self.bind_matrices:
            rig_skeleton(self.skeleton_helper, dot_list, self.bind_matrices, lerp)

    def update_matrix(self):
        """刷新 pygfx 的骨骼矩阵"""
        if hasattr(self.obj.children[0], 'skeleton'):
            self.obj.children[0].skeleton.update()
        self.skeleton_helper.update()


class MotionApp:
    """
    主程序：负责渲染循环、视频流读取和场景管理
    """

    def __init__(self, video_path, model_path):
        # 1. 初始化渲染环境
        self.canvas = RenderCanvas(size=(800, 600), update_mode="fastest", title="Motion Tracker", vsync=False)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.camera = self._setup_camera()
        self._setup_lights_and_grid()

        # 2. 初始化核心模块
        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        self.pose_estimator = PoseEstimator()
        self.avatar = AvatarModel(model_path)

        # 3. 将角色加入场景
        self.avatar.add_to_scene(self.scene)

        # 4. 性能监控
        self.stats = gfx.Stats(viewport=self.renderer)

    def _setup_camera(self):
        camera = gfx.PerspectiveCamera(75, 640 / 480, depth_range=(0.1, 1000))
        camera.local.position = (0, 100, 200)
        camera.look_at((0, 80, 0))
        # 控制器
        self.controller = gfx.OrbitController(camera, register_events=self.renderer)
        return camera

    def _setup_lights_and_grid(self):
        self.scene.add(gfx.AmbientLight(), gfx.DirectionalLight())
        grid = gfx.Grid(
            None,
            gfx.GridMaterial(major_step=5, thickness_space="world", major_thickness=0.6, infinite=True),
            orientation="xz",
        )
        grid.local.y = -1
        self.scene.add(grid)
        self.scene.add(gfx.AxesHelper(50))

    def get_next_frame(self):
        """获取下一帧视频，如果结束则返回 None"""
        if self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                return frame
        return None

    def animate(self):
        # 1. 读取视频帧
        frame = self.get_next_frame()

        if frame is not None:
            # 2. 姿态估计
            pose_data = self.pose_estimator.process(frame)

            # 3. 驱动模型
            if pose_data:
                self.avatar.update_pose(pose_data)
        else:
            print("Video ended or failed to read.")
            # 可选：循环播放
            # self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 4. 更新渲染状态
        self.avatar.update_matrix()

        with self.stats:
            self.renderer.render(self.scene, self.camera, flush=False)
        self.stats.render()
        self.canvas.request_draw()

    def run(self):
        self.renderer.request_draw(self.animate)
        loop.run()


if __name__ == "__main__":
    # 配置路径
    VIDEO_FILE = "../assets/dance4.mp4"
    MODEL_FILE = "../assets/boy.glb"

    # 启动应用
    app = MotionApp(VIDEO_FILE, MODEL_FILE)
    app.run()