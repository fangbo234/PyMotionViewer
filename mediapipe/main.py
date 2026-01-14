import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
import cv2
import numpy as np

# 假设这些工具函数在 utils.animation_utils 中
from utils.animation_utils import clone_skeleton, find_bone, pre_rig_skeleton

# ==========================================
# 配置与常量
# ==========================================
TARGET_BONES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"
]


class PoseEstimator:
    """负责从图像中提取姿态数据"""

    def __init__(self):
        # 延迟导入，避免循环依赖
        from pose_check import PoseCheck
        self.detector = PoseCheck()

    def predict(self, image_path):
        """读取图片并返回关键点数据"""
        cv2_img = cv2.imread(image_path)
        if cv2_img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        results, latency = self.detector.check(cv2_img)

        if not results.pose_world_landmarks:
            print("No pose detected.")
            return None

        # 转换数据格式
        dot_list = []
        for lm in results.pose_world_landmarks[0]:
            # 注意：这里保留了你原始代码中的坐标转换逻辑
            dot_list.append([lm.x, -lm.y, -lm.z, lm.visibility])

        return dot_list


class Avatar:
    """负责 3D 模型管理与骨骼重定向"""

    def __init__(self, glb_path, scale=(0.5, 0.5, 0.5)):
        self.model = self._load_model(glb_path, scale)
        self.skeleton_helper = gfx.SkeletonHelper(self.model)

    def _load_model(self, path, scale):
        gltf = gfx.load_gltf(path, quiet=True)
        model = gltf.scene.children[0]
        model.local.scale = scale
        return model

    def add_to_scene(self, scene):
        """将模型和骨骼辅助线添加到场景"""
        scene.add(self.model)
        scene.add(self.skeleton_helper)

    def update(self):
        """更新骨骼矩阵动画"""
        self.model.children[0].skeleton.update()
        self.skeleton_helper.update()

    def apply_pose(self, landmarks):
        """核心重定向逻辑：将 Landmark 数据应用到模型骨骼"""
        if landmarks is None:
            return

        # 1. 获取基础骨骼矩阵 (Rest/Bind Pose)
        # 注意：这部分逻辑依赖具体的骨骼命名，如果模型变了可能需要调整
        found_hips, hips = find_bone(self.skeleton_helper, "Hips")
        found_spine, spine = find_bone(self.skeleton_helper, "Spine")
        found_spine1, spine1 = find_bone(self.skeleton_helper, "Spine1")

        if not (found_hips and found_spine and found_spine1):
            print("Error: Essential bones (Hips, Spine, Spine1) not found.")
            return

        matrix_list = []
        matrix_list.append(hips.world.matrix)
        matrix_list.append(spine.local.matrix)
        matrix_list.append(spine1.local.matrix)

        # 2. 克隆虚拟骨架用于计算 IK/FK
        root_bone = clone_skeleton(self.skeleton_helper)

        # 3. 预计算骨骼姿态 (这也是你原代码中引用的外部函数)
        pre_rig_skeleton(root_bone, landmarks, matrix_list)

        # 4. 将计算结果映射回渲染模型
        for name in TARGET_BONES:
            found, bone = find_bone(self.skeleton_helper, name)
            if not found:
                continue

            src_bone = root_bone.findByName(name)
            if src_bone:
                if name == "Hips":
                    bone.world.rotation = src_bone.getWorldRotation()
                else:
                    bone.local.rotation = src_bone.getLocalRotation()


class SceneViewer:
    """负责 Pygfx 场景渲染管理"""

    def __init__(self, title="Animations"):
        self.canvas = RenderCanvas(size=(800, 600), update_mode="fastest", title=title, vsync=False)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.camera = self._setup_camera()
        self._setup_environment()

        # 控制器
        self.controller = gfx.OrbitController(self.camera, register_events=self.renderer)
        self.stats = gfx.Stats(viewport=self.renderer)

    def _setup_camera(self):
        camera = gfx.PerspectiveCamera(75, 640 / 480, depth_range=(0.1, 1000))
        camera.local.position = (0, 100, 200)
        camera.look_at((0, 80, 0))
        return camera

    def _setup_environment(self):
        self.scene.add(gfx.AmbientLight(), gfx.DirectionalLight())

        grid = gfx.Grid(
            None,
            gfx.GridMaterial(
                major_step=5,
                thickness_space="world",
                major_thickness=0.6,
                infinite=True,
            ),
            orientation="xz",
        )
        grid.local.y = 0
        self.scene.add(grid)
        self.scene.add(gfx.AxesHelper(50))

    def add_object(self, obj):
        self.scene.add(obj)  # 简单的添加方法，如果 Avatar 类有特殊添加逻辑，可以在外部处理

    def render(self):
        with self.stats:
            self.renderer.render(self.scene, self.camera, flush=False)
        self.stats.render()
        self.canvas.request_draw()

    def start(self, animation_callback):
        self.renderer.request_draw(animation_callback)
        loop.run()


# ==========================================
# 主程序入口
# ==========================================
def main():
    # 路径配置
    GLB_PATH = "../assets/boy.glb"
    IMAGE_PATH = "../assets/example.png"

    # 1. 初始化模块
    app = SceneViewer()
    estimator = PoseEstimator()
    character = Avatar(GLB_PATH, scale=(0.5, 0.5, 0.5))

    # 2. 组装场景
    character.add_to_scene(app.scene)

    # 3. 执行姿态估计与重定向 (Rigging)
    print(f"Analyzing image: {IMAGE_PATH}...")
    landmarks = estimator.predict(IMAGE_PATH)

    if landmarks:
        print("Applying pose to character...")
        character.apply_pose(landmarks)

    # 4. 定义动画循环
    def animate():
        character.update()
        app.render()

    # 5. 启动
    print("Starting render loop...")
    app.start(animate)


if __name__ == "__main__":
    main()