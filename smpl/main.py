import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
import numpy as np
import joblib
import time
import pylinalg as la
import math
from utils.animation_utils import rig_with_pos
# 导入配置
import config





# ==========================================
# 模块 1: 数据加载与计算 (MotionLoader)
# ==========================================
class MotionLoader:
    """负责加载 pkl 并计算 FK，支持多人嵌套结构及帧号映射"""

    def __init__(self, pkl_path, person_id=None):
        self.pkl_path = pkl_path
        self.person_id = person_id

        self.pose = None
        self.trans = None
        self.frame_ids = None
        self.frame_map = None  # 核心：帧号 -> 数组索引的查找表
        self.joints_seq = None
        self.num_frames = 0

        self._load_and_process()

    def _load_and_process(self):
        print(f"Loading {self.pkl_path}...")
        try:
            raw_data = joblib.load(self.pkl_path)
        except Exception as e:
            print(f"Joblib failed, trying pickle: {e}")
            import pickle
            with open(self.pkl_path, 'rb') as f:
                raw_data = pickle.load(f)

        target_data = None

        # --- 适配多人/单人结构 ---
        if isinstance(raw_data, dict):
            # 检查是否是多人字典 {0: {...}, 1: {...}}
            first_key = list(raw_data.keys())[0] if raw_data else None
            is_multi = isinstance(first_key, (int, np.integer, str)) and \
                       isinstance(raw_data[first_key], dict) and \
                       'pose_world' in raw_data[first_key]

            if is_multi:
                if self.person_id is not None:
                    if self.person_id in raw_data:
                        target_data = raw_data[self.person_id]
                    else:
                        print(f"⚠️ ID {self.person_id} 不存在，回退到 {first_key}")
                        target_data = raw_data[first_key]
                else:
                    target_data = raw_data[first_key]

            elif 'pose_world' in raw_data:
                target_data = raw_data

        elif isinstance(raw_data, list):
            idx = self.person_id if self.person_id is not None else 0
            if idx < len(raw_data):
                target_data = raw_data[idx]
            else:
                target_data = raw_data[0]

        if target_data is None:
            raise ValueError("无法提取数据")

        # --- 提取数据 ---
        self.pose = np.array(target_data['pose_world'], dtype=np.float32)
        self.trans = np.array(target_data['trans_world'], dtype=np.float32)
        self.num_frames = self.pose.shape[0]

        # Reshape
        self.pose = self.pose.reshape(self.num_frames, 24, 3)

        # --- 核心：建立 Frame Map ---
        if 'frame_ids' in target_data:
            self.frame_ids = np.array(target_data['frame_ids'])
            # 建立映射: {视频帧号: 数据索引}
            # 例如: 第 100 帧对应数据的第 0 个索引 -> {100: 0}
            self.frame_map = {int(fid): i for i, fid in enumerate(self.frame_ids)}
        else:
            self.frame_map = None

        print(f"Computing FK for {self.num_frames} frames...")
        self.joints_seq = self._compute_fk(self.pose, self.trans)
        print("FK Done.")

    def _rodrigues_batch(self, axis_angle):
        N = axis_angle.shape[0]
        theta = np.linalg.norm(axis_angle, axis=1, keepdims=True)
        with np.errstate(invalid='ignore'):
            r = axis_angle / theta
        r = np.nan_to_num(r)
        rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
        K = np.zeros((N, 3, 3), dtype=np.float32)
        K[:, 0, 1], K[:, 0, 2] = -rz, ry
        K[:, 1, 0], K[:, 1, 2] = rz, -rx
        K[:, 2, 0], K[:, 2, 1] = -ry, rx
        I = np.eye(3, dtype=np.float32)
        theta = theta[:, :, None]
        return I + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)

    def _compute_fk(self, pose_frames, trans_frames):
        num_frames = pose_frames.shape[0]
        num_joints = 24
        pose_flat = pose_frames.reshape(-1, 3)
        rot_mats = self._rodrigues_batch(pose_flat).reshape(num_frames, num_joints, 3, 3)

        global_xforms = np.zeros((num_frames, num_joints, 4, 4), dtype=np.float32)
        global_xforms[:, :, 3, 3] = 1.0

        for i in range(num_joints):
            parent = config.PARENTS[i]
            local_transform = np.zeros((num_frames, 4, 4), dtype=np.float32)
            local_transform[:, :3, :3] = rot_mats[:, i]
            local_transform[:, 3, 3] = 1.0

            if i == 0:
                local_transform[:, :3, 3] = trans_frames
            else:
                local_transform[:, :3, 3] = config.OFFSETS[i]

            if parent == -1:
                global_xforms[:, i] = local_transform
            else:
                global_xforms[:, i] = np.matmul(global_xforms[:, parent], local_transform)

        return global_xforms[:, :, :3, 3]


# ==========================================
# 模块 2: 角色控制器 (MixamoCharacter)
# ==========================================
class MixamoCharacter:
    """负责 GLB 模型管理、骨骼重定向及显隐控制"""

    def __init__(self, glb_path, scene, scale=0.01, offset=(0, 0, 0)):
        self.scene = scene
        self.bones_cache = {}
        self.retarget_chain = []
        self.motion_data = None
        self.offset = offset  # 存储位置偏移

        # 1. 加载模型
        self.model = self._load_model(glb_path, scale)
        # 2. 缓存骨骼
        self._cache_bones()
        # 3. 定义重定向链条
        self._setup_retarget_chain()

    def bind_motion(self, motion_loader):
        self.motion_data = motion_loader

    def _load_model(self, path, scale):
        gltf = gfx.load_gltf(path, quiet=True)
        model = gltf.scene.children[0]
        model.local.scale = (scale, scale, scale)
        model.visible = True
        self.helper = gfx.SkeletonHelper(model)
        self.scene.add(self.helper)
        self.scene.add(model)
        return model

    def _find_bone(self, node, name):
        for bone in node.bones:
            if bone.name == name: return bone
        return None

    def _cache_bones(self):
        for smpl_name, mixamo_name in config.SMPL_TO_MIXAMO.items():
            bone = self._find_bone(self.helper, mixamo_name)
            if bone:
                self.bones_cache[smpl_name] = bone
            else:
                short_name = mixamo_name.replace("mixamorig:", "")
                bone = self._find_bone(self.helper, short_name)
                if bone:
                    self.bones_cache[smpl_name] = bone

    def _setup_retarget_chain(self):
        self.retarget_rules = [
            ('L_Hip', 'L_Knee', 1, 4),
            ('L_Knee', 'L_Ankle', 4, 7),
            ('L_Ankle', 'L_Foot', 7, 10),
            ('R_Hip', 'R_Knee', 2, 5),
            ('R_Knee', 'R_Ankle', 5, 8),
            ('R_Ankle', 'R_Foot', 8, 11),
            ('Spine1', 'Spine2', 3, 6),
            ('Spine2', 'Spine3', 6, 9),
            ('Spine3', 'Neck', 9, 12),
            ('Neck', 'Head', 12, 15),
            ('L_Collar', 'L_Shoulder', 13, 16),
            ('L_Shoulder', 'L_Elbow', 16, 18),
            ('L_Elbow', 'L_Wrist', 18, 20),
            ('L_Wrist', 'L_Hand', 20, 22),
            ('R_Collar', 'R_Shoulder', 14, 17),
            ('R_Shoulder', 'R_Elbow', 17, 19),
            ('R_Elbow', 'R_Wrist', 19, 21),
            ('R_Wrist', 'R_Hand', 21, 23)
        ]

    def update(self, global_frame_idx):
        """
        根据全局视频帧号更新姿态。
        如果当前帧此人不在场，则隐藏模型。
        """
        if self.motion_data is None: return

        idx = 0
        is_visible = False

        # --- A. 基于 Frame ID 的精确匹配 ---
        if self.motion_data.frame_map is not None:
            if global_frame_idx in self.motion_data.frame_map:
                idx = self.motion_data.frame_map[global_frame_idx]
                is_visible = True
            else:
                is_visible = False

        # --- B. 无 Frame ID 时的兜底逻辑 (循环播放) ---
        else:
            idx = global_frame_idx % self.motion_data.num_frames
            is_visible = True

        # --- 设置显隐状态 ---
        if self.model.visible != is_visible:
            self.model.visible = is_visible
            self.helper.visible = is_visible  # 骨骼辅助线也一起隐藏

        if not is_visible:
            return  # 不在场，直接跳过计算

        # --- 获取数据 ---
        current_pose_euler = self.motion_data.pose[idx]
        current_trans = self.motion_data.trans[idx]
        current_joints = self.motion_data.joints_seq[idx]

        # 1. 更新 Hips (应用位移 + 偏移量)
        if 'Pelvis' in self.bones_cache:
            hips = self.bones_cache['Pelvis']
            hips.world.position = (
                current_trans[0] + self.offset[0],
                current_trans[1] + self.offset[1],
                current_trans[2] + self.offset[2]
            )
            hips.world.rotation = la.quat_from_euler(current_pose_euler[0])

        # 2. 批量更新肢体
        for start_key, end_key, idx_a, idx_b in self.retarget_rules:
            if start_key in self.bones_cache and end_key in self.bones_cache:
                bone_start = self.bones_cache[start_key]
                bone_end = self.bones_cache[end_key]
                pos_a = current_joints[idx_a]
                pos_b = current_joints[idx_b]
                rig_with_pos(bone_start, bone_end, pos_a, pos_b)

        self.helper.update()
        self.model.children[0].skeleton.update()


# ==========================================
# 模块 3: 场景与渲染管理器 (SceneManager)
# ==========================================
class SceneManager:
    def __init__(self, title="Animation"):
        self.canvas = RenderCanvas(size=(800, 600), update_mode="fastest", title=title, vsync=False)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self._setup_env()
        self._setup_camera()

    def _setup_env(self):
        self.scene.add(gfx.AmbientLight(), gfx.DirectionalLight())
        grid = gfx.Grid(None,
                        gfx.GridMaterial(major_step=5, thickness_space="world", major_thickness=0.1, infinite=True),
                        orientation="xz")
        grid.local.y = -1
        self.scene.add(grid)
        self.scene.add(gfx.AxesHelper(50))

    def _setup_camera(self):
        self.camera = gfx.PerspectiveCamera(75, 640 / 480, depth_range=(0.1, 1000))
        self.camera.local.position = (0, 2, 4)
        self.camera.look_at((0, 1, 0))
        self.controller = gfx.OrbitController(self.camera, register_events=self.renderer)

    def render(self):
        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    PKL_FILE = "pkl/dance4.pkl"
    # 模型列表，程序会轮流分配给不同的人
    GLB_LIST = ["../uploads/boy.glb", "../uploads/boy.glb"]

    # 空间偏移步长 (米)，防止重叠
    OFFSET_STEP = 1.5

    app = SceneManager(title="Final Multi-Person Retargeting")

    print(f"Scanning {PKL_FILE}...")
    try:
        raw = joblib.load(PKL_FILE)
    except:
        import pickle

        with open(PKL_FILE, 'rb') as f:
            raw = pickle.load(f)

    # --- 1. 解析与过滤 (50% 存活阈值) ---
    raw_people_data = {}

    # 统一数据结构为 {id: data}
    if isinstance(raw, dict):
        keys = list(raw.keys())
        first_key = keys[0]
        if isinstance(first_key, (int, np.integer, str)) and isinstance(raw[first_key], dict) and 'pose_world' in raw[
            first_key]:
            raw_people_data = raw
        elif 'pose_world' in raw:
            raw_people_data = {0: raw}
    elif isinstance(raw, list):
        for i, p_data in enumerate(raw):
            raw_people_data[i] = p_data

    # 计算持续时间
    person_durations = {}
    max_duration = 0

    for pid, data in raw_people_data.items():
        if 'pose_world' in data:
            count = len(data['pose_world'])
            person_durations[pid] = count
            if count > max_duration:
                max_duration = count

    # 过滤
    THRESHOLD_RATIO = 0.5
    cutoff_frames = max_duration * THRESHOLD_RATIO
    valid_person_ids = []

    print(f"\n📊 过滤统计 (Max: {max_duration}, Threshold: {cutoff_frames:.1f}):")
    for pid, count in person_durations.items():
        if count >= cutoff_frames:
            valid_person_ids.append(pid)
            print(f"   ✅ 保留 ID {pid}: {count} 帧")
        else:
            print(f"   ❌ 丢弃 ID {pid}: {count} 帧")

    if not valid_person_ids:
        valid_person_ids = [list(raw_people_data.keys())[0]]

    # --- 2. 初始化演员 ---
    actors = []
    print(f"\n初始化 {len(valid_person_ids)} 位演员...")

    for i, pid in enumerate(valid_person_ids):
        # 计算 X 轴偏移
        pos_offset = (i * OFFSET_STEP, 0, 0)

        # 加载数据
        motion = MotionLoader(PKL_FILE, person_id=pid)

        # 加载模型 (轮换使用 GLB)
        glb_file = GLB_LIST[i % len(GLB_LIST)]
        char = MixamoCharacter(glb_file, app.scene, scale=0.01, offset=pos_offset)

        # 绑定
        char.bind_motion(motion)
        actors.append(char)
        print(f"   -> Actor {pid} loaded. Model: {glb_file}, Offset: {pos_offset}")

    # --- 3. 动画循环 ---
    TARGET_FPS = 30.0
    start_time = time.perf_counter()


    def animate():
        elapsed_time = time.perf_counter() - start_time
        # 这里计算的是绝对帧号 (Global Frame ID)
        target_frame = int(elapsed_time * TARGET_FPS)

        for char in actors:
            char.update(target_frame)

        app.render()


    app.renderer.request_draw(animate)
    loop.run()