import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
import cv2

'''
最原始的姿态跟踪，没有评估算法和大模型校准
'''

from utils.animation_utils import find_bone, rig_skeleton,SmoothLandmarks

from pose_check import PoseCheck

check = PoseCheck()


TARGET_BONES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "RightUpLeg",
    "RightLeg",
    "RightFoot"
]

gltf_path = "../uploads/boy.glb"

canvas = RenderCanvas(
    size=(800, 600), update_mode="fastest", title="Animations", vsync=False
)

renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(75, 640 / 480, depth_range=(0.1, 1000))
camera.local.position = (0, 100, 200)
camera.look_at((0, 80, 0))
scene = gfx.Scene()

scene.add(gfx.AmbientLight(), gfx.DirectionalLight())

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
grid.local.y = -1
scene.add(grid)

axes = gfx.AxesHelper(50)
scene.add(axes)


gltf = gfx.load_gltf(gltf_path, quiet=True)
model_obj = gltf.scene.children[0]
model_obj.local.scale = (0.5, 0.5, 0.5)

skeleton_helper = gfx.SkeletonHelper(model_obj)
_, hips = find_bone(skeleton_helper, "Hips")
_, spine = find_bone(skeleton_helper, "Spine")
_, spine1 = find_bone(skeleton_helper, "Spine1")
matrix_list=[]
matrix_list.append(hips.world.matrix)
matrix_list.append(spine.local.matrix)
matrix_list.append(spine1.local.matrix)


def video_gen():
    cap = cv2.VideoCapture("../uploads/dance.mp4")
    while (cap.isOpened()):
        # 逐帧读取，ret返回布尔值
        # 参数ret为True 或者False,代表有没有读取到图片
        # frame表示截取到一帧的图片
        ret, frame = cap.read()
        if ret == True:
            yield frame
        else:
            break
    cap.release()


def rig_model(cv2_img):
    results, latency = check.check(cv2_img)
    dot_list = []
    dot_2d_list = []
    draw_list = []
    landmarks = []
    if not results.pose_world_landmarks:
        return
    if len(results.pose_world_landmarks)==0:
        return
    for lm in results.pose_world_landmarks[0]:
        draw_list.append([lm.x * 50, -lm.y * 50, -lm.z * 50 + 50])
        dot_list.append([lm.x, -lm.y, -lm.z, lm.visibility])
        landmarks.append(lm)

    for lm in results.pose_landmarks[0]:
        dot_2d_list.append([lm.x, lm.y, lm.z])

    rig_skeleton(skeleton_helper, dot_list, matrix_list,0.9)









scene.add(skeleton_helper)
scene.add(model_obj)

gfx.OrbitController(camera, register_events=renderer)

stats = gfx.Stats(viewport=renderer)


frame = video_gen()

def animate():
    try:
        frame1 = next(frame)
        rig_model(frame1)
    except StopIteration:
        print("stop")
        pass
    model_obj.children[0].skeleton.update()
    skeleton_helper.update()
    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    loop.run()