import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GLB_PATH = os.path.join(BASE_DIR, "mediapipe", "boy.glb")

JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
    'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

# SMPL 父子关系
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)

# SMPL T-Pose 偏移
OFFSETS = np.array([
    [0, 0, 0], [0.07, -0.04, -0.01], [-0.07, -0.04, -0.01], [0, 0.1, -0.01],
    [0.05, -0.40, 0], [-0.05, -0.40, 0], [0, 0.15, 0], [0, -0.40, 0],
    [0, -0.40, 0], [0, 0.15, 0], [0, -0.08, 0.15], [0, -0.08, 0.15],
    [0, 0.15, 0], [0.08, 0.10, -0.02], [-0.08, 0.10, -0.02], [0, 0.15, 0],
    [0.12, 0, 0], [-0.12, 0, 0], [0.25, 0, 0], [-0.25, 0, 0],
    [0.25, 0, 0], [-0.25, 0, 0], [0.10, 0, 0], [-0.10, 0, 0],
], dtype=np.float32)

# SMPL -> Mixamo 骨骼名称映射
SMPL_TO_MIXAMO = {
    'Pelvis': 'mixamorig:Hips',
    'L_Hip': 'mixamorig:LeftUpLeg',
    'R_Hip': 'mixamorig:RightUpLeg',
    'Spine1': 'mixamorig:Spine',
    'Spine2': 'mixamorig:Spine1',
    'Spine3': 'mixamorig:Spine2',
    'Neck': 'mixamorig:Neck',
    'Head': 'mixamorig:Head',
    'L_Collar': 'mixamorig:LeftShoulder',
    'R_Collar': 'mixamorig:RightShoulder',
    'L_Shoulder': 'mixamorig:LeftArm',
    'R_Shoulder': 'mixamorig:RightArm',
    'L_Elbow': 'mixamorig:LeftForeArm',
    'R_Elbow': 'mixamorig:RightForeArm',
    'L_Wrist': 'mixamorig:LeftHand',
    'R_Wrist': 'mixamorig:RightHand',
    'L_Knee': 'mixamorig:LeftLeg',
    'R_Knee': 'mixamorig:RightLeg',
    'L_Ankle': 'mixamorig:LeftFoot',
    'R_Ankle': 'mixamorig:RightFoot',
    'L_Foot': 'mixamorig:LeftToeBase',
    'R_Foot': 'mixamorig:RightToeBase',
    'L_Hand':'mixamorig:LeftHandMiddle1',
    'R_Hand':'mixamorig:RightHandMiddle2',
}
