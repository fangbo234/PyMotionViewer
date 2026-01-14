import functools
import math
from typing import Optional

import pygfx as gfx
from .vector_utils import *

mixamo_bone_map = {"Hips", "Spine", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
                   "RightHandIndex1", "RightUpLeg", "RightLeg", "RightFoot",
                   "RightToeBase", "RightToe_End", "LeftShoulder", "LeftArm",
                   "LeftForeArm", "LeftHand", "LeftHandIndex1", "LeftUpLeg",
                   "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToe_End", "Head", "Neck"}
vrm_bone_map = {"Hips": "J_Bip_C_Hips", "Spine": "J_Bip_C_Spine", "RightShoulder": "J_Bip_R_Shoulder",
                "RightArm": "J_Bip_R_UpperArm",
                "RightForeArm": "J_Bip_R_LowerArm", "RightHand": "J_Bip_R_Hand",
                "RightHandIndex1": "J_Bip_R_Index1", "RightUpLeg": "J_Bip_R_UpperLeg", "RightLeg": "J_Bip_R_LowerLeg",
                "RightFoot": "J_Bip_R_Foot",
                "RightToeBase": "J_Bip_R_ToeBase", "LeftShoulder": "J_Bip_L_Shoulder", "LeftArm": "J_Bip_L_UpperArm",
                "LeftForeArm": "J_Bip_L_LowerArm", "LeftHand": "J_Bip_L_Hand", "LeftHandIndex1": "J_Bip_L_Index1",
                "LeftUpLeg": "J_Bip_L_UpperLeg",
                "LeftLeg": "J_Bip_L_LowerLeg", "LeftFoot": "J_Bip_L_Foot", "LeftToeBase": "J_Bip_L_ToeBase",
                "Head": "J_Bip_C_Head", "Neck": "J_Bip_C_Neck"}

tripo_bone_map = {"Hips": "Root",
                  "Pelvis": "Pelvis",
                  "LeftUpLeg": "L_Thigh",
                  "LeftLeg": "L_Calf",
                  "LeftFoot": "L_Foot",
                  "L_CalfTwist01": "L_CalfTwist01",
                  "L_CalfTwist02": "L_CalfTwist02",
                  "L_ThighTwist01": "L_ThighTwist01",
                  "L_ThighTwist02": "L_ThighTwist02",
                  "RightUpLeg": "R_Thigh",
                  "R_ThighTwist01": "R_ThighTwist01",
                  "R_ThighTwist02": "R_ThighTwist02",
                  "RightLeg": "R_Calf",
                  "RightFoot": "R_Foot",
                  "R_CalfTwist01": "R_CalfTwist01",
                  "R_CalfTwist02": "R_CalfTwist02",
                  "Waist": "Waist",
                  "Spine": "Spine01",
                  "Spine02": "Spine02",
                  "NeckTwist01": "NeckTwist01",
                  "NeckTwist02": "NeckTwist02",
                  "Head": "Head",
                  "LeftShoulder": "L_Clavicle",
                  "LeftArm": "L_Upperarm",
                  "LeftForeArm": "L_Forearm",
                  "L_ForearmTwist01": "L_ForearmTwist01",
                  "L_ForearmTwist02": "L_ForearmTwist02",
                  "LeftHand": "L_Hand",
                  "L_UpperarmTwist01": "L_UpperarmTwist01",
                  "L_UpperarmTwist02": "L_UpperarmTwist02",
                  "RightShoulder": "R_Clavicle",
                  "RightArm": "R_Upperarm",
                  "R_UpperarmTwist01": "R_UpperarmTwist01",
                  "R_UpperarmTwist02": "R_UpperarmTwist02",
                  "RightForeArm": "R_Forearm",
                  "R_ForearmTwist01": "R_ForearmTwist01",
                  "R_ForearmTwist02": "R_ForearmTwist02",
                  "RightHand": "R_Hand"}
bone_names = ["Spine", "Spine1", "Spine2", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
              "RightShoulder", "RightArm", "RightForeArm", "RightHand",
              "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]


def extract_from_glb(self, file_path):
    gltf = gfx.load_gltf("../models/walk.glb")
    action_clip = gltf.animations[0]
    return action_clip


def print_qua(angle):
    print("{:.2f}".format(angle[0]), "{:.2f}".format(angle[1]), "{:.2f}".format(angle[2]), "{:.2f}".format(angle[3]))


def axes_helper():
    return gfx.AxesHelper(size=100, thickness=1)


def find_bone(skeleton_helper, bone_name):
    # RightUpLeg LeftUpLeg LeftShoulder RightShoulder Hips
    # mixamo vrm tripo
    model_type = "mixamo"

    for bone in skeleton_helper.bones:
        if "_Hips" in bone.name:
            model_type = "vrm"
            break
        if "mixamorig:Hips" in bone.name:
            model_type = "mixamo"
            break
        if "Pelvis" in bone.name:
            model_type = "tripo"
            break
    find_name = ""
    if model_type == "vrm" and bone_name in vrm_bone_map:
        find_name = vrm_bone_map[bone_name]
    if model_type == "mixamo":
        find_name = "mixamorig:" + bone_name
    if model_type == "tripo" and bone_name in tripo_bone_map:
        find_name = tripo_bone_map[bone_name]

    for bone in skeleton_helper.bones:
        if find_name == bone.name:
            return True, bone
    return False, None


def is_head_neck(bone):
    if "Head" in bone.name or "Neck" in bone.name:
        return True
    return False


def is_arm(bone):
    if "LeftArm" in bone.name or "RightArm" in bone.name:
        return True
    return False


def is_forearm(bone):
    if "LeftForeArm" in bone.name or "RightForeArm" in bone.name:
        return True
    return False


def is_upleg(bone):
    if "LeftUpLeg" in bone.name or "RightUpLeg" in bone.name:
        return True
    return False


def is_leg(bone):
    if "LeftLeg" in bone.name or "RightLeg" in bone.name:
        return True
    return False


# 计算某个点沿着向量的方向 距离 d的点
def cal_point_with_angle(point, vector, distance):
    pass


def draw_line(landmark_list, a, b):
    return gen_line(landmark_list[a], landmark_list[b])


def gen_line(point_a, point_b, color=(0.8, 0.7, 0.0, 1.0)):
    point_list = [point_a, point_b]
    geometry = gfx.Geometry(positions=point_list)
    material = gfx.LineMaterial(thickness=2.0, color=color)
    line = gfx.Line(geometry, material)
    return line


def gen_sphere(radius):
    geometry = gfx.sphere_geometry(radius, phi_length=np.pi * 2)
    material = gfx.MeshPhongMaterial(color=(1, 0, 0), flat_shading=True)
    s1 = gfx.Mesh(geometry, material)
    return s1


def line_2points(a, b, c=(1, 1, 1, 1)):
    pos_list = [a, b]
    geometry = gfx.Geometry(positions=pos_list)
    material = gfx.LineMaterial(thickness=2.0, color=c)
    line = gfx.Line(geometry, material)
    return line


# 从中心点看向某点的旋转角度
def look_at_vec(bone, a, b, c):
    start, end = get_orientation_vector(a, b, c)
    v = np.subtract(end, start)
    v1 = np.add(bone.world.position, la.vec_normalize(v))
    bone.look_at(v1)


# 获取三个点的面向量
def get_dest_vec(a, b, c):
    start, end = get_orientation_vector(a, b, c)
    v = np.subtract(end, start)
    return v


def line_bone_orientation(bone, a, b, c):
    start, end = get_orientation_vector(a, b, c)
    v = np.subtract(end, start)
    return line_2points(np.add(v, bone.world.position), bone.world.position)


def draw_gfx_person(m_group, dot_list):
    landmark_list = []
    for dot in dot_list:
        landmark_list.append(dot)

    for pos in landmark_list:
        geometry = gfx.sphere_geometry(0.5, phi_length=np.pi * 1.5)
        material = gfx.MeshPhongMaterial(color=(1, 0, 0), flat_shading=True)
        wobject = gfx.Mesh(geometry, material)
        wobject.local.position = pos

    m_group.add(draw_line(landmark_list, 11, 12))
    m_group.add(draw_line(landmark_list, 24, 23))
    m_group.add(draw_line(landmark_list, 11, 23))
    m_group.add(draw_line(landmark_list, 24, 12))
    m_group.add(draw_line(landmark_list, 24, 26))
    m_group.add(draw_line(landmark_list, 26, 28))
    m_group.add(draw_line(landmark_list, 28, 32))
    m_group.add(draw_line(landmark_list, 32, 30))
    m_group.add(draw_line(landmark_list, 28, 30))
    m_group.add(draw_line(landmark_list, 23, 25))
    m_group.add(draw_line(landmark_list, 25, 27))
    m_group.add(draw_line(landmark_list, 27, 29))
    m_group.add(draw_line(landmark_list, 29, 31))
    m_group.add(draw_line(landmark_list, 27, 31))
    m_group.add(draw_line(landmark_list, 12, 14))
    m_group.add(draw_line(landmark_list, 14, 16))
    m_group.add(draw_line(landmark_list, 16, 22))
    m_group.add(draw_line(landmark_list, 16, 18))
    m_group.add(draw_line(landmark_list, 18, 20))
    m_group.add(draw_line(landmark_list, 16, 20))
    m_group.add(draw_line(landmark_list, 11, 13))
    m_group.add(draw_line(landmark_list, 13, 15))
    m_group.add(draw_line(landmark_list, 15, 21))
    m_group.add(draw_line(landmark_list, 15, 17))
    m_group.add(draw_line(landmark_list, 17, 19))
    m_group.add(draw_line(landmark_list, 19, 15))
    m_group.add(draw_line(landmark_list, 9, 10))
    m_group.add(draw_line(landmark_list, 0, 5))
    m_group.add(draw_line(landmark_list, 5, 8))
    m_group.add(draw_line(landmark_list, 0, 2))
    m_group.add(draw_line(landmark_list, 2, 7))


def draw_line_along_bone(bone):
    point = la.vec_transform_quat((0, 100, 0), bone.world.rotation)
    return line_2points(bone.world.position, np.add(point, bone.world.position))


def print_euler(angle):
    print("{:.2f}".format(angle[0]), "{:.2f}".format(angle[1]), "{:.2f}".format(angle[2]))


def rig_bone(bone, bone_end, dot_list, index_a, index_b, lerp):
    if bone == None or bone_end == None:
        return
    if dot_list[index_a][3] < 0.2 or dot_list[index_b][3] < 0.2:
        pass
    v = np.subtract(bone_end.world.position, bone.world.position)
    b = [dot_list[index_b][0], dot_list[index_b][1], dot_list[index_b][2]]
    a = [dot_list[index_a][0], dot_list[index_a][1], dot_list[index_a][2]]
    v1 = np.subtract(b, a)
    t, r, s = la.mat_decompose(bone.world.matrix)
    start = bone.world.rotation
    v = norm(v)
    v1 = norm(v1)
    end = la.quat_mul(la.quat_from_vecs(v, v1), bone.world.rotation)
    result = quat_slerp(start, end, lerp)
    new_matrix = la.mat_compose(t, result, s)
    local_matrix = np.linalg.inv(bone.parent.world.matrix) @ new_matrix
    t, r, s = la.mat_decompose(local_matrix)
    bone.local.rotation = r

def rig_with_pos(bone, bone_end, posa,posb):
    if bone == None or bone_end == None:
        return
    v = np.subtract(bone_end.world.position, bone.world.position)
    v1 = np.subtract(posb, posa)
    t, r, s = la.mat_decompose(bone.world.matrix)
    start = bone.world.rotation
    v = norm(v)
    v1 = norm(v1)
    end = la.quat_mul(la.quat_from_vecs(v, v1), bone.world.rotation)
    result = quat_slerp(start, end, 0.9)
    new_matrix = la.mat_compose(t, result, s)
    local_matrix = np.linalg.inv(bone.parent.world.matrix) @ new_matrix
    t, r, s = la.mat_decompose(local_matrix)
    bone.local.rotation = r

def pre_rig_bone(boneStart, boneEnd, dot_list, index_a, index_b, lerp):
    if dot_list[index_a][3] < 0.2 or dot_list[index_b][3] < 0.2:
        pass
    t1,r1,s1 = la.mat_decompose(boneStart.worldMatrix)
    t2,r2,s2 = la.mat_decompose(boneEnd.worldMatrix)
    v = np.subtract(t2, t1)
    b = [dot_list[index_b][0], dot_list[index_b][1], dot_list[index_b][2]]
    a = [dot_list[index_a][0], dot_list[index_a][1], dot_list[index_a][2]]
    v1 = np.subtract(b, a)
    start = r1
    v = norm(v)
    v1 = norm(v1)
    end = la.quat_mul(la.quat_from_vecs(v, v1),r1)
    result = quat_slerp(start, end, lerp)
    new_matrix = la.mat_compose(t1, result, s1)
    boneStart.setWorldMatrix(new_matrix)


def clamp(v, vmin, vmax):
    return max(min(vmax, v), vmin)


def justify_angle(bone_name, r):
    x, y, z = la.quat_to_euler(r)
    return la.quat_from_euler((x, y, z))


def print_qua(angle):
    print("{:.2f}".format(angle[0]), "{:.2f}".format(angle[1]), "{:.2f}".format(angle[2]), "{:.2f}".format(angle[3]))


def draw_point(a):
    geometry = gfx.sphere_geometry(1, phi_length=np.pi * 2)
    material = gfx.MeshPhongMaterial(color=(0.8, 0, 0), flat_shading=True)
    wobject = gfx.Mesh(geometry, material)
    wobject.local.position = a
    return wobject


def mid_point(a, b):
    if len(a) > 3:
        return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, (a[3] + b[3]) / 2]
    else:
        return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]


def rig_skeleton(skeleton_helper, landmark_list, matrix_list, lerp=1):
    if landmark_list is None or len(landmark_list) != 33:
        return
    dot_list = [item for item in landmark_list]
    ## add some key points
    ## 33  neck
    dot_list.append(mid_point(dot_list[11], dot_list[12]))
    ## 34  hip
    dot_list.append(mid_point(dot_list[23], dot_list[24]))
    ## 35 left middle index
    dot_list.append(mid_point(dot_list[17], dot_list[19]))
    ## 36 right middle index
    dot_list.append(mid_point(dot_list[18], dot_list[20]))
    ## 37 head
    dot_list.append(mid_point(dot_list[7], dot_list[8]))
    ## 38 mouse
    dot_list.append(mid_point(dot_list[9], dot_list[10]))
    ## 39 spine1
    dot_list.append(mid_point(dot_list[33], dot_list[34]))
    ## 40 spine
    dot_list.append(mid_point(dot_list[34], dot_list[39]))
    ## 41 spine2
    dot_list.append(mid_point(dot_list[33], dot_list[39]))
    ## 42 center of eyes
    dot_list.append(mid_point(dot_list[2], dot_list[5]))

    _, hips = find_bone(skeleton_helper, "Hips")
    _, spine = find_bone(skeleton_helper, "Spine")
    _, spine1 = find_bone(skeleton_helper, "Spine1")
    _, spine2 = find_bone(skeleton_helper, "Spine2")
    _, left_up_leg = find_bone(skeleton_helper, "LeftUpLeg")
    _, left_leg = find_bone(skeleton_helper, "LeftLeg")
    _, left_foot = find_bone(skeleton_helper, "LeftFoot")
    _, right_up_leg = find_bone(skeleton_helper, "RightUpLeg")
    _, right_leg = find_bone(skeleton_helper, "RightLeg")
    _, right_foot = find_bone(skeleton_helper, "RightFoot")
    _, left_shoulder = find_bone(skeleton_helper, "LeftShoulder")
    _, left_arm = find_bone(skeleton_helper, "LeftArm")
    _, left_fore_arm = find_bone(skeleton_helper, "LeftForeArm")
    _, left_hand = find_bone(skeleton_helper, "LeftHand")
    _, right_shoulder = find_bone(skeleton_helper, "RightShoulder")
    _, right_arm = find_bone(skeleton_helper, "RightArm")
    _, right_fore_arm = find_bone(skeleton_helper, "RightForeArm")
    _, right_hand = find_bone(skeleton_helper, "RightHand")
    _, left_hand_middle = find_bone(skeleton_helper, "LeftHandMiddle1")
    _, left_hand_thumb1 = find_bone(skeleton_helper, "LeftHandThumb1")
    _, left_hand_thumb2 = find_bone(skeleton_helper, "LeftHandThumb2")
    _, right_hand_middle = find_bone(skeleton_helper, "RightHandMiddle1")
    _, right_hand_thumb1 = find_bone(skeleton_helper, "RightHandThumb1")
    _, right_hand_thumb2 = find_bone(skeleton_helper, "RightHandThumb2")
    _, left_toe_base = find_bone(skeleton_helper, "LeftToeBase")
    _, right_toe_base = find_bone(skeleton_helper, "RightToeBase")
    _, neck = find_bone(skeleton_helper, "Neck")
    _, head = find_bone(skeleton_helper, "Head")
    _, head_end = find_bone(skeleton_helper, "HeadTop_End")

    c = [dot_list[24][0], dot_list[24][1], dot_list[24][2]]
    a = [dot_list[12][0], dot_list[12][1], dot_list[12][2]]
    b = [dot_list[23][0], dot_list[23][1], dot_list[23][2]]
    d = [dot_list[11][0], dot_list[11][1], dot_list[11][2]]
    e = [dot_list[39][0], dot_list[39][1], dot_list[39][2]]
    v1 = get_norm_vect(b, e, d)
    v2 = get_norm_vect(a, e, c)
    v3 = get_norm_vect(c, e, b)
    v4 = get_norm_vect(d, e, a)
    v5 = mid_point(mid_point(v1, v2), mid_point(v3, v4))

    rot = la.quat_from_vecs((0, 0, 1), v5)

    mat_hips = matrix_list[0]
    mat_hips = mat_hips @ la.mat_from_quat(rot)
    t2, r2, s2 = la.mat_decompose(mat_hips)
    mat_spine = matrix_list[1]
    mat_spine = mat_hips @ mat_spine
    t, r, s = la.mat_decompose(mat_spine)
    mat_spine1 = matrix_list[2]
    mat_spine1 = mat_hips @ matrix_list[1] @ mat_spine1
    t1, r1, s1 = la.mat_decompose(mat_spine1)

    v1 = np.subtract(t1, t)
    v2 = np.subtract(dot_list[39], dot_list[40])
    v2 = (v2[0], v2[1], v2[2])
    r = la.quat_mul(la.quat_from_vecs(v1, v2), r)
    spine_local = la.quat_mul(la.quat_inv(r2), r)
    hips.world.rotation = la.quat_mul(rot, spine_local)

    rig_bone(spine, spine1, dot_list, 40, 39, lerp)
    rig_bone(spine1, spine2, dot_list, 39, 41, lerp)
    rig_bone(spine2, neck, dot_list, 41, 33, lerp)
    rig_bone(neck, head, dot_list, 33, 37, lerp)
    rig_bone(head, head_end, dot_list, 38, 42, lerp)
    rig_bone(right_shoulder, right_arm, dot_list, 33, 12, lerp)
    rig_bone(right_arm, right_fore_arm, dot_list, 12, 14, lerp)
    rig_bone(right_fore_arm, right_hand, dot_list, 14, 16, lerp)
    rig_bone(right_hand, right_hand_middle, dot_list, 16, 36, lerp)
    rig_bone(right_hand_thumb1, right_hand_thumb2, dot_list, 16, 22, lerp)
    rig_bone(left_shoulder, left_arm, dot_list, 33, 11, lerp)
    rig_bone(left_arm, left_fore_arm, dot_list, 11, 13, lerp)
    rig_bone(left_fore_arm, left_hand, dot_list, 13, 15, lerp)
    rig_bone(left_hand, left_hand_middle, dot_list, 15, 35, lerp)
    rig_bone(left_hand_thumb1, left_hand_thumb2, dot_list, 15, 21, lerp)
    rig_bone(right_up_leg, right_leg, dot_list, 24, 26, lerp)
    rig_bone(right_leg, right_foot, dot_list, 26, 28, lerp)
    rig_bone(right_foot, right_toe_base, dot_list, 28, 32, lerp)
    rig_bone(left_up_leg, left_leg, dot_list, 23, 25, lerp)
    rig_bone(left_leg, left_foot, dot_list, 25, 27, lerp)
    rig_bone(left_foot, left_toe_base, dot_list, 27, 31, lerp)

def clone_skeleton(skeleton_helper):
    _, hips = find_bone(skeleton_helper, "Hips")
    def cloneBone(bone):
        clone_bone = CloneBone()
        clone_bone.worldMatrix = bone.world.matrix
        clone_bone.localMatrix = bone.local.matrix
        if "mixamorig:" in bone.name:
            clone_bone.name = bone.name[10:]
        for item in bone.children:
            child = cloneBone(item)
            child.parent = clone_bone
            clone_bone.children.append(child)
        return clone_bone
    return cloneBone(hips)




def pre_rig_skeleton(rootBone, landmark_list, matrix_list, lerp=1):
    if landmark_list is None or len(landmark_list) != 33:
        return
    dot_list = [item for item in landmark_list]
    ## add some key points
    ## 33  neck
    dot_list.append(mid_point(dot_list[11], dot_list[12]))
    ## 34  hip
    dot_list.append(mid_point(dot_list[23], dot_list[24]))
    ## 35 left middle index
    dot_list.append(mid_point(dot_list[17], dot_list[19]))
    ## 36 right middle index
    dot_list.append(mid_point(dot_list[18], dot_list[20]))
    ## 37 head
    dot_list.append(mid_point(dot_list[7], dot_list[8]))
    ## 38 mouse
    dot_list.append(mid_point(dot_list[9], dot_list[10]))
    ## 39 spine1
    dot_list.append(mid_point(dot_list[33], dot_list[34]))
    ## 40 spine
    dot_list.append(mid_point(dot_list[34], dot_list[39]))
    ## 41 spine2
    dot_list.append(mid_point(dot_list[33], dot_list[39]))
    ## 42 center of eyes
    dot_list.append(mid_point(dot_list[2], dot_list[5]))
    c = [dot_list[24][0], dot_list[24][1], dot_list[24][2]]
    a = [dot_list[12][0], dot_list[12][1], dot_list[12][2]]
    b = [dot_list[23][0], dot_list[23][1], dot_list[23][2]]
    d = [dot_list[11][0], dot_list[11][1], dot_list[11][2]]
    e = [dot_list[39][0], dot_list[39][1], dot_list[39][2]]
    v1 = get_norm_vect(b, e, d)
    v2 = get_norm_vect(a, e, c)
    v3 = get_norm_vect(c, e, b)
    v4 = get_norm_vect(d, e, a)
    v5 = mid_point(mid_point(v1, v2), mid_point(v3, v4))

    rot = la.quat_from_vecs((0, 0, 1), v5)

    mat_hips = matrix_list[0]
    mat_hips = mat_hips @ la.mat_from_quat(rot)
    t2, r2, s2 = la.mat_decompose(mat_hips)
    mat_spine = matrix_list[1]
    mat_spine = mat_hips @ mat_spine
    t, r, s = la.mat_decompose(mat_spine)
    mat_spine1 = matrix_list[2]
    mat_spine1 = mat_hips @ matrix_list[1] @ mat_spine1
    t1, r1, s1 = la.mat_decompose(mat_spine1)

    v1 = np.subtract(t1, t)
    v2 = np.subtract(dot_list[39], dot_list[40])
    v2 = (v2[0], v2[1], v2[2])
    r = la.quat_mul(la.quat_from_vecs(v1, v2), r)
    spine_local = la.quat_mul(la.quat_inv(r2), r)

    r2 = la.quat_mul(rot, spine_local)

    rootBone.setWorldMatrix(la.mat_compose(t2,r2,s2))

    spine = rootBone.findByName("Spine")
    spine1 =rootBone.findByName("Spine1")
    spine2 = rootBone.findByName("Spine2")
    left_up_leg = rootBone.findByName("LeftUpLeg")
    left_leg = rootBone.findByName("LeftLeg")
    left_foot = rootBone.findByName("LeftFoot")
    right_up_leg = rootBone.findByName("RightUpLeg")
    right_leg = rootBone.findByName("RightLeg")
    right_foot = rootBone.findByName("RightFoot")
    left_shoulder = rootBone.findByName("LeftShoulder")
    left_arm =rootBone.findByName("LeftArm")
    left_fore_arm = rootBone.findByName("LeftForeArm")
    left_hand = rootBone.findByName("LeftHand")
    right_shoulder = rootBone.findByName("RightShoulder")
    right_arm = rootBone.findByName("RightArm")
    right_fore_arm =rootBone.findByName("RightForeArm")
    right_hand = rootBone.findByName("RightHand")
    left_toe_base = rootBone.findByName("LeftToeBase")
    right_toe_base = rootBone.findByName("RightToeBase")
    neck = rootBone.findByName("Neck")
    head = rootBone.findByName("Head")
    head_end = rootBone.findByName("HeadTop_End")
    pre_rig_bone(spine, spine1, dot_list, 40, 39, lerp)
    pre_rig_bone(spine1, spine2, dot_list, 39, 41, lerp)
    pre_rig_bone(spine2, neck, dot_list, 41, 33, lerp)
    pre_rig_bone(neck, head, dot_list, 33, 37, lerp)
    pre_rig_bone(head, head_end, dot_list, 38, 42, lerp)
    pre_rig_bone(right_shoulder, right_arm, dot_list, 33, 12, lerp)
    pre_rig_bone(right_arm, right_fore_arm, dot_list, 12, 14, lerp)
    pre_rig_bone(right_fore_arm, right_hand, dot_list, 14, 16, lerp)
    pre_rig_bone(left_shoulder, left_arm, dot_list, 33, 11, lerp)
    pre_rig_bone(left_arm, left_fore_arm, dot_list, 11, 13, lerp)
    pre_rig_bone(left_fore_arm, left_hand, dot_list, 13, 15, lerp)
    pre_rig_bone(right_up_leg, right_leg, dot_list, 24, 26, lerp)
    pre_rig_bone(right_leg, right_foot, dot_list, 26, 28, lerp)
    pre_rig_bone(right_foot, right_toe_base, dot_list, 28, 32, lerp)
    pre_rig_bone(left_up_leg, left_leg, dot_list, 23, 25, lerp)
    pre_rig_bone(left_leg, left_foot, dot_list, 25, 27, lerp)
    pre_rig_bone(left_foot, left_toe_base, dot_list, 27, 31, lerp)

def build_matrix_clone(rootBone):
    hips =rootBone.findByName("Hips")
    spine = rootBone.findByName("Spine")
    spine1 = rootBone.findByName("Spine1")
    spine2 = rootBone.findByName("Spine2")
    neck = rootBone.findByName("Neck")
    head = rootBone.findByName("Head")
    right_shoulder =rootBone.findByName("RightShoulder")
    right_arm = rootBone.findByName("RightArm")
    right_fore_arm = rootBone.findByName("RightForeArm")
    right_hand =rootBone.findByName("RightHand")
    left_shoulder = rootBone.findByName("LeftShoulder")
    left_arm = rootBone.findByName("LeftArm")
    left_fore_arm =rootBone.findByName("LeftForeArm")
    left_hand = rootBone.findByName("LeftHand")
    right_up_leg = rootBone.findByName("RightUpLeg")
    right_leg = rootBone.findByName("RightLeg")
    right_foot = rootBone.findByName("RightFoot")
    left_up_leg = rootBone.findByName("LeftUpLeg")
    left_leg = rootBone.findByName("LeftLeg")
    left_foot = rootBone.findByName("LeftFoot")
    list=[]
    list.append(hips.getWorldRotation())
    list.append(spine.getLocalRotation())
    list.append(spine1.getLocalRotation())
    list.append(spine2.getLocalRotation())
    list.append(neck.getLocalRotation())
    list.append(head.getLocalRotation())
    list.append(right_shoulder.getLocalRotation())
    list.append(right_arm.getLocalRotation())
    list.append(right_fore_arm.getLocalRotation())
    list.append(right_hand.getLocalRotation())
    list.append(left_shoulder.getLocalRotation())
    list.append(left_arm.getLocalRotation())
    list.append(left_fore_arm.getLocalRotation())
    list.append(left_hand.getLocalRotation())
    list.append(right_up_leg.getLocalRotation())
    list.append(right_leg.getLocalRotation())
    list.append(right_foot.getLocalRotation())
    list.append(left_up_leg.getLocalRotation())
    list.append(left_leg.getLocalRotation())
    list.append(left_foot.getLocalRotation())


    list = np.array(list).flatten()
    return list


def build_matrix(skeleton_helper):
    _, hips = find_bone(skeleton_helper, "Hips")
    _, spine = find_bone(skeleton_helper, "Spine")
    _, spine1 = find_bone(skeleton_helper, "Spine1")
    _, spine2 = find_bone(skeleton_helper, "Spine2")
    _, neck = find_bone(skeleton_helper, "Neck")
    _, head = find_bone(skeleton_helper, "Head")
    _, right_shoulder = find_bone(skeleton_helper, "RightShoulder")
    _, right_arm = find_bone(skeleton_helper, "RightArm")
    _, right_fore_arm = find_bone(skeleton_helper, "RightForeArm")
    _, right_hand = find_bone(skeleton_helper, "RightHand")
    _, left_shoulder = find_bone(skeleton_helper, "LeftShoulder")
    _, left_arm = find_bone(skeleton_helper, "LeftArm")
    _, left_fore_arm = find_bone(skeleton_helper, "LeftForeArm")
    _, left_hand = find_bone(skeleton_helper, "LeftHand")
    _, right_up_leg = find_bone(skeleton_helper, "RightUpLeg")
    _, right_leg = find_bone(skeleton_helper, "RightLeg")
    _, right_foot = find_bone(skeleton_helper, "RightFoot")
    _, left_up_leg = find_bone(skeleton_helper, "LeftUpLeg")
    _, left_leg = find_bone(skeleton_helper, "LeftLeg")
    _, left_foot = find_bone(skeleton_helper, "LeftFoot")
    list=[]
    list.append(hips.world.rotation)
    list.append(spine.local.rotation)
    list.append(spine1.local.rotation)
    list.append(spine2.local.rotation)
    list.append(neck.local.rotation)
    list.append(head.local.rotation)
    list.append(right_shoulder.local.rotation)
    list.append(right_arm.local.rotation)
    list.append(right_fore_arm.local.rotation)
    list.append(right_hand.local.rotation)
    list.append(left_shoulder.local.rotation)
    list.append(left_arm.local.rotation)
    list.append(left_fore_arm.local.rotation)
    list.append(left_hand.local.rotation)
    list.append(right_up_leg.local.rotation)
    list.append(right_leg.local.rotation)
    list.append(right_foot.local.rotation)
    list.append(left_up_leg.local.rotation)
    list.append(left_leg.local.rotation)
    list.append(left_foot.local.rotation)


    list = np.array(list).flatten()
    return list

def use_matrix(skeleton_helper,matrix):
    _, hips = find_bone(skeleton_helper, "Hips")
    _, spine = find_bone(skeleton_helper, "Spine")
    _, spine1 = find_bone(skeleton_helper, "Spine1")
    _, spine2 = find_bone(skeleton_helper, "Spine2")
    _, neck = find_bone(skeleton_helper, "Neck")
    _, head = find_bone(skeleton_helper, "Head")
    _, right_shoulder = find_bone(skeleton_helper, "RightShoulder")
    _, right_arm = find_bone(skeleton_helper, "RightArm")
    _, right_fore_arm = find_bone(skeleton_helper, "RightForeArm")
    _, right_hand = find_bone(skeleton_helper, "RightHand")
    _, left_shoulder = find_bone(skeleton_helper, "LeftShoulder")
    _, left_arm = find_bone(skeleton_helper, "LeftArm")
    _, left_fore_arm = find_bone(skeleton_helper, "LeftForeArm")
    _, left_hand = find_bone(skeleton_helper, "LeftHand")
    _, right_up_leg = find_bone(skeleton_helper, "RightUpLeg")
    _, right_leg = find_bone(skeleton_helper, "RightLeg")
    _, right_foot = find_bone(skeleton_helper, "RightFoot")
    _, left_up_leg = find_bone(skeleton_helper, "LeftUpLeg")
    _, left_leg = find_bone(skeleton_helper, "LeftLeg")
    _, left_foot = find_bone(skeleton_helper, "LeftFoot")
    #hips.world.rotation = matrix[0:4]
    spine.local.rotation = matrix[4:8]
    spine1.local.rotation = matrix[8:12]
    spine2.local.rotation = matrix[12:16]
    neck.local.rotation = matrix[16:20]
    head.local.rotation = matrix[20:24]
    right_shoulder.local.rotation = matrix[24:28]
    right_arm.local.rotation = matrix[28:32]
    right_fore_arm.local.rotation = matrix[32:36]
    right_hand.local.rotation = matrix[36:40]
    left_shoulder.local.rotation = matrix[40:44]
    left_arm.local.rotation = matrix[44:48]
    left_fore_arm.local.rotation = matrix[48:52]
    left_hand.local.rotation = matrix[52:56]
    right_up_leg.local.rotation = matrix[56:60]
    right_leg.local.rotation = matrix[60:64]
    right_foot.local.rotation = matrix[64:68]
    left_up_leg.local.rotation = matrix[68:72]
    left_leg.local.rotation = matrix[72:76]
    left_foot.local.rotation = matrix[76:80]




class CloneBone():
    def __init__(self):
        self.localMatrix = None
        self.worldMatrix = None
        self.children=[]
        self.name=''
        self.parent = None
    def setWorldMatrix(self,matrix):
        self.worldMatrix = matrix
        for item in self.children:
            item.setWorldMatrix(self.worldMatrix@item.localMatrix)

    def getWorldRotation(self):
        t,r,s = la.mat_decompose(self.worldMatrix)
        return r

    def getLocalRotation(self):
        if self.parent:
            self.localMatrix = np.linalg.inv(self.parent.worldMatrix) @ self.worldMatrix
            t,r,s = la.mat_decompose(self.localMatrix)
            return r
        return self.getWorldRotation()
    def findByName(self,name):
        if self.name==name:
            return self
        for item in self.children:
            result = item.findByName(name)
            if result:
                return result

def print_bone(bone):
    print(bone.name)
    if bone.parent:
        print(f"parent {bone.parent.name}")
    for item in bone.children:
        print_bone(item)

class Landmark():
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    visibility: Optional[float] = None
    presence: Optional[float] = None


class SmoothLandmarks():
    def __init__(self):
        self._frameSets = [];
        self._smoothFrame = [];

    def deal(self, landmarks):
        self._frameSets.append(landmarks)

        def add(m, n):
            return m + n

        if len(self._frameSets) == 8:
            for index in range(33):
                x = list(map(lambda m: m[index].x, self._frameSets))
                y = list(map(lambda m: m[index].y, self._frameSets))
                z = list(map(lambda m: m[index].z, self._frameSets))
                v = list(map(lambda m: m[index].visibility, self._frameSets))

                x.sort()
                y.sort()
                z.sort()
                v.sort()
                x = x[2:6]
                y = y[2:6]
                z = z[2:6]
                v = v[2:6]

                x = functools.reduce(add, x) / len(x)
                y = functools.reduce(add, y) / len(y)
                z = functools.reduce(add, z) / len(z)
                v = functools.reduce(add, v) / len(v)
                ld = Landmark()
                ld.x = x
                ld.y = y
                ld.z = z
                ld.visibility = v
                if len(self._smoothFrame) == 33:
                    self._smoothFrame[index] = ld
                else:
                    self._smoothFrame.append(ld)
            self._frameSets.pop(0)
        if len(self._smoothFrame) > 0:
            return self._smoothFrame
