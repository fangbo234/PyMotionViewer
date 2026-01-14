import math
import pylinalg as la
import numpy as np
from pyquaternion import Quaternion


def norm(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return v / length


def get_mid_vector(v1, v2):
    p1 = norm(v1)
    p2 = norm(v2)
    p3 = p2 - p1
    length = math.sqrt(p3[0] * p3[0] + p3[1] * p3[1] + p3[2] * p3[2])
    return norm(p1 + norm(p3) * (length * 0.5))


def get_quat_from_vector_angle(v, angle):
    unit_v = norm(v)
    s = math.sin(angle / 2)
    x = unit_v[0] * s;
    y = unit_v[1] * s;
    z = unit_v[2] * s;
    w = math.cos(angle / 2)
    return (x, y, z, w)

def look_at_rotation(source,target):
    return la.quat_from_vecs(source,target)


# 求ab ac构成的平面的法向量
def get_plan_reference_up(a, b, c):
    ba = np.subtract(np.array(a), np.array(b))
    bc = np.subtract(np.array(c), np.array(b))
    return np.cross(ba, bc)

def get_norm_vect(a,b,c):
    ref = get_plan_reference_up(a, b, c)
    return ref
def get_orientation_vector(a,b,c):
    center = np.add(np.add(a,b),c)/3
    ref = get_plan_reference_up(a,b,c)
    return center,np.subtract(ref,center)


def vect_mul_quer(q, v):
    v1 = la.vec_transform_quat(v,q)
    return v1

def quat_slerp(q1,q2,t):
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    new_q1 = Quaternion(w=w1, x=x1, y=y1, z=z1)
    new_q2 = Quaternion(w=w2, x=x2, y=y2, z=z2)
    result = Quaternion.slerp(new_q1, new_q2, amount=t)
    return (result.x,result.y,result.z,result.w)


def get_yaw_pitch_roll(q):
    x1, y1, z1, w1 = q
    new_q = Quaternion(w=w1, x=x1, y=y1, z=z1)
    yaw, pitch, roll = new_q.yaw_pitch_roll
    return (yaw,pitch,roll)



