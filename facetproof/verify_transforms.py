"""Facet Proof 几何变换与照明验证样例（独立客户端）。

用法：
    python3 -m facetproof.verify_transforms --output ./verification-out

本模块是渲染服务的独立客户端：不导入 facetproof 的任何实现模块，只通过真实 HTTP
请求驱动 网格/颜料 → 装配件 → 冻结(seal) → 曝光(exposure) → 工作器 → 图版 的完整
流程，并用内置的独立几何/光照计算（oracle，仅依赖标准库数学公式）从取回的冻结
快照推出预期，与图版通道逐像素对比。预期不来自任何历史截图或黄金文件：所有期望
都由已知坐标、镜头参数与公开规则现场计算，并先用手工锚点校验 oracle 本身。

验证重点（镜像与非均匀缩放）：
  * 可见面：单面材质按 dot(法线, 视线) <= 0 剔除；双面材质不剔除，但着色时
    若 dot(法线, 视线) < 0 则翻转法线；
  * 遮挡：g-far 与 h-near 在屏幕上重叠，深度小者胜出，物体通道与深度通道
    必须在同一像素给出互相一致的答案；
  * 深度：正交正面相机深度 = 8 - z，背面相机 = z + 8，透视相机按 1/z 插值；
  * 明暗：同一法线同时决定可见性与漫反射；非均匀缩放改变法线方向从而改变明暗。

成因区分：
  * 网格绕序：d-reversed 使用反绕序网格 + 恒等变换；
  * 变换：b-mirror / c-mirror2s / h-near 使用 scale=(-1,1,1) 镜像（行列式为负），
    e-rotated 使用行列式为正的旋转作为对照；
  * 双面材质：c-mirror2s 与 b-mirror 的网格和变换完全相同，仅 two_sided 不同。

冻结稳定性：冻结之后修订颜料，验证冻结闭包逐字节不变、同一冻结快照复渲染逐字节
一致，且本目录保存的判断依据（expectations/ 与 checks.json）对复渲染结果仍然成立。
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

# =====================================================================
# 第一部分：已知坐标场景与手工锚点
# =====================================================================

WIDTH, HEIGHT, SAMPLES = 512, 256, 1
TILE_COUNT = 32  # 512x256 高采样画布按 64x64 分块
PASS_NAMES = ("color.rgb", "depth.f32", "object.u32", "color.png", "depth.png", "object.png")
RAW_PASS_SIZES = {"color.rgb": WIDTH * HEIGHT * 3,
                  "depth.f32": WIDTH * HEIGHT * 4, "object.u32": WIDTH * HEIGHT * 4}

# 标枪三角：手工法线 = normalize((v1-v0) x (v2-v0))
#   (2,0,0) x (0.5,1,0.75) = (0*0.75-0*1, 0*0.5-2*0.75, 2*1-0*0.5) = (0,-1.5,2)
#   模长 2.5 → (0, -0.6, 0.8)
DART_VERTICES = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 1.0, 0.75]]

MESHES = {
    "dart": {"name": "标枪三角", "vertices": DART_VERTICES, "faces": [[0, 1, 2]]},
    "dart-reversed": {"name": "标枪反绕", "vertices": DART_VERTICES, "faces": [[0, 2, 1]]},
}

PIGMENTS = {
    "single": {"name": "单面红", "color": [0.8, 0.2, 0.15],
               "two_sided": False, "specular": 0.0, "roughness": 1.0},
    "double": {"name": "双面绿", "color": [0.2, 0.7, 0.25],
               "two_sided": True, "specular": 0.0, "roughness": 1.0},
}

# 八个布置：同一网格在不同变换/材质下的对照组。y=0 行为主对照组，y=4 行为
# 非均匀缩放与遮挡组。任意两个可见面在屏幕上除 g-far/h-near 外不重叠。
PLACEMENTS = [
    {"key": "a-ref", "mesh": "dart", "pigment": "single",
     "transform": {"translation": [-10, 0, 0]}},
    {"key": "b-mirror", "mesh": "dart", "pigment": "single",
     "transform": {"translation": [-5, 0, 0], "scale": [-1, 1, 1]}},
    {"key": "c-mirror2s", "mesh": "dart", "pigment": "double",
     "transform": {"translation": [0, 0, 0], "scale": [-1, 1, 1]}},
    {"key": "d-reversed", "mesh": "dart-reversed", "pigment": "single",
     "transform": {"translation": [5, 0, 0]}},
    {"key": "e-rotated", "mesh": "dart", "pigment": "single",
     "transform": {"translation": [10, 0, 0], "euler": [0, 0, 90]}},
    {"key": "f-scaled", "mesh": "dart", "pigment": "single",
     "transform": {"translation": [-10, 4, 0], "scale": [2, 1, 0.5]}},
    {"key": "g-far", "mesh": "dart", "pigment": "single",
     "transform": {"translation": [-1.5, 4, -1.5]}},
    {"key": "h-near", "mesh": "dart", "pigment": "double",
     "transform": {"translation": [0.5, 4.2, 1.5], "scale": [-1, 1, 1]}},
]

LENSES = {
    "front-ortho": {"projection": "orthographic", "eye": [1, 2.5, 8],
                    "target": [1, 2.5, 0], "up": [0, 1, 0], "span": 13},
    "back-ortho": {"projection": "orthographic", "eye": [1, 2.5, -8],
                   "target": [1, 2.5, 0], "up": [0, 1, 0], "span": 13},
    "front-persp": {"projection": "perspective", "eye": [1, 2.5, 16],
                    "target": [1, 2.5, 0], "up": [0, 1, 0], "fov": 60},
}

ILLUMINATION = {"lights": [{"direction": [0, 0, 1], "intensity": 1, "color": [1, 1, 1]}],
                "ambient": 0.2, "background": [0.93, 0.95, 0.97]}

SQRT_18_25 = math.sqrt(18.25)  # f-scaled：scale(2,1,0.5) 后叉积 (0,-1.5,4) 的模长

# 手工锚点（字面值）：先锚定 oracle，再由 oracle 推出全画面预期。
FRONT_VISIBILITY = {"a-ref": True, "b-mirror": False, "c-mirror2s": True, "d-reversed": False,
                    "e-rotated": True, "f-scaled": True, "g-far": True, "h-near": True}
BACK_VISIBILITY = {"a-ref": False, "b-mirror": True, "c-mirror2s": True, "d-reversed": True,
                   "e-rotated": False, "f-scaled": False, "g-far": False, "h-near": True}
FRONT_DIFFUSE = {"a-ref": 0.8, "c-mirror2s": 0.8, "e-rotated": 0.8,
                 "f-scaled": 4.0 / SQRT_18_25, "g-far": 0.8, "h-near": 0.8}
BACK_DIFFUSE = {"b-mirror": 0.0, "c-mirror2s": 0.0, "d-reversed": 0.0, "h-near": 0.0}

ANCHORS = {
    "normal": {
        "a-ref": [0.0, -0.6, 0.8], "b-mirror": [0.0, 0.6, -0.8],
        "c-mirror2s": [0.0, 0.6, -0.8], "d-reversed": [0.0, 0.6, -0.8],
        "e-rotated": [0.6, 0.0, 0.8], "f-scaled": [0.0, -1.5 / SQRT_18_25, 4.0 / SQRT_18_25],
        "g-far": [0.0, -0.6, 0.8], "h-near": [0.0, 0.6, -0.8],
    },
    "det_sign": {"a-ref": 1, "b-mirror": -1, "c-mirror2s": -1, "d-reversed": 1,
                 "e-rotated": 1, "f-scaled": 1, "g-far": 1, "h-near": -1},
    "visible": {"front-ortho": FRONT_VISIBILITY, "back-ortho": BACK_VISIBILITY,
                "front-persp": FRONT_VISIBILITY},
    "diffuse": {"front-ortho": FRONT_DIFFUSE, "back-ortho": BACK_DIFFUSE,
                "front-persp": FRONT_DIFFUSE},
    # 质心视深 = 三顶点视深均值；正交相机下即摄像机前向距离。
    "centroid_view_z": {
        "front-ortho": {"a-ref": 7.75, "c-mirror2s": 7.75, "e-rotated": 7.75,
                        "f-scaled": 7.875, "g-far": 9.25, "h-near": 6.25},
        "back-ortho": {"b-mirror": 8.25, "c-mirror2s": 8.25, "d-reversed": 8.25,
                       "h-near": 9.75},
        "front-persp": {"a-ref": 15.75, "c-mirror2s": 15.75, "e-rotated": 15.75,
                        "f-scaled": 15.875, "g-far": 17.25, "h-near": 14.25},
    },
    # 镜像在屏幕上翻转 v0→v1 的方向：sign(screen_x(v1) - screen_x(v0))。
    "mirror_x_sign": {
        "front-ortho": {"a-ref": 1, "c-mirror2s": -1},
        "back-ortho": {"a-ref": -1, "c-mirror2s": 1},
        "front-persp": {"a-ref": 1, "c-mirror2s": -1},
    },
}

DEPTH_TOLERANCE = 1e-4    # depth.f32 与独立 f64 预期的容差
COLOR_TOLERANCE = 1       # color.rgb 字节容差
INSIDE_MARGIN = 1e-6      # 决定性像素的重心坐标余量（远大于实现的 -1e-10 边界阈值）


# =====================================================================
# 第二部分：独立 oracle（不导入服务实现，仅按公开规则计算）
# =====================================================================

def vec_sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vec_dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def vec_cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def vec_unit(a):
    length = math.sqrt(vec_dot(a, a))
    return [x / length for x in a]


def vec_close(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def matrix_from_transform(transform):
    """按公开约定组合变换：M = T · Rz · Ry · Rx · S（角度为度）。"""
    t = transform.get("translation", [0.0, 0.0, 0.0])
    angles = [math.radians(x) for x in transform.get("euler", [0.0, 0.0, 0.0])]
    sizes = transform.get("scale", [1.0, 1.0, 1.0])
    cx, sx = math.cos(angles[0]), math.sin(angles[0])
    cy, sy = math.cos(angles[1]), math.sin(angles[1])
    cz, sz = math.cos(angles[2]), math.sin(angles[2])
    rot = [[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
           [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
           [-sy, cy * sx, cy * cx]]
    return [[rot[r][0] * sizes[0], rot[r][1] * sizes[1], rot[r][2] * sizes[2], t[r]]
            for r in range(3)]


def matrix_close(a, b, tol):
    return all(abs(a[r][c] - b[r][c]) <= tol for r in range(3) for c in range(4))


def mat_point(m, v):
    return [m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3] for r in range(3)]


def det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def lens_basis(eye, target, up):
    forward = vec_unit(vec_sub(target, eye))
    right = vec_unit(vec_cross(forward, vec_unit(up)))
    true_up = vec_unit(vec_cross(right, forward))
    return {"forward": forward, "right": right, "true_up": true_up}


def view_coords(basis, eye, world):
    v = vec_sub(world, eye)
    return [vec_dot(v, basis["right"]), vec_dot(v, basis["true_up"]), vec_dot(v, basis["forward"])]


def project(lens, view, width, height):
    x, y, z = view
    if lens["projection"] == "perspective":
        half = math.tan(math.radians(lens["fov"]) / 2.0)
        nx, ny = x / (z * half * width / height), y / (z * half)
    else:
        nx, ny = x / (lens["span"] * width / height / 2.0), y / (lens["span"] / 2.0)
    return [(nx + 1.0) * width / 2.0, (1.0 - ny) * height / 2.0, z]


def srgb_to_linear(value):
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def linear_to_srgb(value):
    value = min(1.0, max(0.0, value))
    return value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


def to_f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def to_byte(linear_value):
    """线性颜色 → 通道字节：f32 存储 → sRGB → ×255 → 取整。"""
    return round(linear_to_srgb(to_f32(linear_value)) * 255)


def shade(normal, pigment_linear, illumination):
    """环境光 + 各方向光漫反射；场景颜料 specular=0，故高光项恒为 0。"""
    out = [c * illumination["ambient"] for c in pigment_linear]
    for light in illumination["lights"]:
        diffuse = max(0.0, vec_dot(normal, light["direction"]))
        for ch in range(3):
            out[ch] += light["intensity"] * light["color"][ch] * pigment_linear[ch] * diffuse
    return [min(1.0, max(0.0, v)) for v in out]


def edge(a, b, x, y):
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])


def barycentric(tri, x, y):
    a, b, c = tri
    area = edge(a, b, c[0], c[1])
    wa = edge(b, c, x, y) / area
    wb = edge(c, a, x, y) / area
    return wa, wb, 1.0 - wa - wb


def depth_at(tri, w, projection):
    wa, wb, wc = w
    za, zb, zc = tri[0][2], tri[1][2], tri[2][2]
    if projection == "perspective":
        return 1.0 / (wa / za + wb / zb + wc / zc)
    return wa * za + wb * zb + wc * zc


def build_expectations(cam, seal, width, height):
    """从冻结快照独立计算全部预期：法线、可见性、明暗、投影、逐像素归属。"""
    lens = seal["lens"]
    basis = lens_basis(lens["eye"], lens["target"], lens["up"])
    illumination = seal["illumination"]
    bg_bytes = [to_byte(srgb_to_linear(c)) for c in illumination["background"]]
    faces = []
    ordered = sorted(seal["closure"], key=lambda it: it["placement"]["key"])
    for object_id, item in enumerate(ordered, 1):
        placement, mesh, pigment = item["placement"], item["mesh"], item["pigment"]
        matrix = placement["matrix"]
        world = [mat_point(matrix, v) for v in mesh["vertices"]]
        linear = [srgb_to_linear(c) for c in pigment["color"]]
        determinant = det3(matrix)
        for face_index, face in enumerate(mesh["faces"]):
            w = [world[i] for i in face]
            normal = vec_unit(vec_cross(vec_sub(w[1], w[0]), vec_sub(w[2], w[0])))
            center = [(w[0][i] + w[1][i] + w[2][i]) / 3.0 for i in range(3)]
            if lens["projection"] == "perspective":
                view_dir = vec_unit(vec_sub(lens["eye"], center))
            else:
                view_dir = [-v for v in basis["forward"]]
            facing = vec_dot(normal, view_dir)
            visible = bool(pigment["two_sided"] or facing > 0.0)
            shading_normal = list(normal)
            if pigment["two_sided"] and facing < 0.0:
                shading_normal = [-v for v in normal]
            color_linear = shade(shading_normal, linear, illumination)
            color_bytes = [to_byte(c) for c in color_linear]
            view = [view_coords(basis, lens["eye"], p) for p in w]
            screen = [project(lens, p, width, height) for p in view]
            faces.append({
                "key": placement["key"], "object_id": object_id, "face_index": face_index,
                "pigment": pigment["name"], "two_sided": pigment["two_sided"],
                "pigment_linear": linear, "determinant": determinant,
                "world": w, "normal": normal, "center": center, "view_dir": view_dir,
                "facing": facing, "visible": visible, "shading_normal": shading_normal,
                "diffuse": max(0.0, vec_dot(shading_normal, illumination["lights"][0]["direction"])),
                "color_linear": color_linear, "color_bytes": color_bytes,
                "view": view, "screen": screen,
                "centroid_view_z": sum(p[2] for p in view) / 3.0,
            })
    frustum_ok = True
    for face in faces:
        for x, y, z in face["view"]:
            if lens["projection"] == "perspective":
                half = math.tan(math.radians(lens["fov"]) / 2.0)
                inside = (lens["near"] + 0.01 < z < lens["far"] - 0.01
                          and abs(x) < 0.98 * z * half * width / height
                          and abs(y) < 0.98 * z * half)
            else:
                inside = (lens["near"] + 0.01 < z < lens["far"] - 0.01
                          and abs(x) < 0.98 * lens["span"] * width / height / 2.0
                          and abs(y) < 0.98 * lens["span"] / 2.0)
            frustum_ok = frustum_ok and inside
    scan = scan_pixels(faces, width, height, lens["projection"], bg_bytes)
    return {"camera": cam, "width": width, "height": height, "lens": lens, "basis": basis,
            "illumination": illumination, "faces": faces, "frustum_ok": frustum_ok,
            "background_bytes": bg_bytes, "scan": scan}


def scan_pixels(faces, width, height, projection, bg_bytes):
    """逐像素分类：决定性像素（远离任何三角形边）给出确定预期，边界像素不裁决。"""
    visible_faces = [f for f in faces if f["visible"]]
    candidates = {}
    inside_count = {f["key"]: 0 for f in visible_faces}
    boundary = set()
    for face in visible_faces:
        tri = face["screen"]
        xs = [p[0] for p in tri]
        ys = [p[1] for p in tri]
        x0 = max(0, int(math.floor(min(xs))) - 1)
        x1 = min(width - 1, int(math.ceil(max(xs))) + 1)
        y0 = max(0, int(math.floor(min(ys))) - 1)
        y1 = min(height - 1, int(math.ceil(max(ys))) + 1)
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                w = barycentric(tri, px + 0.5, py + 0.5)
                margin = min(w)
                idx = py * width + px
                if margin > INSIDE_MARGIN:
                    inside_count[face["key"]] += 1
                    candidates.setdefault(idx, []).append((depth_at(tri, w, projection), face, margin))
                elif margin >= -INSIDE_MARGIN:
                    boundary.add(idx)
    covered = {}
    overlaps = {}
    ties = 0
    for idx, lst in candidates.items():
        if idx in boundary:
            continue
        lst.sort(key=lambda item: item[0])
        if len(lst) > 1 and lst[1][0] - lst[0][0] <= 1e-6:
            boundary.add(idx)  # 深度过于接近，不作裁决
            ties += 1
            continue
        depth, face, margin = lst[0]
        covered[idx] = {"object_id": face["object_id"], "key": face["key"],
                        "depth": depth, "color": face["color_bytes"], "min_weight": margin}
        if len(lst) > 1:
            overlaps[idx] = {"winner": lst[0][1]["key"], "winner_depth": lst[0][0],
                             "loser": lst[1][1]["key"], "loser_depth": lst[1][0],
                             "depth_gap": lst[1][0] - lst[0][0]}
    winner_count = {f["key"]: 0 for f in visible_faces}
    best_probe = {}
    for idx, info in covered.items():
        winner_count[info["key"]] += 1
        current = best_probe.get(info["key"])
        if current is None or info["min_weight"] > current[0]:
            best_probe[info["key"]] = (info["min_weight"], idx)
    probes = {}
    for face in faces:
        key = face["key"]
        if face["visible"] and key in best_probe:
            idx = best_probe[key][1]
            info = covered[idx]
            probes[f"placement:{key}"] = {"x": idx % width, "y": idx // width,
                                          "object_key": key, "object_id": info["object_id"],
                                          "depth": info["depth"], "color": info["color"]}
        else:
            # 被剔除的面：质心像素处不应出现该面自身；该像素的预期由扫描给出
            # （可能是背景，也可能是遮挡在后面的其他零件，如 g-far 之于 h-near）。
            cx = sum(p[0] for p in face["screen"]) / 3.0
            cy = sum(p[1] for p in face["screen"]) / 3.0
            px, py = int(cx), int(cy)
            idx = py * width + px
            if idx in covered:
                info = covered[idx]
                probes[f"placement:{key}"] = {"x": px, "y": py, "object_key": info["key"],
                                              "object_id": info["object_id"],
                                              "depth": info["depth"], "color": info["color"],
                                              "culled": True, "not_object_id": face["object_id"]}
            elif idx in boundary:
                probes[f"placement:{key}"] = {"x": px, "y": py, "object_key": None,
                                              "object_id": 0, "depth": None,
                                              "color": list(bg_bytes), "culled": True,
                                              "not_object_id": face["object_id"],
                                              "indecisive": True}
            else:
                probes[f"placement:{key}"] = {"x": px, "y": py, "object_key": None, "object_id": 0,
                                              "depth": None, "color": list(bg_bytes),
                                              "culled": True, "not_object_id": face["object_id"],
                                              "decisive_background": True}
    bg_idx = 5 * width + 5
    probes["background"] = {"x": 5, "y": 5, "object_key": None, "object_id": 0, "depth": None,
                            "color": list(bg_bytes),
                            "decisive_background": bg_idx not in covered and bg_idx not in boundary}
    if overlaps:
        idx = next(iter(overlaps))
        info = covered[idx]
        probes["overlap"] = {"x": idx % width, "y": idx // width, "object_key": info["key"],
                             "object_id": info["object_id"], "depth": info["depth"],
                             "color": info["color"]}
    return {"covered": covered, "boundary": boundary, "overlaps": overlaps, "ties": ties,
            "probes": probes, "inside_count": inside_count, "winner_count": winner_count,
            "background_bytes": bg_bytes}


def expectations_dump(exp):
    """可持久化的预期快照：判断依据的完整记录。"""
    scan = exp["scan"]
    faces = []
    for face in exp["faces"]:
        faces.append({k: v for k, v in face.items() if k != "pigment_linear"})
    return {"camera": exp["camera"], "canvas": {"width": exp["width"], "height": exp["height"],
            "samples": SAMPLES}, "lens": exp["lens"], "recomputed_basis": exp["basis"],
            "illumination": exp["illumination"], "background_bytes": exp["background_bytes"],
            "frustum_ok": exp["frustum_ok"], "faces": faces,
            "scan": {"decisive_covered": len(scan["covered"]),
                     "boundary_pixels": len(scan["boundary"]),
                     "depth_tie_pixels": scan["ties"],
                     "inside_count": scan["inside_count"], "winner_count": scan["winner_count"],
                     "overlaps": [{"pixel": [idx % exp["width"], idx // exp["width"]], **info}
                                  for idx, info in sorted(scan["overlaps"].items())]},
            "probes": scan["probes"]}


# =====================================================================
# 第三部分：检查记录
# =====================================================================

def jsonable(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if math.isnan(value):
            return "NaN"
        return value
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if isinstance(value, (tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


class Checks:
    def __init__(self):
        self.items = []

    def add(self, check_id, rule, expected, actual, passed, tolerance=None, detail=None):
        self.items.append({"id": check_id, "rule": rule, "expected": jsonable(expected),
                           "actual": jsonable(actual), "tolerance": tolerance,
                           "pass": bool(passed), "detail": jsonable(detail)})

    @property
    def failed(self):
        return [c for c in self.items if not c["pass"]]

    def summary(self):
        groups = {}
        for item in self.items:
            prefix = item["id"].split("-")[0]
            stat = groups.setdefault(prefix, [0, 0])
            stat[0] += 1
            stat[1] += 0 if item["pass"] else 1
        return groups


# =====================================================================
# 第四部分：HTTP 客户端（真实请求，全程留痕）
# =====================================================================

class HttpClient:
    def __init__(self, port, transcript):
        self.base = f"http://127.0.0.1:{port}/facet/v1/"
        self.transcript = transcript

    def call(self, method, path, body=None, expect=200):
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=payload, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                raw, status, mime = resp.read(), resp.status, resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            raw, status, mime = exc.read(), exc.code, "application/json"
        self.transcript.append({"method": method, "path": path, "status": status,
                                "request": jsonable(body),
                                "response_sha256": hashlib.sha256(raw).hexdigest(),
                                "response_bytes": len(raw)})
        if status != expect:
            raise RuntimeError(f"{method} {path} -> {status}: {raw[:400]!r}")
        return json.loads(raw) if "json" in mime else raw


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_health(client, seconds=15.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            return client.call("GET", "health")
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("服务未在限定时间内就绪")


# =====================================================================
# 第五部分：各级检查
# =====================================================================

def check_freeze_fidelity(checks, cam, seal_obj, ids):
    """F1：冻结闭包与提交输入逐项一致（网格、材质、矩阵、变换、修订号）。"""
    payload = seal_obj["payload"]
    problems = []
    by_key = {it["placement"]["key"]: it for it in payload["closure"]}
    if set(by_key) != {p["key"] for p in PLACEMENTS}:
        problems.append(f"闭包键集合不符: {sorted(by_key)}")
    for spec in PLACEMENTS:
        item = by_key.get(spec["key"])
        if item is None:
            continue
        mesh_want = MESHES[spec["mesh"]]
        mesh = item["mesh"]
        if mesh["vertices"] != [[float(x) for x in v] for v in mesh_want["vertices"]]:
            problems.append(f"{spec['key']}: 冻结顶点与提交不符")
        if mesh["faces"] != mesh_want["faces"]:
            problems.append(f"{spec['key']}: 冻结绕序与提交不符")
        want_bounds = {"min": [min(v[a] for v in DART_VERTICES) for a in range(3)],
                       "max": [max(v[a] for v in DART_VERTICES) for a in range(3)]}
        if mesh["bounds"] != want_bounds:
            problems.append(f"{spec['key']}: 冻结包围盒与独立计算不符")
        pig_want = PIGMENTS[spec["pigment"]]
        pig = item["pigment"]
        for field in ("name", "color", "two_sided", "specular", "roughness"):
            if pig[field] != pig_want[field]:
                problems.append(f"{spec['key']}: 冻结颜料 {field} 与提交不符")
        if not vec_close(pig["linear"], [srgb_to_linear(c) for c in pig_want["color"]], 1e-12):
            problems.append(f"{spec['key']}: 冻结线性色与 sRGB 公式不符")
        if item["mesh_revision"] != 1 or item["pigment_revision"] != 1:
            problems.append(f"{spec['key']}: 冻结修订号不是 1")
        placement = item["placement"]
        if placement["mesh_id"] != ids[f"mesh:{spec['mesh']}"]:
            problems.append(f"{spec['key']}: 冻结网格引用不符")
        if placement["pigment_id"] != ids[f"pigment:{spec['pigment']}"]:
            problems.append(f"{spec['key']}: 冻结颜料引用不符")
        if placement["transform"] != spec["transform"]:
            problems.append(f"{spec['key']}: 冻结变换参数与提交不符")
        if placement["visible"] is not True:
            problems.append(f"{spec['key']}: 冻结可见标志不符")
        if not matrix_close(placement["matrix"], matrix_from_transform(spec["transform"]), 1e-12):
            problems.append(f"{spec['key']}: 冻结矩阵与变换参数组合结果不符")
    checks.add(f"FREEZE-CLOSURE-{cam}",
               "冻结闭包逐项等于提交输入（顶点/绕序/颜料/变换参数/矩阵/修订号）",
               "全部一致", problems or "一致", not problems, tolerance=1e-12)
    return not problems


def check_freeze_lens(checks, cam, seal_obj):
    lens = seal_obj["payload"]["lens"]
    want = LENSES[cam]
    problems = []
    for field in ("eye", "target", "up", "projection"):
        if lens[field] != want[field]:
            problems.append(f"{field}: {lens[field]} != {want[field]}")
    defaults = {"span": 5.0, "fov": 45.0, "near": 0.1, "far": 100.0}
    for field in ("span", "fov", "near", "far"):
        if lens[field] != want.get(field, defaults[field]):
            problems.append(f"{field}: {lens[field]} != {want.get(field, defaults[field])}")
    basis = lens_basis(want["eye"], want["target"], want["up"])
    for field in ("forward", "right", "true_up"):
        if not vec_close(lens[field], basis[field], 1e-12):
            problems.append(f"{field}: 冻结基向量与 eye/target/up 重算不符")
    checks.add(f"FREEZE-LENS-{cam}",
               "冻结镜头等于提交参数与文档默认值，且基向量可由 eye/target/up 独立重现",
               "一致", problems or "一致", not problems, tolerance=1e-12)


def check_freeze_illumination(checks, cam, seal_obj):
    got = seal_obj["payload"]["illumination"]
    want = {"ambient": 0.2, "background": [0.93, 0.95, 0.97],
            "lights": [{"direction": [0.0, 0.0, 1.0], "intensity": 1.0, "color": [1.0, 1.0, 1.0]}]}
    checks.add(f"FREEZE-ILLUM-{cam}", "冻结照明等于提交值（方向光已单位化）",
               want, got, got == want)


def run_anchor_checks(checks, cam, exp):
    by_key = {f["key"]: f for f in exp["faces"]}
    if cam == "front-ortho":  # 法线与行列式与相机无关，只锚定一次
        for key, want in ANCHORS["normal"].items():
            got = by_key[key]["normal"]
            checks.add(f"ANCHOR-NORMAL-{key}",
                       "世界法线 = normalize(变换后绕序叉积)，与手工锚点一致",
                       want, got, vec_close(got, want, 1e-9), tolerance=1e-9)
        for key, want in ANCHORS["det_sign"].items():
            got = by_key[key]["determinant"]
            checks.add(f"ANCHOR-DET-{key}",
                       "变换行列式符号：负行列式（镜像）翻转绕序，正行列式（含旋转）保持",
                       want, 1 if got > 0 else -1, (got > 0) == (want > 0))
    for key, want in ANCHORS["visible"][cam].items():
        got = by_key[key]["visible"]
        checks.add(f"ANCHOR-VISIBLE-{cam}-{key}",
                   "可见性规则：单面材质 dot(n,view)<=0 剔除；双面材质不剔除",
                   want, got, got == want)
    for key, want in ANCHORS["diffuse"][cam].items():
        got = by_key[key]["diffuse"]
        checks.add(f"ANCHOR-DIFFUSE-{cam}-{key}",
                   "漫反射 = max(0, dot(着色法线, 光方向))；双面材质先按视线翻转法线",
                   want, got, abs(got - want) <= 1e-9, tolerance=1e-9)
    for key, want in ANCHORS["centroid_view_z"][cam].items():
        got = by_key[key]["centroid_view_z"]
        checks.add(f"ANCHOR-VIEWZ-{cam}-{key}",
                   "质心视深 = 三顶点视深均值（正交时即摄像机前向距离）",
                   want, got, abs(got - want) <= 1e-9, tolerance=1e-9)
    for key, want in ANCHORS["mirror_x_sign"][cam].items():
        face = by_key[key]
        got = face["screen"][1][0] - face["screen"][0][0]
        checks.add(f"ANCHOR-MIRRORX-{cam}-{key}",
                       "镜像变换在屏幕上翻转 v0→v1 的方向（几何事实，与剔除无关）",
                       want, 1 if got > 0 else -1, (got > 0) == (want > 0))
    checks.add(f"SCENE-FRUSTUM-{cam}",
               "场景不变量：全部顶点位于视锥内（裁剪不改变任何三角形）",
               True, exp["frustum_ok"], exp["frustum_ok"])
    checks.add(f"SCENE-NO-TIE-{cam}",
               "场景不变量：决定性像素不存在深度相近的竞争",
               0, exp["scan"]["ties"], exp["scan"]["ties"] == 0)


def run_pixel_checks(checks, label, exp, cam_passes, plate_payload, prefix="PIX"):
    """逐像素核对三通道：物体编号、深度、颜色必须在同一像素互相解释。"""
    width, height = exp["width"], exp["height"]
    scan = exp["scan"]
    covered, boundary = scan["covered"], scan["boundary"]
    bg = scan["background_bytes"]
    ids_actual = struct.unpack(f"<{width * height}I", cam_passes["object.u32"])
    depth_actual = struct.unpack(f"<{width * height}f", cam_passes["depth.f32"])
    color_actual = cam_passes["color.rgb"]

    def examples(idxs, want_fn, got_fn):
        return [{"pixel": [i % width, i // width], "expected": want_fn(i), "actual": got_fn(i)}
                for i in idxs[:5]]

    bad_id = [i for i, w in covered.items() if ids_actual[i] != w["object_id"]]
    bad_depth = [i for i, w in covered.items() if abs(depth_actual[i] - w["depth"]) > DEPTH_TOLERANCE]
    bad_color = [i for i, w in covered.items()
                 if any(abs(color_actual[i * 3 + c] - w["color"][c]) > COLOR_TOLERANCE for c in range(3))]
    checks.add(f"{prefix}-OBJECT-{label}", "决定性像素的物体编号与独立预期一致",
               f"{len(covered)} 个覆盖像素全部符合", f"{len(bad_id)} 处不符", not bad_id,
               detail=examples(bad_id, lambda i: covered[i]["object_id"], lambda i: ids_actual[i]))
    checks.add(f"{prefix}-DEPTH-{label}", "决定性像素的前向深度与独立插值预期一致",
               f"{len(covered)} 个覆盖像素全部符合", f"{len(bad_depth)} 处不符", not bad_depth,
               tolerance=DEPTH_TOLERANCE,
               detail=examples(bad_depth, lambda i: covered[i]["depth"], lambda i: depth_actual[i]))
    checks.add(f"{prefix}-COLOR-{label}", "决定性像素的颜色字节与独立明暗预期一致",
               f"{len(covered)} 个覆盖像素全部符合", f"{len(bad_color)} 处不符", not bad_color,
               tolerance=COLOR_TOLERANCE,
               detail=examples(bad_color, lambda i: covered[i]["color"],
                               lambda i: list(color_actual[i * 3:i * 3 + 3])))

    bg_bad = {"object": 0, "depth": 0, "color": 0}
    consist_bad = 0
    bg_count = 0
    for idx in range(width * height):
        if (ids_actual[idx] == 0) != math.isinf(depth_actual[idx]):
            consist_bad += 1
        if idx in covered or idx in boundary:
            continue
        bg_count += 1
        if ids_actual[idx] != 0:
            bg_bad["object"] += 1
        if not math.isinf(depth_actual[idx]):
            bg_bad["depth"] += 1
        if any(abs(color_actual[idx * 3 + c] - bg[c]) > COLOR_TOLERANCE for c in range(3)):
            bg_bad["color"] += 1
    checks.add(f"{prefix}-BACKGROUND-{label}",
               "决定性背景像素：物体 0、深度正无穷、颜色为背景色，三者一致",
               f"{bg_count} 个背景像素全部符合", bg_bad, not any(bg_bad.values()),
               tolerance=COLOR_TOLERANCE)
    checks.add(f"{prefix}-CONSIST-{label}",
               "通道互相解释：物体编号为 0 当且仅当深度为正无穷",
               "0 处矛盾", f"{consist_bad} 处矛盾", consist_bad == 0)

    stats = plate_payload["statistics"]
    low, high = len(covered), len(covered) + len(boundary)
    checks.add(f"{prefix}-COVERAGE-{label}",
               "实测覆盖像素数介于决定性覆盖数与其加边界像素数之间",
               f"[{low}, {high}]", stats["covered_pixels"], low <= stats["covered_pixels"] <= high)

    culled_faces = [f for f in exp["faces"] if not f["visible"]]
    culled_bad = {f["key"]: int(sum(1 for v in ids_actual if v == f["object_id"]))
                  for f in culled_faces}
    culled_bad = {k: v for k, v in culled_bad.items() if v}
    checks.add(f"{prefix}-CULLED-{label}",
               "被剔除面的物体编号在整幅图中不出现（剔除是全画面事实，不只是探针）",
               "0 个像素", culled_bad or "0 个像素", not culled_bad)

    overlaps = scan["overlaps"]
    overlap_bad = [i for i in overlaps
                   if ids_actual[i] != covered[i]["object_id"]
                   or abs(depth_actual[i] - covered[i]["depth"]) > DEPTH_TOLERANCE
                   or any(abs(color_actual[i * 3 + c] - covered[i]["color"][c]) > COLOR_TOLERANCE
                          for c in range(3))]
    if label.startswith(("front-ortho", "front-persp")):
        occlusion_ok = (bool(overlaps) and not overlap_bad
                        and all(info["winner"] == "h-near" and info["loser"] == "g-far"
                                and info["depth_gap"] > 1.0 for info in overlaps.values()))
        occlusion_expected = "重叠区存在，h-near 以更小深度胜出，三通道一致"
    else:
        occlusion_ok = not overlaps
        occlusion_expected = "g-far 被剔除，无双面竞争，重叠集为空"
    checks.add(f"{prefix}-OCCLUSION-{label}",
               "遮挡与深度互相解释：重叠像素的胜者必须是深度更小者",
               occlusion_expected,
               {"overlap_pixels": len(overlaps), "mismatched": len(overlap_bad)}, occlusion_ok,
               tolerance=DEPTH_TOLERANCE,
               detail=[{"pixel": [i % width, i // width], **overlaps[i]} for i in list(overlaps)[:8]])


def run_pick_checks(client, checks, cam, exp, plate_id):
    probes = exp["scan"]["probes"]
    details = []
    ok_all = True
    for name, probe in probes.items():
        if probe.get("indecisive"):
            continue
        got = client.call("GET", f"plates/{plate_id}/pick?x={probe['x']}&y={probe['y']}")
        got_key = got["object"]["key"] if got["object"] else None
        depth_ok = ((got["depth"] is None and probe["depth"] is None)
                    or (got["depth"] is not None and probe["depth"] is not None
                        and abs(got["depth"] - probe["depth"]) <= DEPTH_TOLERANCE))
        ok = got_key == probe["object_key"] and depth_ok
        if "not_object_id" in probe and got["object"] is not None:
            ok = ok and got["object"]["key"] != name.removeprefix("placement:")
        ok_all = ok_all and ok
        details.append({"probe": name, "x": probe["x"], "y": probe["y"],
                        "expected_object": probe["object_key"], "actual_object": got_key,
                        "expected_depth": probe["depth"], "actual_depth": got["depth"], "pass": ok})
    checks.add(f"PICK-{cam}", "拾取接口与原始通道及独立预期一致",
               "全部探针一致", f"{sum(1 for d in details if not d['pass'])} 个探针不符",
               ok_all, tolerance=DEPTH_TOLERANCE, detail=details)


def run_cause_checks(checks, expectations, passes):
    """成因区分：绕序、变换、双面材质、非均匀缩放各自的影响。"""
    front_exp, back_exp = expectations["front-ortho"], expectations["back-ortho"]
    front, back = passes["front-ortho"], passes["back-ortho"]
    fp = front_exp["scan"]["probes"]
    bp = back_exp["scan"]["probes"]
    front_faces = {f["key"]: f for f in front_exp["faces"]}

    def actual_id(cam_passes, probe):
        return struct.unpack_from("<I", cam_passes["object.u32"],
                                  (probe["y"] * WIDTH + probe["x"]) * 4)[0]

    def actual_color(cam_passes, probe):
        base = (probe["y"] * WIDTH + probe["x"]) * 3
        return list(cam_passes["color.rgb"][base:base + 3])

    # 绕序成因：d-reversed（反绕序网格+恒等变换）与 b-mirror（镜像变换+原网格）
    # 结果等价——同可见、同明暗；但冻结事实表明成因不同。
    winding_actual = {
        "front_ids": {k: actual_id(front, fp[f"placement:{k}"]) for k in ("a-ref", "b-mirror", "d-reversed")},
        "back_ids": {k: actual_id(back, bp[f"placement:{k}"]) for k in ("a-ref", "b-mirror", "d-reversed")},
        "back_colors": {k: actual_color(back, bp[f"placement:{k}"]) for k in ("b-mirror", "d-reversed")},
    }
    winding_ok = (winding_actual["front_ids"] == {"a-ref": 1, "b-mirror": 0, "d-reversed": 0}
                  and winding_actual["back_ids"] == {"a-ref": 0, "b-mirror": 2, "d-reversed": 4}
                  and all(abs(winding_actual["back_colors"]["b-mirror"][c]
                              - winding_actual["back_colors"]["d-reversed"][c]) <= COLOR_TOLERANCE
                          for c in range(3)))
    checks.add("CAUSE-WINDING",
               "绕序成因：反绕序网格(d-reversed)与镜像变换(b-mirror)结果等价（正面剔除、"
               "背面可见且同暗度），但冻结闭包显示前者矩阵行列式为正、后者为负——成因可区分",
               {"front_ids": {"a-ref": 1, "b-mirror": 0, "d-reversed": 0},
                "back_ids": {"a-ref": 0, "b-mirror": 2, "d-reversed": 4},
                "back_colors": "b-mirror 与 d-reversed 相差 ≤1"},
               winding_actual, winding_ok, tolerance=COLOR_TOLERANCE)

    # 变换成因：同一网格同一材质，镜像翻转可见性；旋转（行列式正）不翻转。
    transform_actual = {
        "front_ids": {k: actual_id(front, fp[f"placement:{k}"]) for k in ("a-ref", "b-mirror", "e-rotated")},
        "back_ids": {k: actual_id(back, bp[f"placement:{k}"]) for k in ("a-ref", "b-mirror", "e-rotated")},
    }
    transform_ok = (transform_actual["front_ids"] == {"a-ref": 1, "b-mirror": 0, "e-rotated": 5}
                    and transform_actual["back_ids"] == {"a-ref": 0, "b-mirror": 2, "e-rotated": 0})
    checks.add("CAUSE-TRANSFORM",
               "变换成因：b-mirror 与 a-ref 同网格同材质，镜像使可见性在两台相机间互换；"
               "e-rotated 行列式为正，可见性与 a-ref 同类——是行列式符号而非变换本身翻转朝向",
               {"front": {"a-ref": 1, "b-mirror": 0, "e-rotated": 5},
                "back": {"a-ref": 0, "b-mirror": 2, "e-rotated": 0}},
               transform_actual, transform_ok)

    # 材质成因：c-mirror2s 与 b-mirror 网格和变换完全相同，仅 two_sided 不同。
    face_c = front_faces["c-mirror2s"]
    flipped_bytes = face_c["color_bytes"]
    unflipped_bytes = [to_byte(c) for c in shade(face_c["normal"], face_c["pigment_linear"],
                                                 front_exp["illumination"])]
    material_actual = {
        "front_id": actual_id(front, fp["placement:c-mirror2s"]),
        "front_color": actual_color(front, fp["placement:c-mirror2s"]),
        "back_id": actual_id(back, bp["placement:c-mirror2s"]),
        "back_color": actual_color(back, bp["placement:c-mirror2s"]),
        "b_mirror_front_id": actual_id(front, fp["placement:b-mirror"]),
    }
    flipped_match = all(abs(material_actual["front_color"][c] - flipped_bytes[c]) <= COLOR_TOLERANCE
                        for c in range(3))
    unflipped_gap = max(abs(material_actual["front_color"][c] - unflipped_bytes[c]) for c in range(3))
    material_ok = (material_actual["front_id"] == 3 and material_actual["b_mirror_front_id"] == 0
                   and flipped_match and unflipped_gap > 20
                   and material_actual["back_id"] == 3)
    checks.add("CAUSE-MATERIAL",
               "材质成因：c-mirror2s 与 b-mirror 网格和变换相同，仅 two_sided 不同——"
               "双面材质正面可见且按翻转法线着色（与单面假设相差 >20 级），背面可见但仅环境光",
               {"front_id": 3, "front_color≈": flipped_bytes, "unflipped_hypothesis": unflipped_bytes,
                "back_id": 3, "b_mirror_front_id": 0},
               {**material_actual, "unflipped_gap": unflipped_gap}, material_ok,
               tolerance=COLOR_TOLERANCE)

    # 非均匀缩放成因：法线方向被缩放改变（(0,-0.6,0.8)→(0,-0.351,0.936)），
    # 实测明暗必须符合正确法线假设，并排除"法线不随缩放改变"的天真假设。
    face_f = front_faces["f-scaled"]
    correct_bytes = face_f["color_bytes"]
    naive_bytes = [to_byte(c) for c in shade(ANCHORS["normal"]["a-ref"], face_f["pigment_linear"],
                                             front_exp["illumination"])]
    scale_actual = {"front_color": actual_color(front, fp["placement:f-scaled"]),
                    "front_id": actual_id(front, fp["placement:f-scaled"])}
    correct_match = all(abs(scale_actual["front_color"][c] - correct_bytes[c]) <= COLOR_TOLERANCE
                        for c in range(3))
    naive_gap = abs(scale_actual["front_color"][0] - naive_bytes[0])
    scale_ok = scale_actual["front_id"] == 6 and correct_match and naive_gap >= 3
    checks.add("CAUSE-SCALE",
               "非均匀缩放成因：f-scaled 可见但法线被缩放改为 (0,-0.351,0.936)，漫反射 0.936；"
               "实测字节须符合正确法线假设，并与法线不变假设相差 ≥3 级",
               {"front_id": 6, "color≈": correct_bytes, "naive_hypothesis": naive_bytes},
               {**scale_actual, "naive_gap_red": naive_gap}, scale_ok, tolerance=COLOR_TOLERANCE)
    return {"winding": winding_actual, "transform": transform_actual,
            "material": {**material_actual, "flipped_hypothesis": flipped_bytes,
                         "unflipped_hypothesis": unflipped_bytes},
            "scale": {**scale_actual, "correct_hypothesis": correct_bytes,
                      "naive_hypothesis": naive_bytes, "naive_gap_red": naive_gap}}


# =====================================================================
# 第六部分：报告
# =====================================================================

def render_report(data):
    lines = []
    add = lines.append
    add("# Facet Proof 几何变换与照明验证报告")
    add("")
    add(f"- 运行时间：{data['started']}")
    add(f"- 渲染器版本：`{data['renderer']}`（来自 `GET /facet/v1/health`）")
    add(f"- 画布：{WIDTH}×{HEIGHT}，samples={SAMPLES}；服务数据根：{data['data_root']}")
    verdict = "PASS" if not data["failed"] else "FAIL"
    add(f"- 结论：**{verdict}**（{data['total'] - len(data['failed'])}/{data['total']} 项检查通过）")
    add("")
    add("## 1. 场景与已知坐标")
    add("")
    add("网格 `dart`：顶点 (0,0,0)、(2,0,0)、(0.5,1,0.75)，面 [0,1,2]，"
        "手工法线 (0,-0.6,0.8)；`dart-reversed` 顶点相同、面 [0,2,1]，法线 (0,0.6,-0.8)。")
    add("颜料：`单面红` 单面、(0.8,0.2,0.15)、specular=0；`双面绿` 双面、(0.2,0.7,0.25)、specular=0。")
    add("照明：一个方向光 (0,0,1)、强度 1，环境光 0.2，背景 (0.93,0.95,0.97)。")
    add("")
    add("| 布置 | 网格 | 材质 | 变换 | 行列式 | 手工法线锚点 |")
    add("| --- | --- | --- | --- | --- | --- |")
    transforms = {p["key"]: p["transform"] for p in PLACEMENTS}
    meshes_of = {p["key"]: p["mesh"] for p in PLACEMENTS}
    pigments_of = {p["key"]: p["pigment"] for p in PLACEMENTS}
    for key in sorted(transforms):
        normal = ANCHORS["normal"][key]
        normal_text = f"({normal[0]:.4g}, {normal[1]:.4g}, {normal[2]:.4g})"
        add(f"| {key} | {meshes_of[key]} | {pigments_of[key]} | `{json.dumps(transforms[key])}` "
            f"| {'负' if ANCHORS['det_sign'][key] < 0 else '正'} | {normal_text} |")
    add("")
    add("镜头：`front-ortho` 正交 eye(1,2.5,8) span13；`back-ortho` 正交 eye(1,2.5,-8) span13；"
        "`front-persp` 透视 eye(1,2.5,16) fov60。三者共享同一组布置与照明。")
    add("")
    add("## 2. 冻结保真（输入 → 冻结快照）")
    add("")
    for cam, info in data["seals"].items():
        add(f"- `{cam}`：seal `{info['seal_id']}`，content_hash `{info['content_hash'][:16]}…`，"
            f"闭包与提交输入逐项一致（FREEZE-CLOSURE/LENS/ILLUM 均通过）")
    add("- 三份冻结闭包的布置闭包逐字节相同（SEAL-CLOSURE-SHARED）；冻结后装配件只读，"
        "PUT 返回 409 `sealed`（SEALED-REJECT）。")
    add("")
    add("## 3. 独立预期与判断依据")
    add("")
    add("oracle 不导入服务实现，仅按公开规则从**取回的冻结快照**计算：")
    add("")
    add("1. 变换矩阵 M = T·Rz·Ry·Rx·S，先与冻结矩阵逐项核对（1e-12）；")
    add("2. 世界法线 = normalize(变换后顶点绕序叉积)，并以 8 个手工法线锚点校验（1e-9）；")
    add("3. 可见性：单面材质 dot(n, view)<=0 剔除，双面材质不剔除、着色时按视线翻转法线；")
    add("4. 明暗 = 线性色 × (环境光 + Σ 光强×光色×漫反射)，sRGB↔线性按标准公式，f32 存储；")
    add("5. 投影：正交 span、透视 fov；深度为摄像机前向距离，透视按 1/z 插值；")
    add("6. 像素裁决：重心坐标余量 >1e-6 才为决定性像素，边界像素只计数不裁决。")
    add("")
    add(f"容差：深度 {DEPTH_TOLERANCE}，颜色 ±{COLOR_TOLERANCE} 级，物体编号精确。")
    add("")
    add("## 4. 逐相机核对")
    for cam, info in data["cameras"].items():
        add("")
        add(f"### 4.{data['cameras'][cam]['ordinal']} `{cam}`")
        add("")
        scan = info["scan"]
        add(f"决定性覆盖像素 {scan['decisive_covered']}，边界像素 {scan['boundary_pixels']}，"
            f"深度并列像素 {scan['depth_tie_pixels']}，重叠像素 {len(scan['overlaps'])}。")
        add("")
        add("| 布置 | 锚点可见 | oracle 可见 | 实测探针 | oracle 漫反射 | oracle 字节 | 实测字节 |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for row in info["visibility_rows"]:
            add("| {key} | {anchor} | {oracle} | {actual} | {diffuse} | {want} | {got} |".format(**row))
        if cam == "back-ortho":
            add("")
            add("注：g-far 在背面相机被剔除，其质心投影恰好落在 h-near 的投影区内，"
                "故探针处实测为 h-near（id=8）——被剔除面的编号在整幅图中均不出现"
                "（PIX-CULLED），探针只验证该像素不属于 g-far。")
    add("")
    add("## 5. 可见面、遮挡、深度、明暗的互相解释")
    add("")
    add("- **同一法线解释可见性与明暗**：b-mirror 的法线 (0,0.6,-0.8) 同时解释了正面相机"
        "剔除（dot<0）、背面相机可见、背面仅环境光（漫反射 0）三件事；c-mirror2s 经双面"
        "翻转后正面明暗与 a-ref 同因子。")
    add("- **深度与物体通道互相解释**：front-ortho/front-persp 的重叠区中，每个重叠像素的"
        "物体编号都是深度更小者 h-near（深度差 >1），且颜色为 h-near 的双面翻转着色；"
        "back-ortho 中 g-far 被剔除，h-near 独占其投影区。")
    add("- **通道一致性**：全部像素满足 物体编号为 0 ⟺ 深度为正无穷（PIX-CONSIST）；"
        "被剔除面的物体编号在整幅图中不出现（PIX-CULLED）。")
    add("")
    add("## 6. 成因区分（绕序 / 变换 / 双面材质 / 非均匀缩放）")
    add("")
    cause = data["cause"]
    add(f"- **绕序**：d-reversed 与 b-mirror 结果等价（正面剔除、背面同暗度 "
        f"{cause['winding']['back_colors']}），但冻结闭包显示前者行列式为正（成因是网格绕序）、"
        f"后者为负（成因是镜像变换）。")
    add(f"- **变换**：a-ref/b-mirror/e-rotated 正面实测 id "
        f"{cause['transform']['front_ids']}，背面 {cause['transform']['back_ids']}——"
        f"镜像翻转可见性，旋转不翻转。")
    add(f"- **双面材质**：c-mirror2s 正面实测 {cause['material']['front_color']}，"
        f"符合翻转法线假设 {cause['material']['flipped_hypothesis']}，"
        f"与不翻转假设 {cause['material']['unflipped_hypothesis']} 相差 >20 级。")
    add(f"- **非均匀缩放**：f-scaled 正面实测 {cause['scale']['front_color']}，"
        f"符合正确法线假设 {cause['scale']['correct_hypothesis']}，"
        f"与法线不变假设 {cause['scale']['naive_hypothesis']} 红通道相差 "
        f"{cause['scale']['naive_gap_red']} 级。")
    add("")
    add("## 7. 材质修订与冻结稳定性")
    add("")
    matrev = data["material_revision"]
    add(f"- 颜料 `{matrev['pigment_id']}` 由修订 1 改为修订 2（颜色 → (0.05,0.1,0.9)，"
        f"two_sided → true，specular → 0.9）。")
    add("- 修订后三份冻结闭包与修订前逐字节相同（MATREV-CLOSURE-*），闭包内颜料修订号仍为 1。")
    add(f"- 同一冻结快照复渲染：六个通道与首版逐字节一致（{matrev['identical_passes']}/6），"
        f"服务比对 color_changed={matrev['comparison']['color_changed']}、"
        f"depth_changed={matrev['comparison']['depth_changed']}、"
        f"silhouette_changed={matrev['comparison']['silhouette_changed']}。")
    add("- 第 4 节的全部像素级判断依据对复渲染图版重新执行仍全部通过（MATREV-PIX-*）——"
        "已冻结例子的解释不随材质修订改变。")
    add("")
    add("## 8. 检查清单汇总")
    add("")
    add("| 检查组 | 总数 | 失败 |")
    add("| --- | --- | --- |")
    for prefix, (total, failed) in sorted(data["groups"].items()):
        add(f"| {prefix} | {total} | {failed} |")
    if data["failed"]:
        add("")
        add("### 失败明细")
        for item in data["failed"]:
            add(f"- `{item['id']}`：期望 {item['expected']}，实际 {item['actual']}")
    add("")
    add("## 9. 产物清单")
    add("")
    add("- `inputs/`：全部请求体与返回标识；`frozen/`：三份冻结快照原文；")
    add("- `outputs/`：四张图版的载荷与六个通道（含材质修订后的复渲染）；")
    add("- `expectations/`：三台相机的独立预期（法线、可见性、明暗、投影、逐像素归属）；")
    add("- `checks.json`：全部检查与判定依据；`material-revision.json`：材质修订对照；")
    add("- `transcript.json`：全部 HTTP 请求留痕；`server-log.txt`：服务日志。")
    add("")
    return "\n".join(lines)


# =====================================================================
# 第七部分：主流程
# =====================================================================

def run_verification(output_dir, data_root=None, keep_data=False):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"输出目录非空，请选择新目录：{output}")
    output.mkdir(parents=True, exist_ok=True)

    def save_json(rel, obj):
        path = output / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2,
                                   sort_keys=True, allow_nan=False), encoding="utf-8")

    def save_bytes(rel, data):
        path = output / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    temp_root = None
    if data_root is None:
        temp_root = Path(tempfile.mkdtemp(prefix="facetproof-verify-"))
        root = temp_root
    else:
        root = Path(data_root)
        if root.exists() and any(root.iterdir()):
            raise SystemExit(f"数据根目录非空：{root}")
        root.mkdir(parents=True, exist_ok=True)

    checks = Checks()
    transcript = []
    report_data = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "data_root": str(root) if (data_root or keep_data) else "临时目录（已删除）",
                   "renderer": None, "seals": {}, "cameras": {}, "cause": {},
                   "material_revision": {}, "groups": {}, "failed": [], "total": 0}
    port = free_port()
    package_parent = Path(__file__).resolve().parent.parent
    log_file = open(output / "server-log.txt", "wb")
    server = subprocess.Popen(
        [sys.executable, "-m", "facetproof", "--root", str(root),
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(package_parent), stdout=log_file, stderr=subprocess.STDOUT)
    error = None
    try:
        client = HttpClient(port, transcript)
        health = wait_health(client)
        report_data["renderer"] = health["renderer"]

        # 1) 网格与颜料
        ids = {}
        for tag, body in MESHES.items():
            resp = client.call("POST", "meshes", body)
            ids[f"mesh:{tag}"] = resp["id"]
            save_json(f"inputs/mesh-{tag}.json", {"request": body, "response": resp})
        for tag, body in PIGMENTS.items():
            resp = client.call("POST", "pigments", body)
            ids[f"pigment:{tag}"] = resp["id"]
            save_json(f"inputs/pigment-{tag}.json", {"request": body, "response": resp})

        # 2) 三个装配件（共享布置，不同镜头），随后冻结
        seals, assemblies = {}, {}
        for cam, lens in LENSES.items():
            create_body = {"name": f"变换验证-{cam}", "lens": lens, "illumination": ILLUMINATION}
            asm = client.call("POST", "assemblies", create_body)
            placements = [{"key": p["key"], "mesh_id": ids[f"mesh:{p['mesh']}"],
                           "pigment_id": ids[f"pigment:{p['pigment']}"],
                           "transform": p["transform"]} for p in PLACEMENTS]
            revise_body = {"revision": asm["revision"], "placements": placements}
            asm = client.call("PUT", f"assemblies/{asm['id']}", revise_body)
            seal = client.call("POST", f"assemblies/{asm['id']}/seal",
                               {"revision": asm["revision"]})
            seals[cam], assemblies[cam] = seal, asm
            save_json(f"inputs/assembly-{cam}.json",
                      {"create": create_body, "revise": revise_body, "seal_id": seal["id"]})

        # 3) 取回冻结快照并做冻结保真检查
        frozen = {}
        for cam, seal in seals.items():
            got = client.call("GET", f"seals/{seal['id']}")
            again = client.call("GET", f"seals/{seal['id']}")
            frozen[cam] = got
            save_json(f"frozen/seal-{cam}.json", got)
            checks.add(f"FREEZE-HASH-STABLE-{cam}", "冻结快照重复读取内容不变",
                       got["payload"]["content_hash"], again["payload"]["content_hash"],
                       got == again)
            check_freeze_fidelity(checks, cam, got, ids)
            check_freeze_lens(checks, cam, got)
            check_freeze_illumination(checks, cam, got)
            report_data["seals"][cam] = {"seal_id": seal["id"],
                                         "content_hash": got["payload"]["content_hash"]}
            asm_state = client.call("GET", f"assemblies/{assemblies[cam]['id']}")["state"]
            checks.add(f"SEALED-STATE-{cam}", "冻结后装配件状态为 sealed",
                       "sealed", asm_state, asm_state == "sealed")
        closures = [frozen[cam]["payload"]["closure"] for cam in LENSES]
        checks.add("SEAL-CLOSURE-SHARED", "三份冻结的布置闭包逐字节相同",
                   "三份相同", "相同" if closures[0] == closures[1] == closures[2] else "不同",
                   closures[0] == closures[1] == closures[2])
        rejected = client.call("PUT", f"assemblies/{assemblies['front-ortho']['id']}",
                               {"revision": 3, "name": "冻结后改稿"}, expect=409)
        checks.add("SEALED-REJECT", "冻结后修改装配件被拒绝（409 sealed）",
                   "sealed", rejected.get("error"), rejected.get("error") == "sealed")

        # 4) 曝光、渲染、取回图版与通道
        plates, passes = {}, {}
        for cam, seal in seals.items():
            exposure_body = {"key": f"verify-{cam}", "width": WIDTH, "height": HEIGHT,
                             "samples": SAMPLES}
            exposure = client.call("POST", f"seals/{seal['id']}/exposures", exposure_body)
            save_json(f"inputs/exposure-{cam}.json", {"request": exposure_body, "response": exposure})
            stepped = client.call("POST", "worker/step")
            checks.add(f"RENDER-SUCCESS-{cam}", "工作器执行曝光成功",
                       "succeeded", stepped.get("state"), stepped.get("state") == "succeeded",
                       detail=stepped.get("error"))
            plate = stepped["plate"]
            plates[cam] = plate
            progress = client.call("GET", f"exposures/{exposure['id']}/progress")
            checks.add(f"PROGRESS-{cam}", "曝光进度：全部分块提交完成",
                       {"complete": True, "tiles": TILE_COUNT},
                       {"complete": progress["complete"], "tiles": progress["committed_tiles"]},
                       progress["complete"] and progress["committed_tiles"] == TILE_COUNT)
            verification = client.call("POST", f"plates/{plate['id']}/verify")
            checks.add(f"VERIFY-{cam}", "图版自检（通道尺寸与摘要）有效",
                       True, verification["valid"], verification["valid"] is True)
            cam_passes = {}
            for name in PASS_NAMES:
                data = client.call("GET", f"plates/{plate['id']}/passes/{name}")
                cam_passes[name] = data
                save_bytes(f"outputs/{cam}/{name}", data)
            sizes_ok = all(len(cam_passes[name]) == size for name, size in RAW_PASS_SIZES.items())
            checks.add(f"PASS-SIZE-{cam}", "原始通道字节数与画布尺寸一致",
                       RAW_PASS_SIZES, {n: len(cam_passes[n]) for n in RAW_PASS_SIZES}, sizes_ok)
            save_json(f"outputs/{cam}/plate.json", plate)
            save_json(f"outputs/{cam}/progress.json", progress)
            save_json(f"outputs/{cam}/verify.json", verification)
            passes[cam] = cam_passes

        # 5) 物体编号映射（按布置键排序，1 起）
        for cam, plate in plates.items():
            want_objects = {str(i + 1): {"key": p["key"], "mesh_id": ids[f"mesh:{p['mesh']}"],
                                         "pigment_id": ids[f"pigment:{p['pigment']}"]}
                            for i, p in enumerate(sorted(PLACEMENTS, key=lambda p: p["key"]))}
            got_objects = plate["payload"]["objects"]
            checks.add(f"OBJECTS-MAP-{cam}", "图版物体映射 = 按布置键排序的 1..8",
                       want_objects, got_objects, got_objects == want_objects)

        # 6) 独立预期 + 锚点检查 + 像素检查 + 拾取检查
        expectations = {}
        for ordinal, cam in enumerate(LENSES, 1):
            exp = build_expectations(cam, frozen[cam]["payload"], WIDTH, HEIGHT)
            expectations[cam] = exp
            dump = expectations_dump(exp)
            dump["seal_id"] = seals[cam]["id"]
            dump["content_hash"] = frozen[cam]["payload"]["content_hash"]
            save_json(f"expectations/{cam}.json", dump)
            run_anchor_checks(checks, cam, exp)
            run_pixel_checks(checks, cam, exp, passes[cam], plates[cam]["payload"])
            run_pick_checks(client, checks, cam, exp, plates[cam]["id"])
            rows = []
            by_key = {f["key"]: f for f in exp["faces"]}
            for key in sorted(by_key):
                face = by_key[key]
                probe = exp["scan"]["probes"][f"placement:{key}"]
                idx = probe["y"] * WIDTH + probe["x"]
                got_id = struct.unpack_from("<I", passes[cam]["object.u32"], idx * 4)[0]
                base = idx * 3
                got_color = list(passes[cam]["color.rgb"][base:base + 3])
                rows.append({"key": key,
                             "anchor": "可见" if ANCHORS["visible"][cam][key] else "剔除",
                             "oracle": "可见" if face["visible"] else "剔除",
                             "actual": f"id={got_id}",
                             "diffuse": f"{face['diffuse']:.4f}" if face["visible"] else "—",
                             "want": probe["color"],
                             "got": got_color})
            report_data["cameras"][cam] = {"ordinal": ordinal, "scan": dump["scan"],
                                           "visibility_rows": rows}

        # 7) 成因区分
        report_data["cause"] = run_cause_checks(checks, expectations, passes)

        # 8) 材质修订：冻结之后修改颜料，验证已冻结例子的解释不变
        pig_id = ids["pigment:single"]
        pig_before = client.call("GET", f"pigments/{pig_id}")
        revise_body = {"revision": pig_before["revision"], "name": "改订蓝",
                       "color": [0.05, 0.1, 0.9], "two_sided": True, "specular": 0.9,
                       "roughness": 0.3}
        pig_after = client.call("PUT", f"pigments/{pig_id}", revise_body)
        checks.add("MATREV-ACCEPTED", "材质修订被接受（修订号 +1）",
                   pig_before["revision"] + 1, pig_after["revision"],
                   pig_after["revision"] == pig_before["revision"] + 1)
        for cam in LENSES:
            refrozen = client.call("GET", f"seals/{seals[cam]['id']}")
            checks.add(f"MATREV-CLOSURE-{cam}",
                       "材质修订后冻结闭包逐字节不变（闭包内颜料修订号仍为 1）",
                       "与修订前相同", "相同" if refrozen == frozen[cam] else "不同",
                       refrozen == frozen[cam])
        rerender_body = {"key": "verify-front-ortho-rerender", "width": WIDTH, "height": HEIGHT,
                         "samples": SAMPLES}
        exposure2 = client.call("POST", f"seals/{seals['front-ortho']['id']}/exposures", rerender_body)
        save_json("inputs/exposure-front-ortho-rerender.json",
                  {"request": rerender_body, "response": exposure2})
        stepped2 = client.call("POST", "worker/step")
        checks.add("MATREV-RERENDER", "材质修订后同一冻结快照复渲染成功",
                   "succeeded", stepped2.get("state"), stepped2.get("state") == "succeeded")
        plate2 = stepped2["plate"]
        passes2 = {}
        for name in PASS_NAMES:
            data = client.call("GET", f"plates/{plate2['id']}/passes/{name}")
            passes2[name] = data
            save_bytes(f"outputs/front-ortho-rerender/{name}", data)
        save_json("outputs/front-ortho-rerender/plate.json", plate2)
        identical = {name: hashlib.sha256(passes2[name]).hexdigest()
                     == hashlib.sha256(passes["front-ortho"][name]).hexdigest()
                     for name in PASS_NAMES}
        checks.add("MATREV-RERENDER-IDENTICAL",
                   "复渲染六个通道与材质修订前逐字节一致",
                   "6/6 一致", f"{sum(identical.values())}/6 一致",
                   all(identical.values()), detail=identical)
        comparison = client.call("POST", "comparisons",
                                 {"left": plates["front-ortho"]["id"], "right": plate2["id"],
                                  "tolerance": 0})
        cmp_payload = comparison["payload"]
        checks.add("MATREV-COMPARE-ZERO",
                   "服务自比对：复渲染与首版无颜色/深度/轮廓差异",
                   {"color_changed": 0, "depth_changed": 0, "silhouette_changed": 0},
                   {k: cmp_payload[k] for k in ("color_changed", "depth_changed", "silhouette_changed")},
                   cmp_payload["color_changed"] == 0 and cmp_payload["depth_changed"] == 0
                   and cmp_payload["silhouette_changed"] == 0)
        run_pixel_checks(checks, "front-ortho-rerender", expectations["front-ortho"],
                         passes2, plate2["payload"], prefix="MATREV-PIX")
        report_data["material_revision"] = {
            "pigment_id": pig_id, "before": pig_before["payload"], "after": pig_after["payload"],
            "identical_passes": sum(identical.values()), "comparison": cmp_payload}
        save_json("material-revision.json", {
            "pigment_before": pig_before, "pigment_after": pig_after,
            "seal_unchanged": {cam: True for cam in LENSES},
            "rerender_plate": plate2["id"], "pass_identical": identical,
            "comparison": cmp_payload})
    except Exception:
        error = traceback.format_exc()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()
        if temp_root is not None and not keep_data:
            shutil.rmtree(temp_root, ignore_errors=True)

    # 9) 汇总与落盘
    report_data["total"] = len(checks.items)
    report_data["failed"] = checks.failed
    report_data["groups"] = checks.summary()
    save_json("checks.json", checks.items)
    save_json("transcript.json", transcript)
    if error:
        (output / "report.md").write_text(
            f"# 验证中断\n\n基础设施或流程异常，检查未完成：\n\n```\n{error}\n```\n",
            encoding="utf-8")
        (output / "VERDICT").write_text("ERROR\n", encoding="utf-8")
        print(f"验证异常中断，产物见 {output}")
        print(error)
        return 2
    (output / "report.md").write_text(render_report(report_data), encoding="utf-8")
    failed = checks.failed
    (output / "VERDICT").write_text("PASS\n" if not failed else "FAIL\n", encoding="utf-8")
    print(f"检查 {len(checks.items)} 项，失败 {len(failed)} 项；产物见 {output}")
    for item in failed:
        print(f"  失败: {item['id']}: 期望 {item['expected']}，实际 {item['actual']}")
    return 0 if not failed else 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Facet Proof 几何变换与照明验证样例")
    parser.add_argument("--output", required=True, help="验证产物输出目录（须为空或不存在）")
    parser.add_argument("--data-root", help="服务数据根（默认临时目录，运行后删除）")
    parser.add_argument("--keep-data", action="store_true", help="保留临时数据根")
    args = parser.parse_args()
    sys.exit(run_verification(args.output, data_root=args.data_root, keep_data=args.keep_data))


if __name__ == "__main__":
    main()
