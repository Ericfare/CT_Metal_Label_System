"""窗宽窗位、切片提取与 Dear PyGui 纹理数据。"""

from __future__ import annotations

from enum import Enum

import numpy as np
import SimpleITK as sitk


class DisplayMode(str, Enum):
    ORIGINAL = "原图"
    MASK = "掩码"
    OVERLAY = "叠加"


class ViewAxis(str, Enum):
    AXIAL = "轴位"
    CORONAL = "冠状位"
    SAGITTAL = "矢状位"


# 脑部 CT 标注窗（《标注方法.md》：脑组织约 0~50 HU，宽窗兼顾金属高/低 HU 条纹）
DEFAULT_WINDOW_LEVEL = 40.0
DEFAULT_WINDOW_WIDTH = 2500.0

# 金属掩码叠加颜色与透明度（橙色，与综合工具界面一致）
MASK_OVERLAY_COLOR = (255.0, 145.0, 0.0)
MASK_OVERLAY_ALPHA = 0.58


def ensure_scalar_image(image: sitk.Image) -> sitk.Image:
    """多通道 DICOM（如 RGB）转为单通道浮点图像。"""
    if image.GetNumberOfComponentsPerPixel() > 1:
        return sitk.VectorIndexSelectionCast(image, 0, sitk.sitkFloat32)
    return sitk.Cast(image, sitk.sitkFloat32)


def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(ensure_scalar_image(image))
    return np.asarray(arr, dtype=np.float32)


def apply_window(hu: np.ndarray, window_level: float, window_width: float) -> np.ndarray:
    low = window_level - window_width / 2.0
    high = window_level + window_width / 2.0
    if high <= low:
        high = low + 1.0
    scaled = (hu - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def extract_slice(volume: np.ndarray, axis: ViewAxis, index: int) -> np.ndarray:
    """volume 为 SimpleITK 数组顺序 (z, y, x)。"""
    depth, height, width = volume.shape
    index = int(np.clip(index, 0, {ViewAxis.AXIAL: depth - 1, ViewAxis.CORONAL: height - 1, ViewAxis.SAGITTAL: width - 1}[axis]))
    if axis == ViewAxis.AXIAL:
        return volume[index, :, :]
    if axis == ViewAxis.CORONAL:
        return volume[:, index, :]
    return volume[:, :, index]


def slice_count(volume: np.ndarray, axis: ViewAxis) -> int:
    depth, height, width = volume.shape
    if axis == ViewAxis.AXIAL:
        return depth
    if axis == ViewAxis.CORONAL:
        return height
    return width


def compose_display_slice(
    hu_slice: np.ndarray,
    mask_slice: np.ndarray | None,
    window_level: float,
    window_width: float,
    mode: DisplayMode,
) -> np.ndarray:
    gray = apply_window(hu_slice, window_level, window_width)
    if mode == DisplayMode.ORIGINAL or mask_slice is None:
        return gray

    mask_bool = mask_slice > 0
    if mode == DisplayMode.MASK:
        out = np.zeros((*gray.shape, 3), dtype=np.uint8)
        out[mask_bool] = (255, 145, 0)
        return out

    # 叠加：灰度底图 + 橙色半透明掩码
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    if not np.any(mask_bool):
        return base.astype(np.uint8)

    overlay = base.copy()
    alpha = MASK_OVERLAY_ALPHA
    r, g, b = MASK_OVERLAY_COLOR
    overlay[mask_bool, 0] = base[mask_bool, 0] * (1.0 - alpha) + r * alpha
    overlay[mask_bool, 1] = base[mask_bool, 1] * (1.0 - alpha) + g * alpha
    overlay[mask_bool, 2] = base[mask_bool, 2] * (1.0 - alpha) + b * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_mpr_crosshairs(
    image: np.ndarray,
    axis: ViewAxis,
    volume_shape: tuple[int, int, int],
    slice_index: dict[ViewAxis, int],
) -> np.ndarray:
    """在 MPR 切片上绘制另两个方向的定位十字线（黄/青）。"""
    if image.ndim == 2:
        out = np.stack([image, image, image], axis=-1).astype(np.uint8)
    else:
        out = image.copy()

    z, y, x = volume_shape
    h, w = out.shape[:2]
    if h < 2 or w < 2:
        return out

    yellow = np.array([255, 220, 0], dtype=np.uint8)
    cyan = np.array([0, 200, 255], dtype=np.uint8)

    def _row(y_pos: int, color: np.ndarray) -> None:
        r = int(np.clip(y_pos, 0, h - 1))
        out[r, :] = color

    def _col(x_pos: int, color: np.ndarray) -> None:
        c = int(np.clip(x_pos, 0, w - 1))
        out[:, c] = color

    if axis == ViewAxis.AXIAL:
        _row(int(slice_index[ViewAxis.CORONAL] * (h - 1) / max(y - 1, 1)), yellow)
        _col(int(slice_index[ViewAxis.SAGITTAL] * (w - 1) / max(x - 1, 1)), cyan)
    elif axis == ViewAxis.CORONAL:
        _row(int(slice_index[ViewAxis.AXIAL] * (h - 1) / max(z - 1, 1)), yellow)
        _col(int(slice_index[ViewAxis.SAGITTAL] * (w - 1) / max(x - 1, 1)), cyan)
    else:
        _row(int(slice_index[ViewAxis.AXIAL] * (h - 1) / max(z - 1, 1)), yellow)
        _col(int(slice_index[ViewAxis.CORONAL] * (w - 1) / max(y - 1, 1)), cyan)
    return out


def physical_slice_size(
    volume_shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    panel_key: str,
) -> tuple[float, float]:
    """切片在物理空间中的宽高 (mm)，用于保持显示宽高比。"""
    z, y, x = volume_shape
    sx, sy, sz = spacing
    if panel_key == "axial":
        return x * sx, y * sy
    if panel_key == "coronal":
        return x * sx, z * sz
    return y * sy, z * sz


def fit_panel_in_box(
    volume_shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    panel_key: str,
    max_w: int,
    max_h: int,
) -> tuple[int, int]:
    """在限定框内等比缩放，返回像素宽高。"""
    pw, ph = physical_slice_size(volume_shape, spacing, panel_key)
    if pw <= 0 or ph <= 0:
        return max(1, max_w), max(1, max_h)
    scale = min(max_w / pw, max_h / ph)
    return max(1, int(pw * scale)), max(1, int(ph * scale))


def voxel_slice_size(volume_shape: tuple[int, int, int], panel_key: str) -> tuple[int, int]:
    """切片体素维度 (宽, 高)，用于屏幕等像素缩放（更接近临床 MPR 观感）。"""
    z, y, x = volume_shape
    if panel_key == "axial":
        return x, y
    if panel_key == "coronal":
        return x, z
    return y, z


def fit_slice_in_square(
    volume_shape: tuple[int, int, int],
    panel_key: str,
    side: int,
) -> tuple[int, int]:
    """方形视口内按体素等比例完整显示（与参考软件方形视图一致）。"""
    side = max(80, int(side))
    iw, ih = voxel_slice_size(volume_shape, panel_key)
    if iw < 1 or ih < 1:
        return side, side
    return side, side


def compute_mpr_layout_sizes(
    volume_shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    mpr_inner_w: int,
    axial_max_h: int,
    bottom_img_h: int,
    *,
    bottom_gap: int = 10,
    axial_landscape_ratio: float = 0.56,
) -> dict[str, tuple[int, int] | int]:
    """
    MPR 三视图布局：
    - 轴位：满宽横宽
    - 冠状 / 矢状：方形视口 + 体素等比 letterbox（避免压成横条）
    """
    ax_w = max(1, mpr_inner_w)
    pw_a, ph_a = physical_slice_size(volume_shape, spacing, "axial")
    if pw_a > 0 and ph_a > 0:
        ax_h = int(ax_w * ph_a / pw_a)
        if ax_h >= ax_w:
            ax_h = int(ax_w * axial_landscape_ratio)
    else:
        ax_h = int(ax_w * axial_landscape_ratio)
    ax_h = max(96, min(ax_h, axial_max_h, ax_w - 1))

    slot_w = max(1, (ax_w - bottom_gap) // 2)
    sa_slot_w = max(1, ax_w - bottom_gap - slot_w)
    box_h = max(160, bottom_img_h)
    co_side = min(slot_w, box_h)
    sa_side = min(sa_slot_w, box_h)
    co_w, co_h = fit_slice_in_square(volume_shape, "coronal", co_side)
    sa_w, sa_h = fit_slice_in_square(volume_shape, "sagittal", sa_side)

    return {
        "axial": (ax_w, ax_h),
        "coronal": (co_w, co_h),
        "sagittal": (sa_w, sa_h),
        "coronal_slot_w": slot_w,
        "sagittal_slot_w": sa_slot_w,
        "bottom_row_h": max(co_h, sa_h),
    }


def panel_texture_size(
    volume: np.ndarray,
    panel_key: str,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_w: int | None = None,
    max_h: int | None = None,
    max_side: int = 1024,
) -> tuple[int, int]:
    """各视口纹理宽高；优先按物理比例适配 max_w × max_h。"""
    if max_w is not None and max_h is not None:
        return fit_panel_in_box(volume.shape, spacing, panel_key, max_w, max_h)
    pw, ph = physical_slice_size(volume.shape, spacing, panel_key)
    scale = min(1.0, max_side / max(pw, ph, 1.0))
    return max(1, int(pw * scale)), max(1, int(ph * scale))


def resize_bilinear(image2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """双线性插值缩放（冠状/矢状显示更平滑）。"""
    h, w = image2d.shape[:2]
    out_h, out_w = max(1, int(out_h)), max(1, int(out_w))
    if h == out_h and w == out_w:
        return image2d

    if image2d.ndim == 2:
        channels = [image2d.astype(np.float32)]
    else:
        channels = [image2d[..., c].astype(np.float32) for c in range(image2d.shape[2])]

    yy = np.linspace(0.0, h - 1.0, out_h, dtype=np.float32)
    xx = np.linspace(0.0, w - 1.0, out_w, dtype=np.float32)
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    y0 = np.floor(gy).astype(np.int32)
    x0 = np.floor(gx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (gy - y0).astype(np.float32)
    wx = (gx - x0).astype(np.float32)

    planes: list[np.ndarray] = []
    for src in channels:
        v00 = src[y0, x0]
        v01 = src[y0, x1]
        v10 = src[y1, x0]
        v11 = src[y1, x1]
        out = (
            v00 * (1.0 - wy) * (1.0 - wx)
            + v01 * (1.0 - wy) * wx
            + v10 * wy * (1.0 - wx)
            + v11 * wy * wx
        )
        planes.append(np.clip(out, 0.0, 255.0).astype(np.uint8))

    if len(planes) == 1:
        return planes[0]
    return np.stack(planes, axis=-1)


def resize_for_texture(image2d: np.ndarray, tex_w: int, tex_h: int) -> np.ndarray:
    """将切片缩放到纹理大小（最近邻，轴位等仍可用）。"""
    h, w = image2d.shape[:2]
    if w == tex_w and h == tex_h:
        return image2d
    ys = np.linspace(0, h - 1, tex_h).astype(np.int32)
    xs = np.linspace(0, w - 1, tex_w).astype(np.int32)
    if image2d.ndim == 3:
        return image2d[np.ix_(ys, xs)]
    return image2d[np.ix_(ys, xs)]


def upscale_mpr_slice_for_display(
    image2d: np.ndarray,
    panel_key: str,
    volume_shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    *,
    min_aspect: float = 0.82,
) -> np.ndarray:
    """
    冠状/矢状层厚方向体素少时，仅用于显示的纵向插值放大。
    避免在方形视口里被压成无法辨认的横条（不改变原始数据）。
    """
    if panel_key not in ("coronal", "sagittal"):
        return image2d

    h, w = image2d.shape[:2]
    if h < 2 or w < 2:
        return image2d

    z, y, x = volume_shape
    sx, sy, sz = spacing
    if panel_key == "coronal":
        phys_h = z * sz
        phys_w = x * sx
    else:
        phys_h = z * sz
        phys_w = y * sy

    cur_aspect = h / w
    phys_aspect = phys_h / max(phys_w, 1e-6)
    target_aspect = max(min_aspect, min(phys_aspect, 1.15))

    if cur_aspect >= target_aspect * 0.95:
        return image2d

    target_h = max(h + 1, int(w * target_aspect))
    return resize_bilinear(image2d, target_h, w)


def resize_letterbox_smooth(image2d: np.ndarray, tex_w: int, tex_h: int) -> np.ndarray:
    """等比 letterbox + 双线性缩放。"""
    h, w = image2d.shape[:2]
    if w < 1 or h < 1:
        if image2d.ndim == 2:
            return np.zeros((tex_h, tex_w), dtype=np.uint8)
        return np.zeros((tex_h, tex_w, 3), dtype=np.uint8)

    scale = min(tex_w / w, tex_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    scaled = resize_bilinear(image2d, nh, nw)

    pad_y0 = (tex_h - nh) // 2
    pad_x0 = (tex_w - nw) // 2
    if scaled.ndim == 2:
        canvas = np.zeros((tex_h, tex_w), dtype=scaled.dtype)
        canvas[pad_y0 : pad_y0 + nh, pad_x0 : pad_x0 + nw] = scaled
        return canvas
    canvas = np.zeros((tex_h, tex_w, 3), dtype=scaled.dtype)
    canvas[pad_y0 : pad_y0 + nh, pad_x0 : pad_x0 + nw, :] = scaled
    return canvas


def resize_mpr_to_texture(
    image2d: np.ndarray,
    panel_key: str,
    tex_w: int,
    tex_h: int,
    volume_shape: tuple[int, int, int] | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> np.ndarray:
    if panel_key in ("coronal", "sagittal") and volume_shape is not None and spacing is not None:
        image2d = upscale_mpr_slice_for_display(image2d, panel_key, volume_shape, spacing)
        return resize_letterbox_smooth(image2d, tex_w, tex_h)
    return resize_letterbox(image2d, tex_w, tex_h)


def resize_letterbox(image2d: np.ndarray, tex_w: int, tex_h: int) -> np.ndarray:
    """等比缩放后居中铺到纹理（可能留黑边）。"""
    h, w = image2d.shape[:2]
    if w < 1 or h < 1:
        if image2d.ndim == 2:
            return np.zeros((tex_h, tex_w), dtype=np.uint8)
        return np.zeros((tex_h, tex_w, 3), dtype=np.uint8)

    scale = min(tex_w / w, tex_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    ys = np.linspace(0, h - 1, nh).astype(np.int32)
    xs = np.linspace(0, w - 1, nw).astype(np.int32)
    scaled = image2d[np.ix_(ys, xs)]

    pad_y0 = (tex_h - nh) // 2
    pad_x0 = (tex_w - nw) // 2
    if image2d.ndim == 2:
        canvas = np.zeros((tex_h, tex_w), dtype=scaled.dtype)
        canvas[pad_y0 : pad_y0 + nh, pad_x0 : pad_x0 + nw] = scaled
        return canvas
    canvas = np.zeros((tex_h, tex_w, 3), dtype=scaled.dtype)
    canvas[pad_y0 : pad_y0 + nh, pad_x0 : pad_x0 + nw, :] = scaled
    return canvas


def resize_cover(image2d: np.ndarray, tex_w: int, tex_h: int) -> np.ndarray:
    """等比放大填满纹理，超出部分居中裁剪（适合扁宽切片）。"""
    h, w = image2d.shape[:2]
    if w < 1 or h < 1:
        if image2d.ndim == 2:
            return np.zeros((tex_h, tex_w), dtype=np.uint8)
        return np.zeros((tex_h, tex_w, 3), dtype=np.uint8)

    scale = max(tex_w / w, tex_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    ys = np.linspace(0, h - 1, nh).astype(np.int32)
    xs = np.linspace(0, w - 1, nw).astype(np.int32)
    scaled = image2d[np.ix_(ys, xs)]

    y0 = max(0, (nh - tex_h) // 2)
    x0 = max(0, (nw - tex_w) // 2)
    cropped = scaled[y0 : y0 + tex_h, x0 : x0 + tex_w]
    if cropped.shape[0] == tex_h and cropped.shape[1] == tex_w:
        return cropped
    if image2d.ndim == 2:
        canvas = np.zeros((tex_h, tex_w), dtype=scaled.dtype)
        ch, cw = min(tex_h, cropped.shape[0]), min(tex_w, cropped.shape[1])
        canvas[:ch, :cw] = cropped[:ch, :cw]
        return canvas
    canvas = np.zeros((tex_h, tex_w, 3), dtype=scaled.dtype)
    ch, cw = min(tex_h, cropped.shape[0]), min(tex_w, cropped.shape[1])
    canvas[:ch, :cw, :] = cropped[:ch, :cw, :]
    return canvas


def auto_window_from_volume(volume: np.ndarray) -> tuple[float, float]:
    """根据体数据估计窗位、窗宽。"""
    lo, hi = np.percentile(volume, [0.5, 99.5])
    width = max(float(hi - lo), 1.0)
    level = float(lo + width / 2.0)
    return level, width


def to_texture_rgba(image2d: np.ndarray) -> tuple[list[float], int, int]:
    """将 HxW 或 HxWx3 图像转为 DPG float RGBA 纹理。"""
    if image2d.ndim == 2:
        gray = image2d.astype(np.float32) / 255.0
        rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1)
    else:
        rgb = image2d.astype(np.float32) / 255.0
        alpha = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
        rgba = np.concatenate([rgb, alpha], axis=-1)

    h, w = rgba.shape[:2]
    return rgba.flatten().tolist(), w, h
