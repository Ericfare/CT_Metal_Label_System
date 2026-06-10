"""
CT 金属伪影检测 + 3D 体渲染综合工具 — Dear PyGui 界面
处理流程见 SimpleITK滤波金属伪影掩码滤波器介绍.md
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
import tkinter as tk
from tkinter import filedialog

import dearpygui.dearpygui as dpg
import numpy as np
import SimpleITK as sitk

from dicom_io import read_dicom_folder
from image_view import (
    DEFAULT_WINDOW_LEVEL,
    DEFAULT_WINDOW_WIDTH,
    DisplayMode,
    ViewAxis,
    compose_display_slice,
    draw_mpr_crosshairs,
    compute_mpr_layout_sizes,
    extract_slice,
    resize_mpr_to_texture,
    sitk_to_numpy,
    slice_count,
    to_texture_rgba,
)
from metal_mask_pipeline import DOC_DEFAULT_MASK_PARAMS, MaskParams, generate_metal_mask
from vtk_async import VtkRenderWorker, _VtkJob
from vtk_volume_viewer import suppress_vtk_output


class AppState:
    def __init__(self) -> None:
        self.image: sitk.Image | None = None
        self.mask: sitk.Image | None = None
        self.volume: np.ndarray | None = None
        self.mask_volume: np.ndarray | None = None
        self.source_path: str = ""
        self.spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

        self.slice_index = {
            ViewAxis.AXIAL: 0,
            ViewAxis.CORONAL: 0,
            ViewAxis.SAGITTAL: 0,
        }
        self.main_view = ViewAxis.AXIAL
        self.display_mode = DisplayMode.OVERLAY
        self.window_level = DEFAULT_WINDOW_LEVEL
        self.window_width = DEFAULT_WINDOW_WIDTH

        self.params = MaskParams(**DOC_DEFAULT_MASK_PARAMS.__dict__)

        self.texture_tags: dict[str, int | str] = {}
        self.texture_sizes: dict[str, tuple[int, int]] = {}

        self.mask_update_pending = False
        self.mask_update_at = 0.0
        self.last_viewport_size: tuple[int, int] = (0, 0)

        self.vtk_worker: VtkRenderWorker | None = None
        self.vtk_pending_at = 0.0
        self.vtk_pending_kind: str | None = None
        self.vtk_interact_at = 0.0
        self.last_vtk_size: tuple[int, int] = (0, 0)
        self.vtk_enabled = True


STATE = AppState()

APP_TITLE = "CT 金属伪影检测 + 3D 体渲染综合工具"

SIDEBAR_WIDTH = 340
INPUT_BOX_WIDTH = 118
DEFAULT_VIEWPORT_WIDTH = 1520
DEFAULT_VIEWPORT_HEIGHT = 920
MPR_COLUMN_RATIO = 0.62
VTK_COLUMN_MIN_W = 360
VTK_COLUMN_MAX_W = 520
MPR_LABEL_H = 26
MPR_SLIDER_H = 32
MPR_BLOCK_PAD = 12
MPR_ROW_GAP = 8
VIEWER_HEADER_OFFSET = 96
VTK_TOOLBAR_H = 78
VTK_FOOTER_H = 52


def _mpr_block_chrome_h() -> int:
    """标题 + 滑条 + 内边距（略留余量，避免子窗口滚动）。"""
    return MPR_LABEL_H + MPR_SLIDER_H + MPR_BLOCK_PAD


def _vtk_chrome_h() -> int:
    return VTK_TOOLBAR_H + VTK_FOOTER_H

VR_PRESET_LABELS = ("骨骼", "软组织", "血管", "金属", "MIP")
VR_PRESET_MAP = {
    "骨骼": "bone",
    "软组织": "soft",
    "血管": "vessel",
    "金属": "metal",
    "MIP": "mip",
}

COLOR_SECTION_BLUE = (100, 175, 255)
COLOR_SECTION_ORANGE = (255, 165, 70)
COLOR_SECTION_GRAY = (190, 195, 205)
COLOR_ACCENT_GREEN = (72, 185, 110)
COLOR_ACCENT_BLUE = (70, 140, 220)
COLOR_MUTED = (150, 155, 165)

# 《标注方法.md》金属伪影 HU 阈值滑条范围
ANNOTATION_LOWER_HU_MIN = -500.0
ANNOTATION_LOWER_HU_MAX = 0.0
ANNOTATION_UPPER_HU_MIN = 500.0
ANNOTATION_UPPER_HU_MAX = 4000.0

# 控件 tag
TAG = {
    "axial_text": "axial_text",
    "coronal_text": "coronal_text",
    "sagittal_text": "sagittal_text",
    "axial_slider": "axial_slider",
    "coronal_slider": "coronal_slider",
    "sagittal_slider": "sagittal_slider",
    "axial_image": "axial_image",
    "coronal_image": "coronal_image",
    "sagittal_image": "sagittal_image",
    "main_panel": "main_panel",
    "mpr_panel": "mpr_panel",
    "vtk_panel": "vtk_panel",
    "mpr_bottom_row": "mpr_bottom_row",
    "vtk3d_image": "vtk3d_image",
    "vtk3d_text": "vtk3d_text",
}

VTK3D_WIDTH = 640
VTK3D_HEIGHT = 480

_BUTTON_THEMES: dict[str, int | str] = {}


def _pick_folder(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path or ""


def load_dicom_folder() -> None:
    folder = _pick_folder("选择 DICOM 文件夹")
    if not folder:
        return
    try:
        image = read_dicom_folder(folder)
        _set_volume(image, folder)
    except Exception as exc:  # noqa: BLE001
        print(f"DICOM 加载失败: {exc}")


def _shutdown_vtk_worker() -> None:
    """停止 VTK 后台线程并丢弃其中缓存的体数据。"""
    worker = STATE.vtk_worker
    if worker is None:
        return
    worker.shutdown()
    STATE.vtk_worker = None


def _clear_viewer_textures() -> None:
    """将 MPR / 3D 纹理恢复为空白占位。"""
    if not dpg.does_item_exist("global_textures"):
        return
    for panel_key in ("axial", "coronal", "sagittal"):
        if dpg.does_item_exist(TAG[f"{panel_key}_image"]):
            _recreate_panel_texture(panel_key, 2, 2)
    if dpg.does_item_exist(TAG["vtk3d_image"]):
        _recreate_vtk_texture(2, 2, [0.04, 0.06, 0.12, 1.0] * 4)


def _reset_viewer_labels_idle() -> None:
    idle = {
        "axial": "轴位 Axial · 未加载",
        "coronal": "冠状位 Coronal · 未加载",
        "sagittal": "矢状位 Sagittal · 未加载",
    }
    for panel_key, text in idle.items():
        tag = TAG.get(f"{panel_key}_text")
        if tag and dpg.does_item_exist(tag):
            dpg.set_value(tag, text)


def _release_previous_study() -> None:
    """卸载当前 DICOM / 掩码 / VTK 缓存，便于加载新病例时回收内存。"""
    had_data = STATE.image is not None or STATE.volume is not None
    if not had_data:
        return

    STATE.mask_update_pending = False
    STATE.mask_update_at = 0.0
    STATE.vtk_pending_kind = None
    STATE.vtk_pending_at = 0.0
    STATE.vtk_interact_at = 0.0

    _shutdown_vtk_worker()

    STATE.image = None
    STATE.mask = None
    STATE.volume = None
    STATE.mask_volume = None
    STATE.source_path = ""
    STATE.spacing = (1.0, 1.0, 1.0)
    for axis in ViewAxis:
        STATE.slice_index[axis] = 0

    _clear_viewer_textures()
    _reset_viewer_labels_idle()
    if dpg.does_item_exist(TAG["vtk3d_text"]):
        dpg.set_value(TAG["vtk3d_text"], "3D 体渲染 · 未加载")
    _set_mask_status("正在加载新影像…")

    gc.collect()
    print("已释放上一套 DICOM 数据")


def _set_volume(image: sitk.Image, source: str) -> None:
    from image_view import ensure_scalar_image

    _release_previous_study()

    image = ensure_scalar_image(image)
    STATE.image = image
    STATE.mask = None
    STATE.volume = sitk_to_numpy(image)
    STATE.mask_volume = None
    STATE.source_path = source
    sp = image.GetSpacing()
    STATE.spacing = (float(sp[0]), float(sp[1]), float(sp[2]))

    z, y, x = STATE.volume.shape
    for axis in ViewAxis:
        count = slice_count(STATE.volume, axis)
        STATE.slice_index[axis] = count // 2

    _apply_mask_defaults_to_ui()
    _apply_window_defaults_to_ui()

    STATE.mask_update_pending = False
    _configure_sliders()
    _init_all_panel_textures()
    if not _generate_mask_silent():
        dpg.set_value("display_mode", DisplayMode.OVERLAY.value)
        _set_mask_status("影像已加载；掩码生成失败，请检查参数后点击「生成伪影掩码」")
    _apply_viewer_layout()
    refresh_all_views()
    print(f"已加载: {os.path.basename(source)}  体素={x}×{y}×{z}")


def _configure_sliders() -> None:
    if STATE.volume is None:
        return
    for axis, slider_tag in (
        (ViewAxis.AXIAL, TAG["axial_slider"]),
        (ViewAxis.CORONAL, TAG["coronal_slider"]),
        (ViewAxis.SAGITTAL, TAG["sagittal_slider"]),
    ):
        count = slice_count(STATE.volume, axis)
        dpg.configure_item(slider_tag, min_value=0, max_value=max(0, count - 1), default_value=STATE.slice_index[axis])


def _apply_mask_defaults_to_ui() -> None:
    """将《标注方法.md》默认参数同步到界面控件。"""
    p = DOC_DEFAULT_MASK_PARAMS
    for tag, val in (
        ("lower_hu", p.lower_hu),
        ("upper_hu", p.upper_hu),
        ("grad_th", p.gradient_threshold),
        ("open_r", p.opening_radius),
        ("close_r", p.closing_radius),
        ("min_area", p.min_component_size),
    ):
        dpg.set_value(tag, val)
        if dpg.does_item_exist(f"{tag}_input"):
            dpg.set_value(f"{tag}_input", val)
    dpg.set_value("use_grad", p.use_gradient)
    dpg.set_value("largest_only", p.keep_largest_only)


def _apply_window_defaults_to_ui() -> None:
    STATE.window_level = DEFAULT_WINDOW_LEVEL
    STATE.window_width = DEFAULT_WINDOW_WIDTH
    dpg.set_value("window_level", DEFAULT_WINDOW_LEVEL)
    dpg.set_value("window_width", DEFAULT_WINDOW_WIDTH)
    if dpg.does_item_exist("window_level_input"):
        dpg.set_value("window_level_input", DEFAULT_WINDOW_LEVEL)
    if dpg.does_item_exist("window_width_input"):
        dpg.set_value("window_width_input", DEFAULT_WINDOW_WIDTH)


def _set_mask_status(text: str) -> None:
    if dpg.does_item_exist("mask_status"):
        dpg.set_value("mask_status", text)


def _mask_foreground_count(mask: sitk.Image) -> int:
    return int(sitk.GetArrayViewFromImage(mask).sum())


def _default_mask_save_name() -> str:
    if STATE.source_path:
        base = os.path.basename(os.path.normpath(STATE.source_path))
        if base:
            return f"{base}_metal_mask.nrrd"
    return "metal_mask.nrrd"


def _write_nrrd(image: sitk.Image, path: str) -> str:
    """写入 NRRD；Windows 中文路径下 ITK 可能失败，则经临时 ASCII 路径写出。"""
    path = os.path.abspath(path)
    if not path.lower().endswith(".nrrd"):
        path = f"{path}.nrrd"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        sitk.WriteImage(image, path, useCompression=False)
        return path
    except RuntimeError:
        fd, tmp = tempfile.mkstemp(suffix=".nrrd")
        os.close(fd)
        try:
            sitk.WriteImage(image, tmp, useCompression=False)
            shutil.copy2(tmp, path)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return path


def _apply_generated_mask(mask: sitk.Image, *, announce: bool) -> bool:
    """写入 STATE 并刷新 2D/3D 显示。"""
    if STATE.volume is not None:
        mv = sitk_to_numpy(mask)
        if mv.shape != STATE.volume.shape:
            print(f"警告: 掩码体素形状 {mv.shape} 与影像 {STATE.volume.shape} 不一致")
    STATE.mask = mask
    STATE.mask_volume = sitk_to_numpy(mask)
    count = _mask_foreground_count(mask)
    total = int(np.prod(STATE.mask_volume.shape)) if STATE.mask_volume is not None else 0
    pct = (100.0 * count / total) if total else 0.0

    dpg.set_value("display_mode", DisplayMode.OVERLAY.value)
    STATE.display_mode = DisplayMode.OVERLAY
    if dpg.does_item_exist("vtk_show_mask") and count > 0:
        dpg.set_value("vtk_show_mask", True)

    if count > 0:
        msg = f"掩码已生成：{count} 体素 ({pct:.3f}%)，三视图橙色叠加已开启"
    else:
        msg = "掩码已生成，但未检测到金属区域（可调整 HU 阈值后重新生成）"
    _set_mask_status(msg)
    if announce:
        print(msg)

    refresh_all_views()
    schedule_vtk_refresh("full", delay=0.8)
    return count > 0


def schedule_mask_update(delay: float = 0.4) -> None:
    """参数变化后延迟重建掩码，避免拖动滑块时卡顿。"""
    if STATE.image is None:
        return
    STATE.mask_update_pending = True
    STATE.mask_update_at = time.time() + delay
    _set_mask_status("参数已变，正在更新掩码…")


def process_pending_mask_update() -> None:
    if not STATE.mask_update_pending or time.time() < STATE.mask_update_at:
        return
    STATE.mask_update_pending = False
    if not _generate_mask_silent():
        _set_mask_status("掩码更新失败，请检查参数")


def on_mask_param_changed() -> None:
    schedule_mask_update()


def on_window_changed() -> None:
    refresh_all_views()


def _generate_mask_silent() -> bool:
    """加载后自动生成掩码（不弹错，仅打印）。"""
    if STATE.image is None:
        return False
    STATE.params = _read_params_from_ui()
    try:
        mask = generate_metal_mask(STATE.image, STATE.params)
        _apply_generated_mask(mask, announce=False)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"自动生成掩码失败: {exc}")
        STATE.mask = None
        STATE.mask_volume = None
        return False


def _read_params_from_ui() -> MaskParams:
    return MaskParams(
        lower_hu=dpg.get_value("lower_hu"),
        upper_hu=dpg.get_value("upper_hu"),
        gradient_threshold=dpg.get_value("grad_th"),
        opening_radius=int(dpg.get_value("open_r")),
        closing_radius=int(dpg.get_value("close_r")),
        min_component_size=int(dpg.get_value("min_area")),
        use_gradient=bool(dpg.get_value("use_grad")),
        keep_largest_only=bool(dpg.get_value("largest_only")),
    )


def generate_mask() -> None:
    if STATE.image is None:
        print("请先加载 DICOM 影像")
        _set_mask_status("请先加载 DICOM 影像")
        return
    STATE.mask_update_pending = False
    STATE.params = _read_params_from_ui()
    _set_mask_status("正在生成掩码…")
    try:
        mask = generate_metal_mask(STATE.image, STATE.params)
        _apply_generated_mask(mask, announce=True)
    except Exception as exc:  # noqa: BLE001
        print(f"掩码生成失败: {exc}")
        _set_mask_status(f"掩码生成失败: {exc}")
        STATE.mask = None
        STATE.mask_volume = None
        refresh_all_views()


def save_mask() -> None:
    if STATE.mask is None:
        print("请先生成掩码（加载 DICOM 后会自动生成，或点击「生成伪影掩码」）")
        _set_mask_status("请先生成掩码")
        return
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title="保存掩码 (NRRD)",
        defaultextension=".nrrd",
        initialfile=_default_mask_save_name(),
        filetypes=[("NRRD 掩码", "*.nrrd")],
    )
    root.destroy()
    if not path:
        return
    try:
        out_path = _write_nrrd(STATE.mask, path)
        print(f"掩码已保存 (NRRD): {out_path}")
        _set_mask_status(f"掩码已保存: {os.path.basename(out_path)}")
    except Exception as exc:  # noqa: BLE001
        print(f"保存失败: {exc}")
        _set_mask_status(f"保存失败: {exc}")


def _get_mask_slice(axis: ViewAxis, index: int) -> np.ndarray | None:
    if STATE.mask_volume is None:
        return None
    return extract_slice(STATE.mask_volume, axis, index)


def _recreate_panel_texture(panel_key: str, tex_w: int, tex_h: int) -> None:
    """先绑定新纹理再删除旧纹理，避免 Dear PyGui 无法 delete 正在使用的纹理。"""
    image_tag = TAG[f"{panel_key}_image"]
    old_tex = STATE.texture_tags.get(panel_key)
    blank = [0.0, 0.0, 0.0, 1.0] * (tex_w * tex_h)
    new_tex = dpg.add_dynamic_texture(
        tex_w, tex_h, blank, tag=f"tex_{panel_key}_live", parent="global_textures"
    )
    dpg.configure_item(image_tag, texture_tag=new_tex)
    if old_tex is not None and dpg.does_item_exist(old_tex):
        dpg.delete_item(old_tex)
    STATE.texture_tags[panel_key] = new_tex
    STATE.texture_sizes[panel_key] = (tex_w, tex_h)


def _init_all_panel_textures() -> None:
    if STATE.volume is None:
        return
    sizes = _compute_viewer_sizes()
    for panel_key in ("axial", "coronal", "sagittal"):
        tex_w, tex_h = sizes[panel_key]
        _recreate_panel_texture(panel_key, tex_w, tex_h)


def _update_texture(panel_key: str, image2d: np.ndarray) -> None:
    tex_w, tex_h = STATE.texture_sizes.get(panel_key, (0, 0))
    if tex_w < 1 or tex_h < 1:
        return
    vol_shape = STATE.volume.shape if STATE.volume is not None else None
    spacing = STATE.spacing if STATE.volume is not None else None
    resized = resize_mpr_to_texture(
        image2d, panel_key, tex_w, tex_h, volume_shape=vol_shape, spacing=spacing
    )
    data, w, h = to_texture_rgba(resized)
    if (w, h) != (tex_w, tex_h):
        return
    dpg.set_value(STATE.texture_tags[panel_key], data)


def _refresh_panel(panel_key: str, axis: ViewAxis) -> None:
    if STATE.volume is None:
        return
    idx = STATE.slice_index[axis]
    hu_slice = extract_slice(STATE.volume, axis, idx)
    mask_slice = _get_mask_slice(axis, idx)
    mode = DisplayMode(dpg.get_value("display_mode"))
    wl = dpg.get_value("window_level")
    ww = dpg.get_value("window_width")
    display = compose_display_slice(hu_slice, mask_slice, wl, ww, mode)
    if STATE.volume is not None:
        display = draw_mpr_crosshairs(display, axis, STATE.volume.shape, STATE.slice_index)
    _update_texture(panel_key, display)

    total = slice_count(STATE.volume, axis)
    names = {
        ViewAxis.AXIAL: "轴位 Axial",
        ViewAxis.CORONAL: "冠状位 Coronal",
        ViewAxis.SAGITTAL: "矢状位 Sagittal",
    }
    label = f"{names[axis]}     {idx + 1} / {total}"
    dpg.set_value(TAG[f"{panel_key}_text"], label)


def refresh_all_views() -> None:
    if STATE.volume is None:
        return
    layout = _current_layout()
    for panel_key, axis in layout:
        _refresh_panel(panel_key, axis)


def _current_vr_preset() -> str:
    if not dpg.does_item_exist("vr_preset"):
        return "bone"
    label = str(dpg.get_value("vr_preset"))
    return VR_PRESET_MAP.get(label, "bone")


def _vtk_show_metal_in_3d() -> bool:
    """金属预设或勾选「显示金属」时，在 3D 中叠加金属。"""
    if _current_vr_preset() == "metal":
        return True
    if dpg.does_item_exist("vtk_show_mask"):
        return bool(dpg.get_value("vtk_show_mask"))
    return False


def _vtk_payload() -> dict | None:
    if STATE.image is None:
        return None
    return {
        "image": STATE.image,
        "mask": STATE.mask,
        "wl": float(dpg.get_value("window_level")),
        "ww": float(dpg.get_value("window_width")),
        "vr_preset": _current_vr_preset(),
        "show_metal": _vtk_show_metal_in_3d(),
    }


def _ensure_vtk_worker() -> VtkRenderWorker:
    if STATE.vtk_worker is None:
        STATE.vtk_worker = VtkRenderWorker(_vtk_payload)
    return STATE.vtk_worker


def schedule_vtk_refresh(kind: str = "full", *, delay: float = 0.6) -> None:
    """将 VTK 体绘制放入后台线程，避免界面「未响应」。"""
    if not STATE.vtk_enabled or STATE.image is None:
        return
    if not dpg.does_item_exist(TAG["vtk3d_image"]):
        return
    STATE.vtk_pending_kind = kind
    STATE.vtk_pending_at = time.time() + delay
    if dpg.does_item_exist(TAG["vtk3d_text"]):
        dpg.set_value(TAG["vtk3d_text"], "3D 体渲染 · 正在后台渲染…")


def _submit_vtk_job(kind: str) -> None:
    sizes = _compute_viewer_sizes()
    vw, vh = sizes.get("vtk3d", (VTK3D_WIDTH, VTK3D_HEIGHT))
    _ensure_vtk_worker().submit(
        _VtkJob(kind=kind, width=vw, height=vh, show_mask=_vtk_show_metal_in_3d())
    )


def process_pending_vtk_refresh() -> None:
    if STATE.vtk_pending_kind is None or time.time() < STATE.vtk_pending_at:
        return
    worker = _ensure_vtk_worker()
    if worker.busy:
        STATE.vtk_pending_at = time.time() + 0.35
        return
    kind = STATE.vtk_pending_kind
    STATE.vtk_pending_kind = None
    _submit_vtk_job(kind)


def process_vtk_results() -> None:
    if STATE.vtk_worker is None:
        return
    result = STATE.vtk_worker.poll_result()
    if result is None:
        return
    if not result.get("ok"):
        err = result.get("error", "未知错误")
        print(f"VTK 三维渲染失败: {err}")
        if dpg.does_item_exist(TAG["vtk3d_text"]):
            dpg.set_value(TAG["vtk3d_text"], f"VTK 渲染失败: {err}")
        return
    _apply_vtk_rgba(result["rgba"])
    if dpg.does_item_exist(TAG["vtk3d_text"]):
        dpg.set_value(TAG["vtk3d_text"], "3D 体渲染 · Trackball 交互")


def refresh_vtk_3d_view() -> None:
    schedule_vtk_refresh("full", delay=0.1)


def _recreate_vtk_texture(tex_w: int, tex_h: int, data: list[float]) -> None:
    image_tag = TAG["vtk3d_image"]
    old_tex = STATE.texture_tags.get("vtk3d")
    new_tex = dpg.add_dynamic_texture(tex_w, tex_h, data, tag="tex_vtk3d_live", parent="global_textures")
    dpg.configure_item(image_tag, texture_tag=new_tex)
    if old_tex is not None and dpg.does_item_exist(old_tex):
        dpg.delete_item(old_tex)
    STATE.texture_tags["vtk3d"] = new_tex
    STATE.texture_sizes["vtk3d"] = (tex_w, tex_h)


def _apply_vtk_rgba(rgba: np.ndarray) -> None:
    h, w = rgba.shape[:2]
    data = rgba.flatten().tolist()
    if STATE.texture_sizes.get("vtk3d") != (w, h):
        _recreate_vtk_texture(w, h, data)
    else:
        tex = STATE.texture_tags.get("vtk3d")
        if tex is not None and dpg.does_item_exist(tex):
            dpg.set_value(tex, data)


def reset_vtk_camera() -> None:
    if STATE.image is None:
        return
    sizes = _compute_viewer_sizes()
    vw, vh = sizes.get("vtk3d", (VTK3D_WIDTH, VTK3D_HEIGHT))
    _ensure_vtk_worker().submit(
        _VtkJob(kind="reset", width=vw, height=vh, show_mask=_vtk_show_metal_in_3d())
    )


def on_vtk_mouse_drag() -> None:
    if STATE.image is None or not STATE.vtk_enabled:
        return
    if not dpg.is_item_hovered(TAG["vtk3d_image"]):
        return
    if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
        return
    now = time.time()
    if now - STATE.vtk_interact_at < 0.12:
        return
    dx, dy = dpg.get_mouse_drag_delta()
    if abs(dx) + abs(dy) < 0.01:
        return
    STATE.vtk_interact_at = now
    sizes = _compute_viewer_sizes()
    vw, vh = sizes.get("vtk3d", (VTK3D_WIDTH, VTK3D_HEIGHT))
    _ensure_vtk_worker().submit(
        _VtkJob(
            kind="rotate",
            width=vw,
            height=vh,
            show_mask=_vtk_show_metal_in_3d(),
            dx=dx,
            dy=dy,
        )
    )


def on_vtk_mouse_wheel(sender: str, app_data: int) -> None:  # noqa: ARG001
    if STATE.image is None or not STATE.vtk_enabled:
        return
    if not dpg.is_item_hovered(TAG["vtk3d_image"]):
        return
    now = time.time()
    if now - STATE.vtk_interact_at < 0.12:
        return
    STATE.vtk_interact_at = now
    sizes = _compute_viewer_sizes()
    vw, vh = sizes.get("vtk3d", (VTK3D_WIDTH, VTK3D_HEIGHT))
    zoom = 1.12 if app_data > 0 else 0.88
    _ensure_vtk_worker().submit(
        _VtkJob(kind="zoom", width=vw, height=vh, show_mask=_vtk_show_metal_in_3d(), zoom=zoom)
    )


def _current_layout() -> list[tuple[str, ViewAxis]]:
    """固定三视图布局：上轴位、左冠状、右矢状（与样例一致）。"""
    return [
        ("axial", ViewAxis.AXIAL),
        ("coronal", ViewAxis.CORONAL),
        ("sagittal", ViewAxis.SAGITTAL),
    ]


def on_slice_change(sender: str, value: int) -> None:  # noqa: ARG001
    mapping = {
        TAG["axial_slider"]: (ViewAxis.AXIAL, "axial"),
        TAG["coronal_slider"]: (ViewAxis.CORONAL, "coronal"),
        TAG["sagittal_slider"]: (ViewAxis.SAGITTAL, "sagittal"),
    }
    axis, _ = mapping[sender]
    STATE.slice_index[axis] = int(value)
    refresh_all_views()


def on_display_changed() -> None:
    refresh_all_views()


def on_mask_checkbox_changed() -> None:
    on_mask_param_changed()


def on_vr_preset_changed() -> None:
    if _current_vr_preset() == "metal" and dpg.does_item_exist("vtk_show_mask"):
        dpg.set_value("vtk_show_mask", True)
    schedule_vtk_refresh("full", delay=0.25)


def _colored_section(title: str, color: tuple[int, int, int]) -> None:
    dpg.add_spacer(height=8)
    with dpg.group(horizontal=True):
        dpg.add_text("■", color=color)
        dpg.add_text(title, color=color)
    dpg.add_spacer(height=4)


def _make_button_theme(
    normal: tuple[int, int, int],
    hover: tuple[int, int, int],
    active: tuple[int, int, int],
    *,
    text: tuple[int, int, int] = (245, 245, 248),
) -> int | str:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, normal)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, active)
            dpg.add_theme_color(dpg.mvThemeCol_Text, text)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
    return theme


def _init_button_themes() -> None:
    _BUTTON_THEMES["grey"] = _make_button_theme((72, 76, 86), (88, 92, 102), (58, 62, 72))
    _BUTTON_THEMES["green"] = _make_button_theme((48, 128, 78), (58, 148, 92), (38, 108, 68))
    _BUTTON_THEMES["blue"] = _make_button_theme((52, 108, 178), (64, 124, 198), (42, 92, 158))
    _BUTTON_THEMES["dark"] = _make_button_theme((48, 52, 62), (62, 66, 78), (38, 42, 50))


def _styled_button(label: str, tag: str, theme_key: str, callback, *, height: int = 36) -> None:
    dpg.add_button(label=label, tag=tag, width=-1, height=height, callback=callback)
    theme = _BUTTON_THEMES.get(theme_key)
    if theme is not None:
        dpg.bind_item_theme(tag, theme)


def _bind_slider_input_float(
    tag: str, input_tag: str, min_v: float, max_v: float, *, on_change
) -> None:
    def from_slider() -> None:
        dpg.set_value(input_tag, dpg.get_value(tag))
        on_change()

    def from_input() -> None:
        value = float(np.clip(dpg.get_value(input_tag), min_v, max_v))
        dpg.set_value(tag, value)
        dpg.set_value(input_tag, value)
        on_change()

    dpg.set_item_callback(tag, lambda: from_slider())
    dpg.set_item_callback(input_tag, lambda: from_input())


def _bind_slider_input_int(
    tag: str, input_tag: str, min_v: int, max_v: int, *, on_change
) -> None:
    def from_slider() -> None:
        dpg.set_value(input_tag, dpg.get_value(tag))
        on_change()

    def from_input() -> None:
        value = int(np.clip(int(dpg.get_value(input_tag)), min_v, max_v))
        dpg.set_value(tag, value)
        dpg.set_value(input_tag, value)
        on_change()

    dpg.set_item_callback(tag, lambda: from_slider())
    dpg.set_item_callback(input_tag, lambda: from_input())


def _add_mask_slider_float(
    label: str, tag: str, default: float, min_v: float, max_v: float, *, fmt: str = "%.0f"
) -> None:
    input_tag = f"{tag}_input"
    with dpg.group(horizontal=True):
        dpg.add_text(label)
        dpg.add_input_float(
            tag=input_tag,
            default_value=default,
            width=INPUT_BOX_WIDTH,
            min_value=min_v,
            max_value=max_v,
            step=1.0,
            format=fmt,
        )
    dpg.add_slider_float(
        tag=tag,
        default_value=default,
        min_value=min_v,
        max_value=max_v,
        width=-1,
        format=fmt,
    )
    _bind_slider_input_float(tag, input_tag, min_v, max_v, on_change=on_mask_param_changed)
    dpg.add_spacer(height=6)


def _add_mask_slider_int(label: str, tag: str, default: int, min_v: int, max_v: int) -> None:
    input_tag = f"{tag}_input"
    with dpg.group(horizontal=True):
        dpg.add_text(label)
        dpg.add_input_int(
            tag=input_tag,
            default_value=default,
            width=INPUT_BOX_WIDTH,
            min_value=min_v,
            max_value=max_v,
            step=1,
        )
    dpg.add_slider_int(
        tag=tag,
        default_value=default,
        min_value=min_v,
        max_value=max_v,
        width=-1,
        format="%d",
    )
    _bind_slider_input_int(tag, input_tag, min_v, max_v, on_change=on_mask_param_changed)
    dpg.add_spacer(height=6)


def _add_window_slider_float(label: str, tag: str, default: float, min_v: float, max_v: float) -> None:
    input_tag = f"{tag}_input"
    with dpg.group(horizontal=True):
        dpg.add_text(label)
        dpg.add_input_float(
            tag=input_tag,
            default_value=default,
            width=INPUT_BOX_WIDTH,
            min_value=min_v,
            max_value=max_v,
            step=10.0,
            format="%.0f",
        )
    dpg.add_slider_float(
        tag=tag,
        default_value=default,
        min_value=min_v,
        max_value=max_v,
        width=-1,
        format="%.0f",
    )
    _bind_slider_input_float(tag, input_tag, min_v, max_v, on_change=on_window_changed)
    dpg.add_spacer(height=6)


def _compute_viewer_sizes() -> dict[str, tuple[int, int] | int]:
    """三栏布局：MPR 与 3D 同高适配视口，避免内部滚动与 3D 过长。"""
    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    content_w = max(720, vw - SIDEBAR_WIDTH - 40)
    viewer_h = max(420, vh - VIEWER_HEADER_OFFSET)

    vtk_col_w = int(np.clip(content_w * (1.0 - MPR_COLUMN_RATIO), VTK_COLUMN_MIN_W, VTK_COLUMN_MAX_W))
    mpr_col_w = content_w - vtk_col_w - 12
    mpr_inner_w = max(200, mpr_col_w - 20)

    chrome = _mpr_block_chrome_h()
    vtk_chrome = _vtk_chrome_h()
    bottom_row_h = max(180, int(viewer_h * 0.34))
    axial_max_h = max(
        96,
        viewer_h - bottom_row_h - chrome * 2 - MPR_ROW_GAP - 12,
    )

    if STATE.volume is not None:
        mpr_sizes = compute_mpr_layout_sizes(
            STATE.volume.shape,
            STATE.spacing,
            mpr_inner_w,
            axial_max_h,
            bottom_row_h,
        )
        ax_w, ax_h = mpr_sizes["axial"]
        co_w, co_h = mpr_sizes["coronal"]
        sa_w, sa_h = mpr_sizes["sagittal"]
        co_slot_w = int(mpr_sizes["coronal_slot_w"])
        sa_slot_w = int(mpr_sizes["sagittal_slot_w"])
        bottom_row_h = int(mpr_sizes["bottom_row_h"])
    else:
        ax_w = mpr_inner_w
        ax_h = max(96, min(axial_max_h, int(ax_w * 0.56)))
        co_slot_w = max(120, (ax_w - 10) // 2)
        sa_slot_w = max(120, ax_w - 10 - co_slot_w)
        co_w, co_h = co_slot_w, bottom_row_h
        sa_w, sa_h = sa_slot_w, bottom_row_h

    row_img_h = max(co_h, sa_h)
    mpr_panel_h = ax_h + chrome + MPR_ROW_GAP + row_img_h + chrome + 8
    vtk_inner_w = max(280, vtk_col_w - 20)
    vtk_img_h = max(200, min(mpr_panel_h - 24, viewer_h - vtk_chrome - 8))
    vtk_panel_h = vtk_img_h + vtk_chrome
    column_h = min(viewer_h, max(mpr_panel_h, vtk_panel_h))

    return {
        "mpr_col_w": mpr_col_w,
        "vtk_col_w": vtk_col_w,
        "column_h": column_h,
        "mpr_panel_h": mpr_panel_h,
        "vtk_panel_h": vtk_panel_h,
        "axial": (ax_w, ax_h),
        "coronal": (co_w, co_h),
        "sagittal": (sa_w, sa_h),
        "coronal_block_w": co_slot_w,
        "sagittal_block_w": sa_slot_w,
        "bottom_row_h": bottom_row_h,
        "vtk3d": (vtk_inner_w, vtk_img_h),
    }


def _sync_panel_textures_to_layout() -> None:
    if STATE.volume is None:
        return
    sizes = _compute_viewer_sizes()
    changed = False
    for key in ("axial", "coronal", "sagittal"):
        target = sizes[key]
        if STATE.texture_sizes.get(key) != target:
            changed = True
            _recreate_panel_texture(key, target[0], target[1])
    if changed:
        refresh_all_views()


def _apply_viewer_layout() -> None:
    if not dpg.does_item_exist(TAG["axial_image"]):
        return
    sizes = _compute_viewer_sizes()
    col_h = int(sizes["column_h"])
    if dpg.does_item_exist(TAG["mpr_panel"]):
        dpg.configure_item(
            TAG["mpr_panel"],
            width=int(sizes["mpr_col_w"]),
            height=col_h,
        )
    if dpg.does_item_exist(TAG["vtk_panel"]):
        dpg.configure_item(
            TAG["vtk_panel"],
            width=int(sizes["vtk_col_w"]),
            height=col_h,
        )
    if dpg.does_item_exist(TAG["main_panel"]):
        dpg.configure_item(TAG["main_panel"], height=col_h)
    _sync_panel_textures_to_layout()
    for key in ("axial", "coronal", "sagittal", "vtk3d"):
        w, h = sizes[key]
        dpg.configure_item(TAG[f"{key}_image"], width=w, height=h)
        if key != "vtk3d":
            dpg.configure_item(TAG[f"{key}_slider"], width=w)
    if STATE.image is not None and STATE.vtk_enabled:
        new_size = sizes["vtk3d"]
        if new_size != STATE.last_vtk_size:
            STATE.last_vtk_size = new_size
            schedule_vtk_refresh("render", delay=0.8)


def _poll_viewport_resize() -> None:
    """兼容旧版 Dear PyGui：在主循环中检测视口尺寸变化。"""
    size = (dpg.get_viewport_client_width(), dpg.get_viewport_client_height())
    if size != STATE.last_viewport_size:
        STATE.last_viewport_size = size
        _apply_viewer_layout()


def _setup_chinese_font() -> None:
    """加载系统中文字体，避免界面中文显示为问号。"""
    windir = os.environ.get("WINDIR", r"C:\Windows")
    fonts_dir = os.path.join(windir, "Fonts")
    candidates = [
        os.path.join(fonts_dir, "msyh.ttc"),    # 微软雅黑
        os.path.join(fonts_dir, "simhei.ttf"),  # 黑体
        os.path.join(fonts_dir, "simsun.ttc"),  # 宋体
    ]

    with dpg.font_registry(tag="chinese_font_registry"):
        for font_path in candidates:
            if not os.path.isfile(font_path):
                continue
            try:
                with dpg.font(font_path, 18, tag="default_chinese_font") as default_font:
                    pass
                dpg.bind_font(default_font)
                return
            except Exception as exc:  # noqa: BLE001
                print(f"字体加载失败 ({font_path}): {exc}")

    print("警告：未找到可用中文字体，界面中文可能显示为问号。")


def build_ui() -> None:
    dpg.create_context()
    suppress_vtk_output()
    _setup_chinese_font()
    _init_button_themes()
    dpg.create_viewport(
        title="CT Metal Artifact VR Tool",
        width=DEFAULT_VIEWPORT_WIDTH,
        height=DEFAULT_VIEWPORT_HEIGHT,
    )

    init_tex_sizes = _compute_viewer_sizes()
    vtk_tw, vtk_th = init_tex_sizes["vtk3d"]

    with dpg.texture_registry(tag="global_textures"):
        for key in ("axial", "coronal", "sagittal"):
            STATE.texture_tags[key] = dpg.add_dynamic_texture(
                2, 2, [0.0, 0.0, 0.0, 1.0] * 4, tag=f"tex_{key}"
            )
            STATE.texture_sizes[key] = (2, 2)
        blank_vtk = [0.04, 0.06, 0.12, 1.0] * (vtk_tw * vtk_th)
        STATE.texture_tags["vtk3d"] = dpg.add_dynamic_texture(
            vtk_tw, vtk_th, blank_vtk, tag="tex_vtk3d"
        )
        STATE.texture_sizes["vtk3d"] = (vtk_tw, vtk_th)

    with dpg.window(tag="primary_window"):
        dpg.add_text(APP_TITLE, tag="app_title", color=(235, 238, 245))
        dpg.add_separator()
        dpg.add_spacer(height=6)

        with dpg.group(horizontal=True):
            # ---------- 左侧控制栏 ----------
            with dpg.child_window(
                width=SIDEBAR_WIDTH,
                height=-1,
                border=True,
                tag="sidebar",
            ):
                _colored_section("文件操作", COLOR_SECTION_GRAY)
                _styled_button("加载 DICOM 文件夹", "btn_load_dicom", "grey", lambda: load_dicom_folder())
                dpg.add_spacer(height=6)
                _styled_button("生成伪影掩码", "btn_gen_mask", "green", lambda: generate_mask(), height=38)
                dpg.add_spacer(height=6)
                _styled_button("保存掩码", "btn_save_mask", "blue", lambda: save_mask())

                _colored_section("窗宽 / 窗位", COLOR_SECTION_BLUE)
                _add_window_slider_float("窗位 WL", "window_level", DEFAULT_WINDOW_LEVEL, -1024.0, 3071.0)
                _add_window_slider_float("窗宽 WW", "window_width", DEFAULT_WINDOW_WIDTH, 1.0, 8192.0)

                _colored_section("阈值参数（标注方法.md）", COLOR_SECTION_ORANGE)
                p = DOC_DEFAULT_MASK_PARAMS
                _add_mask_slider_float(
                    "低 HU 阈值 (<)",
                    "lower_hu",
                    p.lower_hu,
                    ANNOTATION_LOWER_HU_MIN,
                    ANNOTATION_LOWER_HU_MAX,
                )
                _add_mask_slider_float(
                    "高 HU 阈值 (>)",
                    "upper_hu",
                    p.upper_hu,
                    ANNOTATION_UPPER_HU_MIN,
                    ANNOTATION_UPPER_HU_MAX,
                )
                _add_mask_slider_float("梯度阈值", "grad_th", p.gradient_threshold, 50.0, 500.0)

                _colored_section("形态学参数", COLOR_SECTION_GRAY)
                _add_mask_slider_int("开运算半径", "open_r", p.opening_radius, 0, 5)
                _add_mask_slider_int("闭运算半径", "close_r", p.closing_radius, 0, 10)
                _add_mask_slider_int("最小连通域", "min_area", p.min_component_size, 10, 500)
                dpg.add_checkbox(
                    label="使用梯度约束",
                    tag="use_grad",
                    default_value=True,
                    callback=lambda: on_mask_checkbox_changed(),
                )
                dpg.add_checkbox(
                    label="只保留最大连通域",
                    tag="largest_only",
                    default_value=False,
                    callback=lambda: on_mask_checkbox_changed(),
                )

                _colored_section("切片显示", COLOR_SECTION_GRAY)
                dpg.add_text("显示模式", color=COLOR_MUTED)
                dpg.add_combo(
                    tag="display_mode",
                    items=[m.value for m in DisplayMode],
                    default_value=DisplayMode.OVERLAY.value,
                    width=-1,
                    callback=lambda: on_display_changed(),
                )
                dpg.add_spacer(height=10)
                dpg.add_text("拖动滑块将自动更新橙色掩码", tag="mask_status", wrap=300, color=COLOR_MUTED)

            # ---------- 右侧：MPR 三视图 + 3D 体绘制 ----------
            init_sizes = _compute_viewer_sizes()
            with dpg.child_window(
                tag=TAG["main_panel"],
                width=-1,
                height=init_sizes["column_h"],
                border=False,
                no_scrollbar=True,
                horizontal_scrollbar=False,
            ):
                mpr_col_w = int(init_sizes["mpr_col_w"])
                vtk_col_w = int(init_sizes["vtk_col_w"])
                ax_w, ax_h = init_sizes["axial"]
                vtk_w, vtk_h = init_sizes["vtk3d"]
                co_w, co_h = init_sizes["coronal"]
                sa_w, sa_h = init_sizes["sagittal"]

                with dpg.group(horizontal=True):
                    # 中间栏：轴位（大）+ 冠状 / 矢状（并排）
                    with dpg.child_window(
                        tag=TAG["mpr_panel"],
                        width=mpr_col_w,
                        height=init_sizes["column_h"],
                        border=True,
                        no_scrollbar=True,
                        horizontal_scrollbar=False,
                    ):
                        with dpg.group(tag="axial_block"):
                            dpg.add_text("轴位 Axial · 未加载", tag=TAG["axial_text"], color=(200, 210, 225))
                            dpg.add_image("tex_axial", tag=TAG["axial_image"], width=ax_w, height=ax_h)
                            dpg.add_slider_int(
                                tag=TAG["axial_slider"],
                                label="",
                                default_value=0,
                                min_value=0,
                                max_value=0,
                                width=ax_w,
                                callback=on_slice_change,
                            )
                        dpg.add_spacer(height=6)
                        with dpg.group(horizontal=True, tag=TAG["mpr_bottom_row"]):
                            with dpg.group(tag="coronal_block"):
                                dpg.add_text(
                                    "冠状位 Coronal · 未加载",
                                    tag=TAG["coronal_text"],
                                    color=(200, 210, 225),
                                )
                                dpg.add_image("tex_coronal", tag=TAG["coronal_image"], width=co_w, height=co_h)
                                dpg.add_slider_int(
                                    tag=TAG["coronal_slider"],
                                    label="",
                                    default_value=0,
                                    min_value=0,
                                    max_value=0,
                                    width=co_w,
                                    callback=on_slice_change,
                                )
                            dpg.add_spacer(width=8)
                            with dpg.group(tag="sagittal_block"):
                                dpg.add_text(
                                    "矢状位 Sagittal · 未加载",
                                    tag=TAG["sagittal_text"],
                                    color=(200, 210, 225),
                                )
                                dpg.add_image("tex_sagittal", tag=TAG["sagittal_image"], width=sa_w, height=sa_h)
                                dpg.add_slider_int(
                                    tag=TAG["sagittal_slider"],
                                    label="",
                                    default_value=0,
                                    min_value=0,
                                    max_value=0,
                                    width=sa_w,
                                    callback=on_slice_change,
                                )

                    dpg.add_spacer(width=8)

                    # 右侧栏：3D 体绘制（整列高度）
                    with dpg.child_window(
                        tag=TAG["vtk_panel"],
                        width=vtk_col_w,
                        height=init_sizes["column_h"],
                        border=True,
                        no_scrollbar=True,
                        horizontal_scrollbar=False,
                    ):
                        with dpg.group(horizontal=True):
                            dpg.add_text(
                                "3D 体渲染 · Trackball 交互",
                                tag=TAG["vtk3d_text"],
                                color=(140, 190, 255),
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(
                                label="体绘制",
                                tag="vtk_enabled",
                                default_value=True,
                                callback=lambda s, a: _on_vtk_enabled_changed(),
                            )
                            dpg.add_checkbox(
                                label="显示金属掩码",
                                tag="vtk_show_mask",
                                default_value=False,
                                callback=lambda: schedule_vtk_refresh("full", delay=0.3),
                            )
                            dpg.add_button(
                                label="重置视角",
                                tag="btn_reset_vtk",
                                height=24,
                                callback=lambda: reset_vtk_camera(),
                            )
                            dpg.bind_item_theme("btn_reset_vtk", _BUTTON_THEMES["dark"])
                            dpg.add_button(
                                label="刷新",
                                tag="btn_refresh_vtk",
                                height=24,
                                callback=lambda: refresh_vtk_3d_view(),
                            )
                            dpg.bind_item_theme("btn_refresh_vtk", _BUTTON_THEMES["dark"])
                        dpg.add_image("tex_vtk3d", tag=TAG["vtk3d_image"], width=vtk_w, height=vtk_h)
                        dpg.add_spacer(height=4)
                        dpg.add_text("渲染类型", color=COLOR_MUTED)
                        dpg.add_radio_button(
                            items=list(VR_PRESET_LABELS),
                            tag="vr_preset",
                            default_value="骨骼",
                            horizontal=True,
                            callback=lambda: on_vr_preset_changed(),
                        )

    with dpg.handler_registry(tag="global_handlers"):
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=lambda: on_vtk_mouse_drag())
        dpg.add_mouse_wheel_handler(callback=on_vtk_mouse_wheel)

    dpg.bind_theme(_build_dark_theme())
    _bind_slider_themes()
    dpg.setup_dearpygui()
    _apply_viewer_layout()
    # 部分环境下需在 setup 后再次绑定字体
    if dpg.does_item_exist("default_chinese_font"):
        dpg.bind_font("default_chinese_font")
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)
    STATE.last_viewport_size = (
        dpg.get_viewport_client_width(),
        dpg.get_viewport_client_height(),
    )


_SLIDER_THEME: int | str | None = None


def _bind_slider_themes() -> None:
    global _SLIDER_THEME
    with dpg.theme() as slider_theme:
        with dpg.theme_component(dpg.mvSliderInt):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (38, 48, 68))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (90, 155, 235))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (120, 180, 255))
        with dpg.theme_component(dpg.mvSliderFloat):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (38, 48, 68))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (90, 155, 235))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (120, 180, 255))
    _SLIDER_THEME = slider_theme
    for tag in (
        TAG["axial_slider"],
        TAG["coronal_slider"],
        TAG["sagittal_slider"],
        "window_level",
        "window_width",
        "lower_hu",
        "upper_hu",
        "grad_th",
        "open_r",
        "close_r",
        "min_area",
    ):
        if dpg.does_item_exist(tag):
            dpg.bind_item_theme(tag, slider_theme)


def _build_dark_theme() -> int | str:
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (18, 20, 26))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (26, 28, 36))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (48, 52, 64))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (34, 38, 48))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (44, 48, 60))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (58, 96, 142))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (72, 118, 168))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (48, 82, 120))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (100, 175, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (58, 96, 142))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 232, 238))
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (50, 54, 66))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 8)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 14)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 12)
    return global_theme


def _on_vtk_enabled_changed() -> None:
    STATE.vtk_enabled = bool(dpg.get_value("vtk_enabled"))
    if STATE.vtk_enabled and STATE.image is not None:
        schedule_vtk_refresh("full", delay=0.3)
    elif dpg.does_item_exist(TAG["vtk3d_text"]):
        dpg.set_value(TAG["vtk3d_text"], "3D 体渲染 · 已关闭")


def main() -> None:
    build_ui()
    try:
        while dpg.is_dearpygui_running():
            _poll_viewport_resize()
            process_pending_mask_update()
            process_pending_vtk_refresh()
            process_vtk_results()
            dpg.render_dearpygui_frame()
    finally:
        _release_previous_study()
        dpg.destroy_context()


if __name__ == "__main__":
    main()
