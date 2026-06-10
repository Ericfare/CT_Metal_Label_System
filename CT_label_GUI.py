import gc
import os
import shutil
import sys
import tempfile
import SimpleITK as sitk
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

import vtk
# VTK与Qt界面融合的核心组件
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from image_view import (
    apply_window,
    physical_slice_size,
    resize_bilinear,
    upscale_mpr_slice_for_display,
)
from dicom_io import read_dicom_folder
from metal_mask_pipeline import (
    DOC_DEFAULT_MASK_PARAMS,
    MaskParams,
    generate_metal_mask,
    generate_patient_roi_mask,
)

# 《标注方法.md》脑部 CT 标注窗
ANNOTATION_WINDOW_LEVEL = 40.0
ANNOTATION_WINDOW_WIDTH = 80.0


def _normalize_mask_save_path(path: str) -> str:
    """补全扩展名；单独 .gz 会改为 .nii.gz（SimpleITK 无法识别裸 .gz）。"""
    path = os.path.abspath(path)
    lower = path.lower()
    if lower.endswith((".nii.gz", ".nrrd", ".nii", ".mha", ".mhd")):
        return path
    if lower.endswith(".gz"):
        return path[:-3] + ".nii.gz"
    if "." not in os.path.basename(path):
        return path + ".nii.gz"
    return path + ".nii.gz"


def _needs_temp_io_path(path: str) -> bool:
    """Windows 下中文等非 ASCII 路径时 ITK 无法直接读写，且可能弹出错误框。"""
    try:
        path.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _write_mask_image(image: sitk.Image, path: str) -> str:
    """写入掩码；中文路径经临时 ASCII 文件写出，避免 ITK 弹窗且保证保存成功。"""
    path = _normalize_mask_save_path(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lower = path.lower()
    use_compression = lower.endswith((".nii.gz", ".nii"))
    if lower.endswith(".nrrd"):
        use_compression = False

    suffix = ".nrrd" if lower.endswith(".nrrd") else ".nii.gz"

    def _do_write(target: str) -> None:
        sitk.WriteImage(image, target, useCompression=use_compression)

    if _needs_temp_io_path(path):
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            _do_write(tmp)
            shutil.copy2(tmp, path)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    else:
        _do_write(path)
    return path


NIFTI_FILTER = "NIfTI 影像 (*.nii.gz *.nii);;所有文件 (*.*)"


def _read_sitk_image(path: str) -> sitk.Image:
    """读取体数据；中文路径经临时 ASCII 文件加载，避免 ITK 报错弹窗。"""
    path = os.path.abspath(path)
    if not _needs_temp_io_path(path):
        return sitk.ReadImage(path)

    lower = path.lower()
    if lower.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif lower.endswith(".nii"):
        suffix = ".nii"
    elif lower.endswith(".nrrd"):
        suffix = ".nrrd"
    else:
        suffix = ".nii.gz"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        shutil.copy2(path, tmp)
        return sitk.ReadImage(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _prepare_ct_image(image: sitk.Image) -> sitk.Image:
    if image.GetNumberOfComponentsPerPixel() > 1:
        image = sitk.VectorIndexSelectionCast(image, 0, sitk.sitkFloat32)
    return sitk.Cast(image, sitk.sitkFloat32)


def _prepare_binary_mask(image: sitk.Image) -> sitk.Image:
    """将标注图转为 0/1 二值掩码。"""
    ref = image
    if image.GetNumberOfComponentsPerPixel() > 1:
        image = sitk.VectorIndexSelectionCast(image, 0, sitk.sitkFloat32)
    else:
        image = sitk.Cast(image, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(image)
    binary = (arr > 0.5).astype(np.uint8)
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(ref)
    return sitk.Cast(out, sitk.sitkUInt8)


def _align_mask_to_ct(mask: sitk.Image, ct: sitk.Image) -> sitk.Image:
    """将标注掩码重采样到 CT 网格（最近邻）。"""
    mask = _prepare_binary_mask(mask)
    if (
        mask.GetSize() == ct.GetSize()
        and mask.GetSpacing() == ct.GetSpacing()
        and mask.GetOrigin() == ct.GetOrigin()
        and mask.GetDirection() == ct.GetDirection()
    ):
        return mask
    return sitk.Resample(
        mask,
        ct,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkUInt8,
    )


from vtk_volume_viewer import (
    _bone_surface_actor,
    _create_volume_mapper,
    _metal_hu_surface_actor,
    prepare_ct_for_vr,
    sitk_to_vtk_image,
    suppress_vtk_output,
)

# 禁止 VTK 弹出 vtkOutputWindow（警告/错误改输出到终端）
suppress_vtk_output()

# 界面主题色
UI_BG = "#1a1d26"
UI_PANEL = "#232733"
UI_PANEL_BORDER = "#3a4158"
UI_TEXT = "#e8eaef"
UI_MUTED = "#9aa3b5"
UI_ACCENT_BLUE = "#5b9cf5"
UI_ACCENT_ORANGE = "#f0a050"
UI_ACCENT_GREEN = "#4cc97a"
UI_VIEWPORT_BG = "#0c0e14"
UI_SLICE_BADGE = "#2a3550"

APP_QSS = f"""
QMainWindow, QWidget {{
    background-color: {UI_BG};
    color: {UI_TEXT};
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QFrame#sidebar, QFrame#viewerPanel {{
    background-color: {UI_PANEL};
    border: 1px solid {UI_PANEL_BORDER};
    border-radius: 10px;
}}
QLabel#appTitle {{
    font-size: 16px;
    font-weight: 700;
    color: {UI_TEXT};
    padding: 4px 0 8px 0;
}}
QLabel#sectionTitle {{
    font-weight: 600;
    font-size: 13px;
}}
QLabel#paramLabel {{
    color: {UI_MUTED};
    font-size: 12px;
    padding-top: 6px;
}}
QLabel#paramValue {{
    color: {UI_ACCENT_BLUE};
    font-weight: 600;
    font-size: 12px;
    min-width: 42px;
}}
QLabel#sliceBadge {{
    color: {UI_ACCENT_BLUE};
    font-weight: 600;
    font-size: 11px;
    background-color: {UI_SLICE_BADGE};
    padding: 5px 10px;
    border-radius: 6px;
}}
QLabel#viewTitle {{
    font-weight: 600;
    font-size: 13px;
    color: #c5d0e6;
}}
QLabel#statusHint {{
    color: {UI_MUTED};
    font-size: 11px;
    padding: 8px 4px;
}}
QLabel#imageViewport {{
    background-color: {UI_VIEWPORT_BG};
    border: 1px solid #2e3548;
    border-radius: 8px;
}}
QPushButton {{
    border: none;
    border-radius: 8px;
    padding: 10px 14px;
    font-weight: 600;
    min-height: 20px;
}}
QPushButton#btnGrey {{
    background-color: #3d4558;
    color: {UI_TEXT};
}}
QPushButton#btnGrey:hover {{ background-color: #4a5368; }}
QPushButton#btnGreen {{
    background-color: #2d8f55;
    color: #ffffff;
    font-size: 14px;
    min-height: 28px;
}}
QPushButton#btnGreen:hover {{ background-color: #36a866; }}
QPushButton#btnBlue {{
    background-color: #3a6eb8;
    color: #ffffff;
}}
QPushButton#btnBlue:hover {{ background-color: #4a7ec8; }}
QPushButton:disabled {{
    background-color: #2a2f3a;
    color: #6a7080;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: #2e3548;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    background: {UI_ACCENT_BLUE};
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3a6eb8, stop:1 #5b9cf5);
    border-radius: 3px;
}}
QProgressBar {{
    border: 1px solid {UI_PANEL_BORDER};
    border-radius: 6px;
    text-align: center;
    background: #1e2230;
    color: {UI_TEXT};
    height: 18px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3a6eb8, stop:1 #5b9cf5);
    border-radius: 5px;
}}
"""

# 多连通域金属伪影着色（与参考 3D 效果图一致）
METAL_COMPONENT_COLORS = (
    (0.25, 0.88, 0.95),  # 青
    (0.25, 0.92, 0.35),  # 绿
    (1.0, 0.88, 0.22),   # 黄
    (1.0, 0.55, 0.12),   # 橙
    (0.75, 0.75, 0.82),  # 灰（固定架等）
)

# ===================== 异步进度条弹窗 =====================
# 功能：处理数据时弹出加载进度条，防止界面卡死
class ProgressDialog(QDialog):
    def __init__(self, title="处理中...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(460, 130)
        self.setModal(True)
        self.setStyleSheet(APP_QSS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(14)

        self.label = QLabel("初始化处理...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f"color: {UI_TEXT}; font-size: 13px;")
        layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

    # 更新进度条值和提示文字
    def set_progress(self, value, text=""):
        self.progress.setValue(value)
        if text:
            self.label.setText(text)

# ===================== 工作线程（异步执行） =====================
# 功能：将耗时操作（加载CT、生成掩码）放到子线程，避免UI卡死
class Worker(QThread):
    # 自定义信号
    progress = Signal(int, str)    # 进度更新信号
    finished = Signal()            # 完成信号
    error = Signal(str)            # 错误信号
    success_msg = Signal(str)       # 成功提示信号

    def __init__(self, func, *args, parent=None):
        super().__init__(parent)
        self.func = func    # 要执行的耗时函数
        self.args = args    # 函数参数

    def run(self):
        # 线程启动后自动执行
        try:
            self.func(self.report_progress, *self.args)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    # 向外发送进度更新
    def report_progress(self, value, text=""):
        self.progress.emit(value, text)

# ===================== 主界面 =====================
# 主窗口：CT金属伪影标注 + 三平面视图 + VTK 3D重建
class ArtifactAnnotationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT 金属伪影检测 · 标注与 3D 重建 (DICOM / NIfTI)")
        self.resize(1680, 940)
        self.setStyleSheet(APP_QSS)

        # 全局变量：CT图像、掩码、numpy数组
        self.ct_image = None       # SimpleITK图像对象
        self.artifact_mask = None  # 伪影掩码图像
        self.patient_roi_mask = None  # 患者 ROI（排除床板/背景）
        self.volume_np = None      # CT体数据numpy数组 (z,y,x)
        self.mask_np = None        # 掩码numpy数组
        self.roi_np = None         # 患者 ROI numpy
        self.spacing = (1.0, 1.0, 1.0)  # (sx, sy, sz) mm，用于 MPR 物理宽高比
        self._ct_iso_vr: sitk.Image | None = None  # 与 3D 体绘制对齐的 CT 网格

        # 三个视图当前切片索引
        self.slice_axial = 0       # 轴位
        self.slice_coronal = 0     # 冠状
        self.slice_sagittal = 0    # 矢状

        # 初始化UI和VTK 3D窗口
        self.init_ui()
        self.init_vtk_3d_view()

    def _section_header(self, layout: QVBoxLayout, title: str, color: str) -> None:
        row = QHBoxLayout()
        dot = QLabel("■")
        dot.setStyleSheet(f"color: {color}; font-size: 10px;")
        lbl = QLabel(title)
        lbl.setObjectName("sectionTitle")
        lbl.setStyleSheet(f"color: {color};")
        row.addWidget(dot)
        row.addWidget(lbl)
        row.addStretch()
        layout.addLayout(row)

    def _add_param_slider(
        self,
        layout: QVBoxLayout,
        caption: str,
        slider: QSlider,
        value_label: QLabel,
    ) -> None:
        cap = QLabel(caption)
        cap.setObjectName("paramLabel")
        layout.addWidget(cap)
        row = QHBoxLayout()
        row.addWidget(slider, stretch=1)
        value_label.setObjectName("paramValue")
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(value_label)
        layout.addLayout(row)

    def _make_view_panel(
        self,
        title: str,
        min_h: int,
    ) -> tuple[QVBoxLayout, QLabel, QLabel, QSlider]:
        panel = QFrame()
        panel.setObjectName("viewerPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setObjectName("viewTitle")
        slice_lbl = QLabel("切片：0 / 0")
        slice_lbl.setObjectName("sliceBadge")
        header.addWidget(title_lbl)
        header.addStretch()
        header.addWidget(slice_lbl)
        layout.addLayout(header)

        image_lbl = QLabel()
        image_lbl.setObjectName("imageViewport")
        image_lbl.setAlignment(Qt.AlignCenter)
        image_lbl.setScaledContents(False)
        image_lbl.setMinimumHeight(min_h)
        slider = QSlider(Qt.Horizontal)
        layout.addWidget(image_lbl, stretch=1)
        layout.addWidget(slider)
        return layout, image_lbl, slice_lbl, slider, panel

    # -------------------------------------------------------------------------
    # 初始化整个界面布局：左侧控制面板 + 右侧三平面视图 + VTK 3D视图
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # ========== 左侧控制面板 ==========
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(360)
        sidebar_outer = QVBoxLayout(sidebar)
        sidebar_outer.setContentsMargins(16, 16, 16, 16)
        sidebar_outer.setSpacing(10)

        title = QLabel("CT 金属伪影检测\n标注与 3D 重建")
        title.setObjectName("appTitle")
        sidebar_outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_body = QWidget()
        control_layout = QVBoxLayout(scroll_body)
        control_layout.setContentsMargins(0, 0, 4, 0)
        control_layout.setSpacing(4)

        self._section_header(control_layout, "文件操作", UI_MUTED)
        self.btn_load = QPushButton("加载 DICOM 序列文件夹")
        self.btn_load.setObjectName("btnGrey")
        self.btn_load.clicked.connect(self.load_dicom_async)
        control_layout.addWidget(self.btn_load)

        self.btn_dicom_to_nifti = QPushButton("DICOM 转 NIfTI (.nii.gz)")
        self.btn_dicom_to_nifti.setObjectName("btnGrey")
        self.btn_dicom_to_nifti.clicked.connect(self.dicom_to_nifti_async)
        control_layout.addWidget(self.btn_dicom_to_nifti)

        self.btn_load_mhd = QPushButton("加载 MHD / RAW 三维影像")
        self.btn_load_mhd.setObjectName("btnGrey")
        self.btn_load_mhd.clicked.connect(self.load_mhd_async)
        control_layout.addWidget(self.btn_load_mhd)

        self.btn_load_single = QPushButton("加载单张图像 (DCM / PNG)")
        self.btn_load_single.setObjectName("btnGrey")
        self.btn_load_single.clicked.connect(self.load_single_async)
        control_layout.addWidget(self.btn_load_single)

        self.btn_load_nifti = QPushButton("加载NIFTI影像（.nii.gz）")
        self.btn_load_nifti.setObjectName("btnGrey")
        self.btn_load_nifti.clicked.connect(self.load_nifti_async)
        control_layout.addWidget(self.btn_load_nifti)

        control_layout.addSpacing(8)
        self._section_header(control_layout, "阈值参数（标注方法.md）", UI_ACCENT_ORANGE)
        p = DOC_DEFAULT_MASK_PARAMS
        self.th_low = QSlider(Qt.Horizontal)
        self.th_low.setRange(-500, 0)
        self.th_low.setValue(int(p.lower_hu))
        self.th_low_label = QLabel(str(int(p.lower_hu)))
        self._add_param_slider(
            control_layout, "低 HU 阈值 (<) · 金属伪影", self.th_low, self.th_low_label
        )

        self.th_high = QSlider(Qt.Horizontal)
        self.th_high.setRange(500, 4000)
        self.th_high.setValue(int(p.upper_hu))
        self.th_high_label = QLabel(str(int(p.upper_hu)))
        self._add_param_slider(
            control_layout, "高 HU 阈值 (>) · 金属伪影", self.th_high, self.th_high_label
        )

        self.grad_th = QSlider(Qt.Horizontal)
        self.grad_th.setRange(50, 500)
        self.grad_th.setValue(int(p.gradient_threshold))
        self.grad_th_label = QLabel(str(int(p.gradient_threshold)))
        self._add_param_slider(control_layout, "梯度阈值 · 条纹筛选", self.grad_th, self.grad_th_label)

        control_layout.addSpacing(6)
        self._section_header(control_layout, "形态学参数", UI_MUTED)
        self.open_r = QSlider(Qt.Horizontal)
        self.open_r.setRange(0, 5)
        self.open_r.setValue(p.opening_radius)
        self.open_r_label = QLabel(str(p.opening_radius))
        self._add_param_slider(control_layout, "开运算半径", self.open_r, self.open_r_label)

        self.close_r = QSlider(Qt.Horizontal)
        self.close_r.setRange(0, 10)
        self.close_r.setValue(p.closing_radius)
        self.close_r_label = QLabel(str(p.closing_radius))
        self._add_param_slider(control_layout, "闭运算半径", self.close_r, self.close_r_label)

        self.min_area = QSlider(Qt.Horizontal)
        self.min_area.setRange(10, 500)
        self.min_area.setValue(p.min_component_size)
        self.min_area_label = QLabel(str(p.min_component_size))
        self._add_param_slider(control_layout, "最小面积 · 连通域", self.min_area, self.min_area_label)

        control_layout.addSpacing(10)
        self._section_header(control_layout, "掩码操作", UI_ACCENT_GREEN)
        self.btn_run = QPushButton("生成伪影掩码")
        self.btn_run.setObjectName("btnGreen")
        self.btn_run.clicked.connect(self.generate_mask_async)
        control_layout.addWidget(self.btn_run)

        self.btn_save = QPushButton("保存掩码 (NIfTI / NRRD)")
        self.btn_save.setObjectName("btnBlue")
        self.btn_save.clicked.connect(self.save_mask_async)
        control_layout.addWidget(self.btn_save)

        control_layout.addStretch()
        scroll.setWidget(scroll_body)
        sidebar_outer.addWidget(scroll, stretch=1)

        self.status_hint = QLabel(
            f"NIfTI：先选影像，再可选标注掩码查看效果。"
            f"窗位 {int(ANNOTATION_WINDOW_LEVEL)} / 窗宽 {int(ANNOTATION_WINDOW_WIDTH)}。"
            f"生成掩码前会自动建立患者 ROI。"
        )
        self.status_hint.setObjectName("statusHint")
        self.status_hint.setWordWrap(True)
        sidebar_outer.addWidget(self.status_hint)

        main_layout.addWidget(sidebar)

        # ========== 右侧影像区 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        axial_vtk_layout = QHBoxLayout()
        axial_vtk_layout.setSpacing(12)

        axial_layout, self.axial_label, self.axial_slice_label, self.slider_axial, axial_panel = (
            self._make_view_panel("轴位 Axial", 300)
        )
        self.slider_axial.valueChanged.connect(lambda v: self.on_slice_change("axial", v))
        axial_vtk_layout.addWidget(axial_panel, stretch=3)

        vtk_panel = QFrame()
        vtk_panel.setObjectName("viewerPanel")
        vtk_box = QVBoxLayout(vtk_panel)
        vtk_box.setContentsMargins(12, 10, 12, 10)
        vtk_box.setSpacing(8)
        vtk_title = QLabel("3D 体渲染 · VTK")
        vtk_title.setObjectName("viewTitle")
        vtk_box.addWidget(vtk_title)
        self.vtk_widget = QWidget()
        self.vtk_widget.setObjectName("imageViewport")
        self.vtk_widget.setStyleSheet(f"background-color: {UI_VIEWPORT_BG}; border-radius: 8px;")
        self.vtk_widget.setMinimumWidth(380)
        self.vtk_widget.setMinimumHeight(300)
        vtk_box.addWidget(self.vtk_widget, stretch=1)
        vtk_hint = QLabel("鼠标拖拽旋转 · 滚轮缩放")
        vtk_hint.setObjectName("statusHint")
        vtk_box.addWidget(vtk_hint)
        axial_vtk_layout.addWidget(vtk_panel, stretch=2)

        right_layout.addLayout(axial_vtk_layout, stretch=5)

        cor_sag_layout = QHBoxLayout()
        cor_sag_layout.setSpacing(12)

        cor_layout, self.coronal_label, self.coronal_slice_label, self.slider_coronal, cor_panel = (
            self._make_view_panel("冠状 Coronal", 380)
        )
        self.slider_coronal.valueChanged.connect(lambda v: self.on_slice_change("coronal", v))
        cor_sag_layout.addWidget(cor_panel)

        sag_layout, self.sagittal_label, self.sagittal_slice_label, self.slider_sagittal, sag_panel = (
            self._make_view_panel("矢状 Sagittal", 380)
        )
        self.slider_sagittal.valueChanged.connect(lambda v: self.on_slice_change("sagittal", v))
        cor_sag_layout.addWidget(sag_panel)

        right_layout.addLayout(cor_sag_layout, stretch=6)
        main_layout.addWidget(right_widget, stretch=1)

        self.connect_all_sliders()

    # -------------------------------------------------------------------------
    # VTK 3D 视图初始化
    # 创建VTK渲染器、窗口、交互器，嵌入Qt界面
    def init_vtk_3d_view(self):
        suppress_vtk_output()
        self.vtk_view = QVTKRenderWindowInteractor(self.vtk_widget)
        self.vtk_renderer = vtk.vtkRenderer()                  # VTK渲染器（场景）
        self.vtk_renderer.SetBackground(0.1, 0.1, 0.1)         # 背景深灰色
        self.vtk_view.GetRenderWindow().AddRenderer(self.vtk_renderer)
        self.iren = self.vtk_view.GetRenderWindow().GetInteractor()
        self.iren.Initialize()  # 初始化交互

        # 把VTK窗口塞进Qt组件
        layout = QVBoxLayout(self.vtk_widget)
        layout.addWidget(self.vtk_view)
        layout.setContentsMargins(0,0,0,0)

    def _make_ghost_volume_property(self) -> vtk.vtkVolumeProperty:
        """参考图：半透明灰褐色头颅，不遮挡金属掩码。"""
        opacity = vtk.vtkPiecewiseFunction()
        color = vtk.vtkColorTransferFunction()
        opacity.AddPoint(-1000, 0.0)
        opacity.AddPoint(-300, 0.0)
        opacity.AddPoint(-80, 0.02)
        opacity.AddPoint(40, 0.05)
        opacity.AddPoint(120, 0.08)
        opacity.AddPoint(300, 0.11)
        opacity.AddPoint(800, 0.14)
        opacity.AddPoint(2000, 0.16)

        color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        color.AddRGBPoint(-80, 0.48, 0.42, 0.36)
        color.AddRGBPoint(40, 0.68, 0.60, 0.52)
        color.AddRGBPoint(200, 0.78, 0.72, 0.64)
        color.AddRGBPoint(800, 0.85, 0.80, 0.74)

        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color)
        prop.SetScalarOpacity(opacity)
        prop.SetInterpolationTypeToLinear()
        prop.ShadeOn()
        prop.SetAmbient(0.35)
        prop.SetDiffuse(0.72)
        prop.SetSpecular(0.08)
        prop.SetSpecularPower(12.0)
        return prop

    def _resample_mask_to_vr_space(self, mask: sitk.Image) -> sitk.Image:
        """将掩码重采样到与 3D 体绘制相同的 CT 网格。"""
        if self._ct_iso_vr is None:
            self._ct_iso_vr = prepare_ct_for_vr(self.ct_image)
        return sitk.Resample(
            sitk.Cast(mask, sitk.sitkUInt8),
            self._ct_iso_vr,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0.0,
            sitk.sitkUInt8,
        )

    def _mask_volume_actor(
        self,
        mask_iso: sitk.Image,
        color: tuple[float, float, float],
    ) -> vtk.vtkVolume | None:
        """掩码体绘制：高不透明度，保证金属环在 3D 中可见。"""
        if int(sitk.GetArrayViewFromImage(mask_iso).sum()) < 20:
            return None

        vtk_mask = sitk_to_vtk_image(sitk.Cast(mask_iso, sitk.sitkFloat32))
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_mask)
        mapper.SetBlendModeToComposite()

        opacity = vtk.vtkPiecewiseFunction()
        color_tf = vtk.vtkColorTransferFunction()
        opacity.AddPoint(0.0, 0.0)
        opacity.AddPoint(0.5, 0.0)
        opacity.AddPoint(1.0, 0.92)
        color_tf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_tf.AddRGBPoint(1.0, *color)

        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color_tf)
        prop.SetScalarOpacity(opacity)
        prop.ShadeOn()
        prop.SetAmbient(0.7)
        prop.SetDiffuse(0.85)
        prop.SetSpecular(0.95)
        prop.SetSpecularPower(48.0)

        vol = vtk.vtkVolume()
        vol.SetMapper(mapper)
        vol.SetProperty(prop)
        return vol

    # -------------------------------------------------------------------------
    # 3D CT 重建：半透明头颅 + 高亮多色金属掩码（参考效果图）
    def render_3d_ct(self, *, show_mask: bool = False):
        if self.ct_image is None:
            return

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.SetBackground(0.0, 0.0, 0.0)
        # 深度剥离在部分显卡上会触发 VTK 警告弹窗，关闭后更稳定
        self.vtk_renderer.SetUseDepthPeeling(0)

        self._ct_iso_vr = prepare_ct_for_vr(self.ct_image)
        vtk_ct = sitk_to_vtk_image(self._ct_iso_vr)

        # 幽灵体绘制：整体半透明
        ct_volume = vtk.vtkVolume()
        ct_volume.SetMapper(_create_volume_mapper(vtk_ct, "soft"))
        ct_volume.SetProperty(self._make_ghost_volume_property())
        self.vtk_renderer.AddVolume(ct_volume)

        # 极淡骨皮质外壳
        bone_actor = _bone_surface_actor(
            vtk_ct,
            iso_value=280.0,
            opacity=0.10 if show_mask else 0.28,
            color=(0.88, 0.84, 0.76),
        )
        if bone_actor is not None:
            self.vtk_renderer.AddActor(bone_actor)

        if show_mask and self.artifact_mask is not None:
            mask_iso = self._resample_mask_to_vr_space(self.artifact_mask)
            mask_sum = int(sitk.GetArrayViewFromImage(mask_iso).sum())
            if mask_sum >= 20:
                # 分连通域：青 / 绿 / 黄 / 橙 实体表面
                for actor in self._mask_component_actors(mask_iso, self._ct_iso_vr):
                    self.vtk_renderer.AddActor(actor)
                # 整体掩码体绘制兜底（高亮黄橙色，类似参考图中的金属环）
                glow = self._mask_volume_actor(mask_iso, (1.0, 0.88, 0.28))
                if glow is not None:
                    self.vtk_renderer.AddVolume(glow)
        elif self.artifact_mask is None:
            metal_actor = _metal_hu_surface_actor(vtk_ct, iso_value=2000.0)
            if metal_actor is not None:
                self.vtk_renderer.AddActor(metal_actor)

        self.vtk_renderer.ResetCamera()
        cam = self.vtk_renderer.GetActiveCamera()
        cam.Azimuth(32)
        cam.Elevation(-18)
        cam.Dolly(1.12)
        self.vtk_renderer.ResetCameraClippingRange()
        self.vtk_view.GetRenderWindow().Render()

    def _mask_component_actors(self, mask_iso: sitk.Image, reference: sitk.Image) -> list:
        """将掩码各连通域提取为不同颜色的 3D 表面。"""
        mask_u8 = sitk.Cast(mask_iso, sitk.sitkUInt8)
        if int(sitk.GetArrayViewFromImage(mask_u8).sum()) < 20:
            return []

        labeled = sitk.RelabelComponent(sitk.ConnectedComponent(mask_u8), sortByObjectSize=True)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(labeled)

        actors: list = []
        labels = list(stats.GetLabels())
        for i, lbl in enumerate(labels[:16]):
            if stats.GetNumberOfPixels(lbl) < 15:
                continue
            comp = sitk.BinaryThreshold(labeled, lbl, lbl, 1, 0)
            color = METAL_COMPONENT_COLORS[i % len(METAL_COMPONENT_COLORS)]
            actor = self._single_mask_surface_actor(comp, reference, color)
            if actor is not None:
                actors.append(actor)
        return actors

    def _single_mask_surface_actor(
        self,
        mask_bin: sitk.Image,
        reference: sitk.Image,
        color: tuple[float, float, float],
    ) -> vtk.vtkActor | None:
        if mask_bin.GetSize() != reference.GetSize():
            mask_rs = sitk.Resample(
                mask_bin,
                reference,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0.0,
                mask_bin.GetPixelID(),
            )
        else:
            mask_rs = mask_bin

        if int(sitk.GetArrayViewFromImage(mask_rs).sum()) < 15:
            return None

        vtk_mask = sitk_to_vtk_image(sitk.Cast(mask_rs, sitk.sitkFloat32))

        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(vtk_mask)
        mc.SetValue(0, 0.5)
        mc.ComputeNormalsOn()
        mc.Update()
        if mc.GetOutput().GetNumberOfPoints() < 15:
            return None

        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(mc.GetOutputPort())
        smooth.SetNumberOfIterations(25)
        smooth.SetRelaxationFactor(0.12)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOn()
        smooth.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smooth.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetOpacity(1.0)
        prop.SetAmbient(0.65)
        prop.SetDiffuse(0.90)
        prop.SetSpecular(1.0)
        prop.SetSpecularPower(55.0)
        prop.SetRepresentationToSurface()
        return actor

    def _release_previous_study(self) -> None:
        """加载新数据前释放上一套 CT / 掩码 / 3D 缓存。"""
        if self.ct_image is None and self.volume_np is None:
            return

        self.ct_image = None
        self.artifact_mask = None
        self.patient_roi_mask = None
        self.volume_np = None
        self.mask_np = None
        self.roi_np = None
        self._ct_iso_vr = None
        self.spacing = (1.0, 1.0, 1.0)
        self.slice_axial = 0
        self.slice_coronal = 0
        self.slice_sagittal = 0

        if hasattr(self, "vtk_renderer"):
            self.vtk_renderer.RemoveAllViewProps()
            if hasattr(self, "vtk_view"):
                self.vtk_view.GetRenderWindow().Render()

        self._clear_view_pixmaps()
        gc.collect()

    def _clear_view_pixmaps(self) -> None:
        blank = QPixmap()
        for label in (self.axial_label, self.coronal_label, self.sagittal_label):
            label.setPixmap(blank)
        for tag, text in (
            (self.axial_slice_label, "切片：0 / 0"),
            (self.coronal_slice_label, "切片：0 / 0"),
            (self.sagittal_slice_label, "切片：0 / 0"),
        ):
            tag.setText(text)

    # -------------------------------------------------------------------------
    # 绑定参数滑动条：仅更新数值标签，不自动运算（须点击「生成伪影掩码」）
    def connect_all_sliders(self):
        self.th_low.valueChanged.connect(lambda v: self.th_low_label.setText(str(v)))
        self.th_high.valueChanged.connect(lambda v: self.th_high_label.setText(str(v)))
        self.grad_th.valueChanged.connect(lambda v: self.grad_th_label.setText(str(v)))
        self.open_r.valueChanged.connect(lambda v: self.open_r_label.setText(str(v)))
        self.close_r.valueChanged.connect(lambda v: self.close_r_label.setText(str(v)))
        self.min_area.valueChanged.connect(lambda v: self.min_area_label.setText(str(v)))

    # -------------------------------------------------------------------------
    # 统一异步任务启动封装：显示进度条 + 启动线程
    def run_async(self, title, func, *args, on_finished=None):
        self.dialog = ProgressDialog(title, self)
        self.worker = Worker(func, *args)
        self.worker.progress.connect(self.dialog.set_progress)
        self.worker.finished.connect(self.dialog.close)
        self.worker.finished.connect(on_finished or self.on_load_task_done)
        self.worker.error.connect(lambda e: (self.dialog.close(), QMessageBox.critical(self, "错误", e)))
        self.worker.success_msg.connect(self.on_success_msg)
        self.dialog.show()
        self.worker.start()

    def _ensure_roi_for_display(self) -> None:
        """有标注但无 ROI 时，生成患者轮廓用于压暗背景。"""
        if self.ct_image is None or self.roi_np is not None:
            return
        self.patient_roi_mask = generate_patient_roi_mask(self.ct_image, DOC_DEFAULT_MASK_PARAMS)
        self.roi_np = sitk.GetArrayFromImage(self.patient_roi_mask)

    def _has_annotation_mask(self) -> bool:
        return self.mask_np is not None and int(self.mask_np.sum()) > 0

    def on_load_task_done(self):
        """加载影像后刷新视图；若已含标注则直接叠加显示。"""
        self.refresh_all_views()
        self.update_all_slice_labels()
        if self._has_annotation_mask():
            self._ensure_roi_for_display()
            self.render_3d_ct(show_mask=True)
            n = int(self.mask_np.sum())
            if hasattr(self, "status_hint"):
                self.status_hint.setText(f"已加载 NIfTI 标注（{n} 体素）。2D 红色叠加 · 3D 分色显示。")
        else:
            self.render_3d_ct()
            if hasattr(self, "status_hint"):
                self.status_hint.setText("影像已加载。可再次加载 NIfTI 标注，或点击「生成伪影掩码」。")

    def on_mask_task_done(self):
        """点击「生成伪影掩码」后：刷新 2D 叠加并强制在 VTK 3D 中显示掩码表面。"""
        self.refresh_all_views()
        self.update_all_slice_labels()
        self.render_3d_ct(show_mask=True)
        if hasattr(self, "status_hint"):
            n = int(self.mask_np.sum()) if self.mask_np is not None else 0
            roi_n = int(self.roi_np.sum()) if self.roi_np is not None else 0
            self.status_hint.setText(
                f"患者 ROI 已建立（{roi_n} 体素）；伪影掩码 {n} 体素（仅 ROI 内）。"
                f"2D 红色为伪影 · 3D 分色金属表面。"
            )

    # 成功提示
    def on_success_msg(self, msg):
        QMessageBox.information(self, "成功", msg)

    # -------------------------------------------------------------------------
    # 加载 MHD 文件（异步）
    def load_mhd_async(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 MHD 文件", "", "MHD Image (*.mhd *.MHD)")
        if not path:
            return
        self._release_previous_study()
        self.run_async("加载 MHD 中...", self._load_mhd_task, path)

    def _load_mhd_task(self, report, path):
        report(20, "读取MHD文件...")
        img = sitk.ReadImage(path)
        report(60, "解析三维数据...")
        vol = sitk.GetArrayFromImage(img)
        self.ct_image = img
        self.volume_np = vol
        self._apply_image_spacing(img)
        self.mask_np = None
        self.roi_np = None
        self.artifact_mask = None
        self.patient_roi_mask = None
        self.init_slice_range()
        report(100, "完成")

    def _apply_image_spacing(self, img: sitk.Image) -> None:
        sp = img.GetSpacing()
        self.spacing = (float(sp[0]), float(sp[1]), float(sp[2]))

    def _clear_mask_state(self) -> None:
        self.mask_np = None
        self.roi_np = None
        self.artifact_mask = None
        self.patient_roi_mask = None

    def _set_volume_from_ct(self, img: sitk.Image) -> None:
        img = _prepare_ct_image(img)
        self.ct_image = img
        self.volume_np = sitk.GetArrayFromImage(img)
        self._apply_image_spacing(img)

    def _set_mask_from_label(self, label: sitk.Image) -> int:
        if self.ct_image is None:
            raise RuntimeError("请先加载 CT 影像")
        aligned = _align_mask_to_ct(label, self.ct_image)
        self.artifact_mask = aligned
        self.mask_np = sitk.GetArrayFromImage(aligned)
        return int(self.mask_np.sum())

    # -------------------------------------------------------------------------
    # 加载 NIfTI（影像 + 可选标注）
    def load_nifti_async(self) -> None:
        ct_path, _ = QFileDialog.getOpenFileName(
            self, "选择 NIfTI 影像 (.nii.gz)", "", NIFTI_FILTER
        )
        if not ct_path:
            return
        label_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标注掩码（可选，取消则仅加载影像）",
            os.path.dirname(ct_path),
            NIFTI_FILTER,
        )
        self._release_previous_study()
        self.run_async(
            "加载 NIfTI 影像...",
            self._load_nifti_task,
            ct_path,
            label_path or "",
        )

    def _load_nifti_task(self, report, ct_path: str, label_path: str) -> None:
        report(15, "读取 CT NIfTI...")
        img = _read_sitk_image(ct_path)
        self._set_volume_from_ct(img)
        self._clear_mask_state()
        self.init_slice_range()
        if label_path:
            report(55, "读取并对齐标注掩码...")
            n = self._set_mask_from_label(_read_sitk_image(label_path))
            if n < 1:
                raise RuntimeError("标注掩码无前景体素")
            report(80, "生成 ROI 显示...")
            self._ensure_roi_for_display()
        report(100, "完成")

    # -------------------------------------------------------------------------
    # 加载 DICOM 序列（异步）
    def load_dicom_async(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 DICOM 序列文件夹")
        if not folder:
            return
        self._release_previous_study()
        self.run_async("加载 DICOM 中...", self._load_dicom_task, folder)

    def _load_dicom_task(self, report, folder):
        report(10, "扫描 DICOM 序列...")
        img = read_dicom_folder(folder)
        report(70, "构建三维体...")
        vol = sitk.GetArrayFromImage(img)
        self.ct_image = img
        self.volume_np = vol
        self._apply_image_spacing(img)
        self.mask_np = None
        self.roi_np = None
        self.artifact_mask = None
        self.patient_roi_mask = None
        self.init_slice_range()
        report(100, "完成")

    # -------------------------------------------------------------------------
    # DICOM 序列 -> NIfTI（仅格式转换，保留 spacing/origin/direction/像素类型）
    def dicom_to_nifti_async(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择 DICOM 序列文件夹")
        if not folder:
            return
        folder = os.path.normpath(folder)
        default_name = f"{os.path.basename(folder)}.nii.gz"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 NIfTI (.nii.gz)",
            os.path.join(folder, default_name),
            "NIfTI (*.nii.gz *.nii)",
        )
        if not out_path:
            return
        save_path = _normalize_mask_save_path(out_path)
        if save_path != os.path.abspath(out_path):
            QMessageBox.information(
                self,
                "文件名已调整",
                f"已补全扩展名：\n{save_path}",
            )
        if os.path.exists(save_path):
            yes = QMessageBox.question(
                self,
                "文件已存在",
                f"{save_path}\n已存在，是否覆盖？",
            )
            if yes != QMessageBox.Yes:
                return
        self.run_async(
            "DICOM 转 NIfTI...",
            self._dicom_to_nifti_task,
            folder,
            save_path,
            on_finished=self.on_dicom_export_done,
        )

    def _dicom_to_nifti_task(self, report, folder: str, out_path: str) -> None:
        report(15, "读取 DICOM 序列（保留元数据）...")
        img = read_dicom_folder(folder)
        report(55, "写入 NIfTI (.nii.gz)...")
        saved = _write_mask_image(img, out_path)
        sp = img.GetSpacing()
        sz = img.GetSize()
        self._last_export_meta = {
            "path": saved,
            "size": sz,
            "spacing": sp,
            "origin": img.GetOrigin(),
            "direction": img.GetDirection(),
            "pixel_type": img.GetPixelIDTypeAsString(),
        }
        report(100, "转换完成")

    def on_dicom_export_done(self) -> None:
        meta = getattr(self, "_last_export_meta", {})
        path = meta.get("path", "")
        sz = meta.get("size", (0, 0, 0))
        sp = meta.get("spacing", (0.0, 0.0, 0.0))
        msg = (
            f"已保存：\n{path}\n\n"
            f"体素: {sz[0]} × {sz[1]} × {sz[2]}\n"
            f"间距 (mm): {sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f}\n"
            f"像素类型: {meta.get('pixel_type', '')}"
        )
        if hasattr(self, "status_hint"):
            self.status_hint.setText(f"DICOM 已转为 NIfTI：{os.path.basename(path)}")
        self.worker.success_msg.emit(msg)

    # -------------------------------------------------------------------------
    # 加载单张图像（异步）
    def load_single_async(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "所有图像 (*.dcm *.png *.jpg *.nii *.mhd)")
        if not path:
            return
        self._release_previous_study()
        self.run_async("加载单张图像...", self._load_single_task, path)

    def _load_single_task(self, report, path):
        img = sitk.ReadImage(path)
        vol = sitk.GetArrayFromImage(img)
        if vol.ndim == 2:
            vol = vol[None]
            img = sitk.GetImageFromArray(vol)
        self.ct_image = img
        self.volume_np = vol
        self._apply_image_spacing(img)
        self.mask_np = None
        self.roi_np = None
        self.artifact_mask = None
        self.patient_roi_mask = None
        self.init_slice_range()
        report(100)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.volume_np is not None:
            QTimer.singleShot(80, self.refresh_all_views)

    # -------------------------------------------------------------------------
    # 生成伪影掩码（异步，核心算法）
    def generate_mask_async(self):
        if self.ct_image is None:
            QMessageBox.warning(self, "提示", "请先加载 DICOM / MHD 影像")
            return
        self.run_async("生成伪影掩码...", self._generate_mask_task, on_finished=self.on_mask_task_done)

    def _read_mask_params_from_ui(self) -> MaskParams:
        p = DOC_DEFAULT_MASK_PARAMS
        return MaskParams(
            lower_hu=float(self.th_low.value()),
            upper_hu=float(self.th_high.value()),
            gradient_threshold=float(self.grad_th.value()),
            opening_radius=int(self.open_r.value()),
            closing_radius=int(self.close_r.value()),
            min_component_size=int(self.min_area.value()),
            use_gradient=True,
            keep_largest_only=False,
            use_patient_roi=True,
            roi_air_hu=p.roi_air_hu,
            roi_closing_radius=p.roi_closing_radius,
            roi_opening_radius=p.roi_opening_radius,
            roi_dilate_radius=p.roi_dilate_radius,
        )

    # 真正执行掩码生成：患者 ROI -> ROI 内金属伪影检测
    def _generate_mask_task(self, report):
        img = self.ct_image
        params = self._read_mask_params_from_ui()

        report(10, "分割患者/颅骨 ROI（排除床板与背景）...")
        self.patient_roi_mask = generate_patient_roi_mask(img, params)
        self.roi_np = sitk.GetArrayFromImage(self.patient_roi_mask)

        report(35, "ROI 内金属伪影双阈值检测...")
        report(60, "梯度约束与形态学处理...")
        final = generate_metal_mask(img, params, patient_roi=self.patient_roi_mask)
        report(90, "生成最终掩码...")

        self.artifact_mask = final
        self.mask_np = sitk.GetArrayFromImage(final)
        report(100, "处理完成")

    # -------------------------------------------------------------------------
    # 保存掩码（异步）
    def save_mask_async(self):
        if self.artifact_mask is None:
            QMessageBox.warning(self, "提示", "先生成掩码！")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存掩码",
            "artifact_mask.nii.gz",
            "NIfTI 掩码 (*.nii.gz *.nii);;NRRD 掩码 (*.nrrd);;所有文件 (*.*)",
        )
        if not path:
            return
        save_path = _normalize_mask_save_path(path)
        if save_path != os.path.abspath(path):
            QMessageBox.information(
                self,
                "文件名已调整",
                f"扩展名已自动补全为可识别格式：\n{save_path}",
            )
        if os.path.exists(save_path):
            yes = QMessageBox.question(self, "文件已存在", f"{save_path}\n已存在，是否覆盖？")
            if yes != QMessageBox.Yes:
                return
        self.run_async("保存掩码中...", self._save_mask_task, save_path)

    def _save_mask_task(self, report, path):
        report(40, "写入掩码文件...")
        out_path = _write_mask_image(self.artifact_mask, path)
        report(100, "保存完成")
        self.worker.success_msg.emit(f"掩码已保存到：\n{out_path}")

    # -------------------------------------------------------------------------
    # 更新切片序号显示
    def update_all_slice_labels(self):
        if self.volume_np is None: return
        d, h, w = self.volume_np.shape
        self.axial_slice_label.setText(f"切片：{self.slice_axial} / {d-1}")
        self.coronal_slice_label.setText(f"切片：{self.slice_coronal} / {h-1}")
        self.sagittal_slice_label.setText(f"切片：{self.slice_sagittal} / {w-1}")

    # 初始化切片范围，默认显示中间层
    def init_slice_range(self):
        d, h, w = self.volume_np.shape
        self.slider_axial.setRange(0, d-1)
        self.slider_coronal.setRange(0, h-1)
        self.slider_sagittal.setRange(0, w-1)
        sa, sc, ss = d//2, h//2, w//2
        self.slice_axial, self.slice_coronal, self.slice_sagittal = sa, sc, ss
        self.slider_axial.setValue(sa)
        self.slider_coronal.setValue(sc)
        self.slider_sagittal.setValue(ss)
        self.update_all_slice_labels()

    # -------------------------------------------------------------------------
    # ✅ 核心：三平面同步滑动逻辑
    # 滑动任意一个视图，另外两个自动同步
    def on_slice_change(self, vt, v):
        if self.volume_np is None:
            return

        # 获取当前体数据尺寸 (z,y,x)
        d, h, w = self.volume_np.shape

        # 轴位滑动 → 同步冠状、矢状
        if vt == "axial":
            self.slice_axial = v
            self.slice_coronal = int((v / d) * h)
            self.slice_sagittal = int((v / d) * w)

        # 冠状滑动 → 同步轴位、矢状
        elif vt == "coronal":
            self.slice_coronal = v
            self.slice_axial = int((v / h) * d)
            self.slice_sagittal = int((v / h) * w)

        # 矢状滑动 → 同步轴位、冠状
        elif vt == "sagittal":
            self.slice_sagittal = v
            self.slice_axial = int((v / w) * d)
            self.slice_coronal = int((v / w) * h)

        # 防止索引越界
        self.slice_axial = np.clip(self.slice_axial, 0, d-1)
        self.slice_coronal = np.clip(self.slice_coronal, 0, h-1)
        self.slice_sagittal = np.clip(self.slice_sagittal, 0, w-1)

        # 暂时阻塞信号，避免循环触发
        self.slider_axial.blockSignals(True)
        self.slider_axial.setValue(self.slice_axial)
        self.slider_axial.blockSignals(False)

        self.slider_coronal.blockSignals(True)
        self.slider_coronal.setValue(self.slice_coronal)
        self.slider_coronal.blockSignals(False)

        self.slider_sagittal.blockSignals(True)
        self.slider_sagittal.setValue(self.slice_sagittal)
        self.slider_sagittal.blockSignals(False)

        # 刷新所有视图
        self.refresh_all_views()
        self.update_all_slice_labels()

    # -------------------------------------------------------------------------
    # 绘制单张切片：CT 窗宽窗位 + ROI 外压暗 + 伪影红色叠加
    def draw_slice(self, im, mk, roi=None):
        gray = apply_window(
            im.astype(np.float32),
            ANNOTATION_WINDOW_LEVEL,
            ANNOTATION_WINDOW_WIDTH,
        )
        c = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        if roi is not None and roi.shape == im.shape:
            c[roi == 0] *= 0.22
        c = np.clip(c, 0, 255).astype(np.uint8)
        if mk is not None and mk.shape == im.shape:
            c[mk > 0] = [255, 80, 80]
        return c

    def _label_for_view(self, view_type: str) -> QLabel:
        return {
            "axial": self.axial_label,
            "coronal": self.coronal_label,
            "sagittal": self.sagittal_label,
        }[view_type]

    def _rgb_to_pixmap(self, rgb: np.ndarray, view_type: str) -> QPixmap:
        """按体素间距保持物理宽高比，避免冠状/矢状被压成横条。"""
        if self.volume_np is None:
            return QPixmap()

        vol_shape = self.volume_np.shape
        rgb = upscale_mpr_slice_for_display(rgb, view_type, vol_shape, self.spacing)

        pw, ph = physical_slice_size(vol_shape, self.spacing, view_type)
        label = self._label_for_view(view_type)
        max_w = max(160, label.width() - 8)
        max_h = max(160, label.height() - 8)
        if max_w < 100 or max_h < 100:
            max_w, max_h = 640, 420

        scale = min(max_w / max(pw, 1e-6), max_h / max(ph, 1e-6))
        out_w = max(1, int(pw * scale))
        out_h = max(1, int(ph * scale))

        h, w = rgb.shape[:2]
        if w != out_w or h != out_h:
            rgb = resize_bilinear(rgb, out_h, out_w)

        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, out_w, out_h, 3 * out_w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    # -------------------------------------------------------------------------
    # 更新三个视图显示
    def update_view_axial(self):
        if self.volume_np is None:
            return
        im = self.volume_np[self.slice_axial]
        mk = self.mask_np[self.slice_axial] if self.mask_np is not None else None
        roi = self.roi_np[self.slice_axial] if self.roi_np is not None else None
        rgb = self.draw_slice(im, mk, roi)
        self.axial_label.setPixmap(self._rgb_to_pixmap(rgb, "axial"))

    def update_view_coronal(self):
        if self.volume_np is None:
            return
        im = self.volume_np[:, self.slice_coronal, :]
        mk = self.mask_np[:, self.slice_coronal, :] if self.mask_np is not None else None
        roi = self.roi_np[:, self.slice_coronal, :] if self.roi_np is not None else None
        rgb = self.draw_slice(im, mk, roi)
        self.coronal_label.setPixmap(self._rgb_to_pixmap(rgb, "coronal"))

    def update_view_sagittal(self):
        if self.volume_np is None:
            return
        im = self.volume_np[:, :, self.slice_sagittal]
        mk = self.mask_np[:, :, self.slice_sagittal] if self.mask_np is not None else None
        roi = self.roi_np[:, :, self.slice_sagittal] if self.roi_np is not None else None
        rgb = self.draw_slice(im, mk, roi)
        self.sagittal_label.setPixmap(self._rgb_to_pixmap(rgb, "sagittal"))

    def refresh_all_views(self):
        self.update_view_axial()
        self.update_view_coronal()
        self.update_view_sagittal()

# -------------------------------------------------------------------------
# 主程序入口
if __name__ == "__main__":
    suppress_vtk_output()
    app = QApplication(sys.argv)
    win = ArtifactAnnotationWindow()
    win.show()
    code = app.exec()
    win._release_previous_study()
    sys.exit(code)