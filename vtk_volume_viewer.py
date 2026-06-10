"""VTK 体绘制（供 Dear PyGui 离屏纹理），传递函数与 nrrd.py 一致。"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support

try:
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
except ImportError:
    pass


class _SuppressedVtkOutputWindow(vtk.vtkOutputWindow):
    def DisplayText(self, text: str) -> None:
        if text and text.strip():
            print(f"[VTK] {text.strip()}")

    def DisplayErrorText(self, text: str) -> None:
        if text and text.strip():
            print(f"[VTK ERROR] {text.strip()}")

    def DisplayWarningText(self, text: str) -> None:
        if text and text.strip():
            print(f"[VTK WARN] {text.strip()}")

    def DisplayGenericWarningText(self, text: str) -> None:
        self.DisplayWarningText(text)

    def DisplayDebugText(self, text: str) -> None:
        pass


def suppress_vtk_output() -> None:
    vtk.vtkObject.GlobalWarningDisplayOff()
    vtk.vtkOutputWindow.SetInstance(_SuppressedVtkOutputWindow())


def _crop_body(ct: sitk.Image, threshold: float = -400.0) -> sitk.Image:
    """裁掉体外大量空气，避免填充值干扰体绘制。"""
    arr = sitk.GetArrayFromImage(ct)
    inside = arr > threshold
    if not bool(np.any(inside)):
        return ct

    coords = np.argwhere(inside)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    size = [int(x1 - x0), int(y1 - y0), int(z1 - z0)]
    index = [int(x0), int(y0), int(z0)]
    return sitk.RegionOfInterest(ct, size, index)


def resample_isotropic(image: sitk.Image, spacing_mm: float) -> sitk.Image:
    size = image.GetSize()
    sp = image.GetSpacing()
    new_size = [max(1, int(round(s * o / spacing_mm))) for s, o in zip(size, sp)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([spacing_mm] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000.0)
    return resampler.Execute(image)


def prepare_ct_for_vr(ct_image: sitk.Image) -> sitk.Image:
    """标准朝向 + 体部裁剪 + 各向同性重采样。"""
    ct = sitk.Cast(ct_image, sitk.sitkFloat32)
    try:
        ct = sitk.DICOMOrient(ct, "RAI")
    except Exception:  # noqa: BLE001
        pass
    ct = _crop_body(ct, threshold=-400.0)
    iso = resample_isotropic(ct, spacing_mm=1.0)
    size = iso.GetSize()
    max_dim = max(size)
    if max_dim > 256:
        factor = int(np.ceil(max_dim / 256))
        iso = sitk.Shrink(iso, [factor, factor, factor])
    return iso


def sitk_to_vtk_image(image: sitk.Image) -> vtk.vtkImageData:
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"仅支持 3D 体数据，当前维度: {array.ndim}")

    array = np.transpose(array, (2, 1, 0))

    vtk_image = vtk.vtkImageData()
    size = image.GetSize()
    vtk_image.SetDimensions(int(size[0]), int(size[1]), int(size[2]))
    vtk_image.SetSpacing(image.GetSpacing())
    vtk_image.SetOrigin(image.GetOrigin())

    flat = np.ascontiguousarray(array.ravel(order="F"))
    vtk_array = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image.GetPointData().SetScalars(vtk_array)
    return vtk_image


def make_volume_property(lo: float, hi: float, preset: str = "bone") -> vtk.vtkVolumeProperty:
    """体绘制传递函数；preset: bone | soft | vessel | metal | mip。"""
    opacity = vtk.vtkPiecewiseFunction()
    color = vtk.vtkColorTransferFunction()

    if lo < -200 and hi > 400:
        if preset == "soft":
            opacity.AddPoint(-1000, 0.0)
            opacity.AddPoint(-400, 0.0)
            opacity.AddPoint(-80, 0.18)
            opacity.AddPoint(60, 0.45)
            opacity.AddPoint(180, 0.55)
            opacity.AddPoint(400, 0.35)
            opacity.AddPoint(900, 0.20)

            color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
            color.AddRGBPoint(-80, 0.35, 0.22, 0.18)
            color.AddRGBPoint(60, 0.88, 0.72, 0.58)
            color.AddRGBPoint(180, 0.92, 0.88, 0.82)
            color.AddRGBPoint(600, 0.95, 0.93, 0.90)
        elif preset == "vessel":
            opacity.AddPoint(-1000, 0.0)
            opacity.AddPoint(-100, 0.0)
            opacity.AddPoint(80, 0.05)
            opacity.AddPoint(150, 0.35)
            opacity.AddPoint(300, 0.85)
            opacity.AddPoint(600, 1.0)

            color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
            color.AddRGBPoint(80, 0.2, 0.1, 0.1)
            color.AddRGBPoint(150, 0.9, 0.15, 0.1)
            color.AddRGBPoint(300, 1.0, 0.85, 0.2)
            color.AddRGBPoint(600, 1.0, 0.95, 0.5)
        elif preset == "metal":
            opacity.AddPoint(-1000, 0.0)
            opacity.AddPoint(400, 0.0)
            opacity.AddPoint(1200, 0.0)
            opacity.AddPoint(1800, 0.12)
            opacity.AddPoint(2200, 0.55)
            opacity.AddPoint(2600, 0.92)
            opacity.AddPoint(3500, 1.0)

            color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
            color.AddRGBPoint(1200, 0.0, 0.0, 0.0)
            color.AddRGBPoint(2000, 0.85, 0.55, 0.12)
            color.AddRGBPoint(2600, 1.0, 0.88, 0.35)
            color.AddRGBPoint(3500, 1.0, 0.95, 0.75)
        else:
            opacity.AddPoint(-1000, 0.0)
            opacity.AddPoint(-500, 0.0)
            opacity.AddPoint(-100, 0.04)
            opacity.AddPoint(80, 0.12)
            opacity.AddPoint(200, 0.35)
            opacity.AddPoint(500, 0.75)
            opacity.AddPoint(1200, 1.0)
            opacity.AddPoint(3000, 1.0)

            color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
            color.AddRGBPoint(-100, 0.55, 0.25, 0.15)
            color.AddRGBPoint(80, 0.85, 0.65, 0.50)
            color.AddRGBPoint(300, 0.92, 0.90, 0.85)
            color.AddRGBPoint(1200, 1.0, 1.0, 0.95)
    else:
        span = max(hi - lo, 1.0)
        opacity.AddPoint(lo, 0.0)
        opacity.AddPoint(lo + 0.15 * span, 0.0)
        opacity.AddPoint(lo + 0.35 * span, 0.20)
        opacity.AddPoint(lo + 0.60 * span, 0.55)
        opacity.AddPoint(lo + 0.85 * span, 0.90)
        opacity.AddPoint(hi, 1.0)

        color.AddRGBPoint(lo, 0.0, 0.0, 0.0)
        color.AddRGBPoint(lo + 0.35 * span, 0.70, 0.55, 0.40)
        color.AddRGBPoint(lo + 0.70 * span, 0.95, 0.92, 0.88)
        color.AddRGBPoint(hi, 1.0, 1.0, 1.0)

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.SetInterpolationTypeToLinear()
    prop.ShadeOn()
    prop.SetAmbient(0.25)
    prop.SetDiffuse(0.85)
    prop.SetSpecular(0.15)
    prop.SetSpecularPower(20.0)
    return prop


def _create_volume_mapper(vtk_img: vtk.vtkImageData, preset: str = "bone") -> vtk.vtkAbstractVolumeMapper:
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_img)
    if preset == "mip":
        mapper.SetBlendModeToMaximumIntensity()
    else:
        mapper.SetBlendModeToComposite()
    return mapper


def _extract_bone_polydata(vtk_ct: vtk.vtkImageData, iso_value: float) -> vtk.vtkPolyData | None:
    lo, hi = vtk_ct.GetScalarRange()
    if hi < iso_value:
        iso_value = lo + 0.72 * (hi - lo)

    mc = vtk.vtkFlyingEdges3D()
    mc.SetInputData(vtk_ct)
    mc.SetValue(0, float(iso_value))
    mc.ComputeNormalsOn()
    mc.Update()
    if mc.GetOutput().GetNumberOfPoints() < 200:
        return None

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(mc.GetOutputPort())
    smooth.SetNumberOfIterations(8)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.SetPassBand(0.12)
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    return smooth.GetOutput()


def _bone_surface_actor(
    vtk_ct: vtk.vtkImageData,
    iso_value: float = 400.0,
    *,
    opacity: float = 0.52,
    color: tuple[float, float, float] = (0.90, 0.86, 0.76),
) -> vtk.vtkActor | None:
    """颅骨半透明表面（离屏环境下比体绘制更稳定）。"""
    poly = _extract_bone_polydata(vtk_ct, iso_value)
    if poly is None:
        return None

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    prop.SetAmbient(0.42)
    prop.SetDiffuse(0.78)
    prop.SetSpecular(0.55)
    prop.SetSpecularPower(28.0)
    return actor


def _metal_surface_actor(mask_image: sitk.Image, reference: sitk.Image) -> vtk.vtkActor | None:
    mask = sitk.Cast(mask_image, sitk.sitkUInt8)
    if int(sitk.GetArrayViewFromImage(mask).sum()) < 40:
        return None

    mask_rs = sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        mask.GetPixelID(),
    )
    vtk_mask = sitk_to_vtk_image(mask_rs)
    if vtk_mask.GetScalarRange()[1] <= 0:
        return None

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_mask)
    mc.SetValue(0, 0.5)
    mc.Update()
    if mc.GetOutput().GetNumberOfPoints() < 120:
        return None

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(mc.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(1.0, 0.55, 0.08)
    prop.SetOpacity(1.0)
    prop.SetAmbient(0.55)
    prop.SetDiffuse(0.85)
    prop.SetSpecular(0.95)
    prop.SetSpecularPower(40.0)
    return actor


def _metal_hu_surface_actor(vtk_ct: vtk.vtkImageData, iso_value: float = 2200.0) -> vtk.vtkActor | None:
    """无掩码时按高 HU 阈值提取金属样高密度区域。"""
    lo, hi = vtk_ct.GetScalarRange()
    if hi < 800:
        return None
    iso = float(iso_value)
    if hi < iso:
        iso = max(1500.0, lo + 0.78 * (hi - lo))
    return _bone_surface_actor(
        vtk_ct,
        iso_value=iso,
        opacity=0.92,
        color=(1.0, 0.72, 0.18),
    )


class VtkVolumeScene:
    def __init__(self, width: int = 480, height: int = 360) -> None:
        self.width = max(64, int(width))
        self.height = max(64, int(height))

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.08)
        self.renderer.GradientBackgroundOff()

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetShowWindow(0)
        self.render_window.SetMultiSamples(0)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(self.width, self.height)

        self._ct_iso: sitk.Image | None = None

    def set_volume(
        self,
        ct_image: sitk.Image,
        mask_image: sitk.Image | None,
        window_level: float,  # noqa: ARG002
        window_width: float,  # noqa: ARG002
        *,
        show_mask: bool = False,
        vr_preset: str = "bone",
    ) -> None:
        self.renderer.RemoveAllViewProps()

        self._ct_iso = prepare_ct_for_vr(ct_image)
        vtk_ct = sitk_to_vtk_image(self._ct_iso)
        lo, hi = vtk_ct.GetScalarRange()
        preset = vr_preset if vr_preset in ("bone", "soft", "vessel", "metal", "mip") else "bone"
        show_metal = show_mask or preset == "metal"

        volume = vtk.vtkVolume()
        volume.SetMapper(_create_volume_mapper(vtk_ct, preset))
        volume.SetProperty(make_volume_property(lo, hi, preset))
        self.renderer.AddVolume(volume)

        metal_actor: vtk.vtkActor | None = None
        if show_metal and mask_image is not None:
            metal_actor = _metal_surface_actor(mask_image, self._ct_iso)
        if metal_actor is None and show_metal and preset == "metal":
            metal_actor = _metal_hu_surface_actor(vtk_ct)
        if metal_actor is not None:
            self.renderer.AddActor(metal_actor)

        self.reset_camera()

    def reset_camera(self) -> None:
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.Azimuth(25)
        cam.Elevation(-12)
        cam.Dolly(1.05)
        self.renderer.ResetCameraClippingRange()

    def rotate(self, delta_x: float, delta_y: float) -> None:
        cam = self.renderer.GetActiveCamera()
        cam.Azimuth(float(delta_x) * 0.45)
        cam.Elevation(float(delta_y) * 0.45)
        self.renderer.ResetCameraClippingRange()

    def zoom(self, factor: float) -> None:
        cam = self.renderer.GetActiveCamera()
        cam.Zoom(float(factor))
        self.renderer.ResetCameraClippingRange()

    def resize(self, width: int, height: int) -> None:
        self.width = max(64, int(width))
        self.height = max(64, int(height))
        self.render_window.SetSize(self.width, self.height)

    def render_rgba(self) -> np.ndarray:
        self.render_window.SetSize(self.width, self.height)
        self.render_window.Render()

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.render_window)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()

        out = w2i.GetOutput()
        w, h, _ = out.GetDimensions()
        scalars = out.GetPointData().GetScalars()
        if scalars is None:
            raise RuntimeError("VTK 渲染未产生图像数据")

        rgb = numpy_support.vtk_to_numpy(scalars).reshape(h, w, 3)
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 255)).astype(np.uint8) if rgb.max() > 1 else (rgb * 255).astype(np.uint8)

        if rgb.mean() < 8:
            raise RuntimeError("VTK 体绘制过暗，请确认已加载 CT(HU) 或 NRRD 体数据")

        rgb = np.flipud(rgb)
        rgba = np.concatenate(
            [rgb.astype(np.float32) / 255.0, np.ones((h, w, 1), dtype=np.float32)],
            axis=2,
        )
        return rgba
