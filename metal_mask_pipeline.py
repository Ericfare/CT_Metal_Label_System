"""CT 金属伪影掩码生成流水线（阈值规则见《标注方法.md》）。"""

from __future__ import annotations

from dataclasses import dataclass

import SimpleITK as sitk


@dataclass
class MaskParams:
    """默认值对应《标注方法.md》金属伪影 HU 规则与实操准则。"""

    lower_hu: float = -200.0         # 金属伪影：HU < 此值（文档：小于 -200）
    upper_hu: float = 1000.0         # 金属伪影：HU > 此值（文档：大于 1000）
    gradient_threshold: float = 80.0  # 梯度约束，配合视觉畸变条纹筛选
    opening_radius: int = 1
    closing_radius: int = 2
    min_component_size: int = 30     # 剔除过小噪点，保留条纹连通域
    use_gradient: bool = True        # 实操准则：需结合形态纹路异常
    keep_largest_only: bool = False

    # 患者 ROI：排除床板、空气与体外背景（先于伪影阈值）
    use_patient_roi: bool = True
    roi_air_hu: float = -400.0       # HU 高于此值视为体内/骨组织候选
    roi_closing_radius: int = 3      # 闭运算连通颅腔与骨壳
    roi_opening_radius: int = 4      # 开运算削弱细连接（如床板窄桥）
    roi_dilate_radius: int = 2       # 略膨胀以包含颅骨缘伪影


# 供界面初始化引用的文档默认参数
DOC_DEFAULT_MASK_PARAMS = MaskParams()


def generate_patient_roi_mask(image: sitk.Image, params: MaskParams | None = None) -> sitk.Image:
    """
    患者 / 颅骨轮廓 ROI：剔除空气、扫描床与 FOV 外背景。
    策略：HU 阈值 -> 闭运算填洞 -> 最大连通域 -> 开运算去细桥 -> 轻微膨胀。
    """
    p = params or DOC_DEFAULT_MASK_PARAMS
    img = sitk.Cast(image, sitk.sitkFloat32)
    dim = image.GetDimension()

    fg = sitk.BinaryThreshold(
        img,
        lowerThreshold=float(p.roi_air_hu),
        upperThreshold=1.0e10,
        insideValue=1,
        outsideValue=0,
    )

    if p.roi_closing_radius > 0:
        fg = sitk.BinaryMorphologicalClosing(
            fg,
            kernelRadius=[p.roi_closing_radius] * dim,
            foregroundValue=1,
        )

    labeled = sitk.ConnectedComponent(fg)
    relabeled = sitk.RelabelComponent(labeled, sortByObjectSize=True)
    roi = sitk.BinaryThreshold(relabeled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)

    if p.roi_opening_radius > 0:
        roi = sitk.BinaryMorphologicalOpening(
            roi,
            kernelRadius=[p.roi_opening_radius] * dim,
            foregroundValue=1,
        )

    if p.roi_dilate_radius > 0:
        roi = sitk.BinaryDilate(
            roi,
            kernelRadius=[p.roi_dilate_radius] * dim,
            foregroundValue=1,
        )

    out = sitk.Cast(roi, sitk.sitkUInt8)
    out.CopyInformation(image)
    return out


def generate_metal_mask(
    image: sitk.Image,
    params: MaskParams,
    patient_roi: sitk.Image | None = None,
) -> sitk.Image:
    """患者 ROI 内：双阈值金属伪影 -> 梯度约束 -> 形态学 -> 连通域过滤。"""
    img = sitk.Cast(image, sitk.sitkFloat32)

    roi: sitk.Image | None = None
    if params.use_patient_roi:
        roi = patient_roi if patient_roi is not None else generate_patient_roi_mask(image, params)

    # 1. 金属伪影双阈值（标注方法.md：HU < -200 或 HU > 1000）
    mask_low = sitk.BinaryThreshold(
        img,
        lowerThreshold=-1.0e10,
        upperThreshold=float(params.lower_hu),
        insideValue=1,
        outsideValue=0,
    )
    mask_high = sitk.BinaryThreshold(
        img,
        lowerThreshold=float(params.upper_hu),
        upperThreshold=1.0e10,
        insideValue=1,
        outsideValue=0,
    )
    mask = sitk.Or(mask_low, mask_high)

    if roi is not None:
        mask = sitk.And(mask, roi)

    # 2. 梯度幅值约束（可选）
    if params.use_gradient:
        gradient = sitk.GradientMagnitudeRecursiveGaussian(img, sigma=1.0)
        grad_mask = sitk.BinaryThreshold(
            gradient,
            lowerThreshold=float(params.gradient_threshold),
            upperThreshold=1e10,
            insideValue=1,
            outsideValue=0,
        )
        mask = sitk.And(mask, grad_mask)

    # 3. 形态学开运算
    if params.opening_radius > 0:
        mask = sitk.BinaryMorphologicalOpening(
            mask,
            kernelRadius=[params.opening_radius] * image.GetDimension(),
            foregroundValue=1,
        )

    # 4. 形态学闭运算
    if params.closing_radius > 0:
        mask = sitk.BinaryMorphologicalClosing(
            mask,
            kernelRadius=[params.closing_radius] * image.GetDimension(),
            foregroundValue=1,
        )

    # 5. 连通域过滤
    labeled = sitk.ConnectedComponent(mask)
    if params.keep_largest_only:
        relabeled = sitk.RelabelComponent(labeled, sortByObjectSize=True)
        mask = sitk.BinaryThreshold(relabeled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
    else:
        relabeled = sitk.RelabelComponent(
            labeled,
            sortByObjectSize=True,
            minimumObjectSize=int(params.min_component_size),
        )
        mask = sitk.Cast(sitk.NotEqual(relabeled, 0), sitk.sitkUInt8)

    if roi is not None:
        mask = sitk.And(mask, roi)

    out = sitk.Cast(mask, sitk.sitkUInt8)
    out.CopyInformation(image)
    return out
