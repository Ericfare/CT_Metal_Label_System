"""DICOM 序列读取。"""

from __future__ import annotations

import os

import SimpleITK as sitk


def read_dicom_folder(folder: str) -> sitk.Image:
    """读取文件夹中切片数最多的 DICOM 序列。"""
    folder = os.path.normpath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder)
    if not series_ids:
        raise ValueError(f"未在文件夹中找到 DICOM 序列: {folder}")

    def _file_count(series_id: str) -> int:
        return len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, series_id))

    series_id = max(series_ids, key=_file_count)
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, series_id)
    if not file_names:
        raise ValueError(f"序列 {series_id} 没有可用 DICOM 文件")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOff()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    return image
