"""VTK 后台渲染线程，避免阻塞 Dear PyGui 主界面。"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from image_view import DEFAULT_WINDOW_LEVEL, DEFAULT_WINDOW_WIDTH
from vtk_volume_viewer import VtkVolumeScene, suppress_vtk_output


@dataclass
class _VtkJob:
    kind: str  # full | render | resize | rotate | zoom | reset
    width: int
    height: int
    show_mask: bool = False
    dx: float = 0.0
    dy: float = 0.0
    zoom: float = 1.0


class VtkRenderWorker:
    def __init__(self, get_payload: Callable[[], dict[str, Any] | None]) -> None:
        self._get_payload = get_payload
        self._jobs: queue.Queue[_VtkJob | None] = queue.Queue()
        self._results: queue.Queue[dict[str, Any]] = queue.Queue()
        self._scene: VtkVolumeScene | None = None
        self._busy = False
        self._thread = threading.Thread(target=self._loop, name="vtk-render", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, job: _VtkJob) -> None:
        # 只保留最新任务，丢弃排队中的旧任务
        while True:
            try:
                old = self._jobs.get_nowait()
                if old is None:
                    self._jobs.put(None)
                    break
            except queue.Empty:
                break
        self._jobs.put(job)

    def poll_result(self) -> dict[str, Any] | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def release_scene(self) -> None:
        """丢弃 VTK 场景与排队任务（线程继续运行）。"""
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                break
        self._scene = None
        self._busy = False

    def shutdown(self) -> None:
        self.release_scene()
        self._jobs.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        suppress_vtk_output()
        while True:
            job = self._jobs.get()
            if job is None:
                break
            self._busy = True
            try:
                payload = self._get_payload()
                if payload is None:
                    self._results.put({"ok": False, "error": "无影像数据"})
                    continue

                if self._scene is None:
                    self._scene = VtkVolumeScene(job.width, job.height)
                else:
                    self._scene.resize(job.width, job.height)

                if job.kind == "full":
                    self._scene = VtkVolumeScene(job.width, job.height)
                    self._scene.set_volume(
                        payload["image"],
                        payload.get("mask"),
                        payload.get("wl", DEFAULT_WINDOW_LEVEL),
                        payload.get("ww", DEFAULT_WINDOW_WIDTH),
                        show_mask=job.show_mask and payload.get("mask") is not None,
                        vr_preset=payload.get("vr_preset", "bone"),
                    )
                elif job.kind == "rotate" and self._scene is not None:
                    self._scene.rotate(job.dx, job.dy)
                elif job.kind == "zoom" and self._scene is not None:
                    self._scene.zoom(job.zoom)
                elif job.kind == "reset" and self._scene is not None:
                    self._scene.reset_camera()
                elif job.kind == "resize" and self._scene is None:
                    self._scene = VtkVolumeScene(job.width, job.height)
                    self._scene.set_volume(
                        payload["image"],
                        payload.get("mask"),
                        payload.get("wl", DEFAULT_WINDOW_LEVEL),
                        payload.get("ww", DEFAULT_WINDOW_WIDTH),
                        show_mask=job.show_mask and payload.get("mask") is not None,
                        vr_preset=payload.get("vr_preset", "bone"),
                    )

                if self._scene is None:
                    self._results.put({"ok": False, "error": "VTK 场景未初始化"})
                    continue

                rgba = self._scene.render_rgba()
                self._results.put({"ok": True, "rgba": rgba})
            except Exception as exc:  # noqa: BLE001
                self._results.put({"ok": False, "error": str(exc)})
            finally:
                self._busy = False
