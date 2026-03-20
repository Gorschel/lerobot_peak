import time
from threading import Lock
import cv2
from typing import Any, Dict, List
from numpy.typing import NDArray


from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from lerobot.cameras import Camera, ColorMode
from configuration_genicam import GenICamCameraConfig

from genicam_video_capture import GenICamVideoCapture

_ROTATION_CODE = {
    # Cv2Rotation enum Reihenfolge analog zu LeRobot
    0: None,  # NO_ROTATION
    1: cv2.ROTATE_90_CLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class GenICamCamera(Camera):
    """
    LeRobot-kompatibler Treiber für GenICam/IDS peak Kameras (USB3 / GigE).
    Implementiert connect/read/async_read/disconnect und nutzt einen
    Hintergrund-Thread im Low-Level-Wrapper für asynchrones Lesen – konsistent
    mit dem Muster der OpenCV/RealSense Kameras in LeRobot.
    """

    def __init__(self, config: GenICamCameraConfig):
        super().__init__(config)
        self.config = config
        self._cap: GenICamVideoCapture | None = None
        self._lock = Lock()  # Schutz lokaler Felder
        # Diese Eigenschaften sind Teil der Camera-Basis (fps, width, height)
        self.fps = config.fps if config.fps is not None else 0
        self.width = config.width if config.width is not None else 0
        self.height = config.height if config.height is not None else 0

    @check_if_not_connected
    def connect(self, warmup: bool = True) -> None:
        # Öffnen
        self._cap = GenICamVideoCapture(self.config.index_or_serial)

        # Parameter setzen (falls vorhanden)
        if self.config.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        # Werte aus Kamera zurücklesen (sofern unterstützt)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or (self.config.fps or 0)
        self._cap._buf_timeout = 500

        # Hintergrundleser vorbereiten (wird bei async_read() auto gestartet)
        if warmup and self.config.warmup_s > 0:
            start = time.time()
            while time.time() - start < self.config.warmup_s:
                ok, _ = self._cap.read()
                if not ok:
                    break
                time.sleep(0.05)

        self.is_connected = True

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
        self.is_connected = False

    def is_connected(self) -> bool:
        return self._cap.isOpened()

    def _postprocess(self, img_bgr: NDArray, color_mode: ColorMode) -> NDArray:
        # Rotation
        rot_code = _ROTATION_CODE.get(int(self.config.rotation), None)
        if rot_code is not None:
            img_bgr = cv2.rotate(img_bgr, rot_code)

        # Farbmodus
        if color_mode == ColorMode.RGB:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif color_mode == ColorMode.BGR:
            return img_bgr
        elif color_mode == ColorMode.GRAY:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            # Fallback: RGB
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def read(self, color_mode: ColorMode | None = None) -> NDArray:
        if not self.is_connected or self._cap is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._cap._buf_timeout = 2000
        ok, frame_bgr = self._cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame")

        return self._postprocess(frame_bgr, color_mode or self.config.color_mode)

    def async_read(self, timeout_ms: float = 200):
        if not self.is_connected or self._cap is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        out = self._cap.async_read(timeout_ms=timeout_ms, return_dict=self.config.return_dict, as_rgb=True)
        if self.config.return_dict:
            # {"rgb": ndarray, "timestamp": float}
            rgb = out["rgb"]
            rgb = self._postprocess(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.config.color_mode)
            out["rgb"] = rgb
            return out
        else:
            # ndarray RGB
            rgb = out
            return self._postprocess(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.config.color_mode)

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        """
        Listet GenICam/IDS-peak Geräte mit Basisinfos (Name, Serial, TransportLayer).
        Kann in ein "lerobot-find-cameras genicam" Tool eingebunden werden.
        """
        from ids_peak import ids_peak
        dm = ids_peak.DeviceManager.Instance()
        dm.Update()
        out = []
        for dev in dm.Devices():
            entry = {
                "name": dev.DisplayName(),
                "serial": dev.SerialNumberString(),
                "model": dev.ModelName(),
                "tl": dev.TlType(),  # z.B. 'GEV' / 'U3V'
            }
            out.append(entry)
        return out
