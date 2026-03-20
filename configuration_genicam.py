from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation


@dataclass
class GenICamCameraConfig(CameraConfig):
    """
    Konfiguration für GenICam/IDS peak Kameras.
    - index_or_serial: int (Index) oder str (Seriennummer)
    - fps, width, height: Ziel-Stream-Parameter (sofern supported)
    - color_mode: RGB (Standard in LeRobot)
    - rotation: optional (wie in OpenCVCamera)
    - warmup_s: Anzahl Sekunden Warmup
    - return_dict: falls True, liefert async_read/read dict {"rgb":..., "timestamp":...}
    """
    index_or_serial: Union[int, str] = 0
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 0
    return_dict: bool = False

    @classmethod
    def type(cls) -> str:
        # Registrierungsname (analog zu "opencv", "realsense")
        return "genicam"


# In LeRobot werden Configs über einen Registry-Mechanismus gefunden.
CameraConfig.register_subclass(GenICamCameraConfig.type())(GenICamCameraConfig)