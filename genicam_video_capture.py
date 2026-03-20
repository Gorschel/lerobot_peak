# genicam_video_capture.py
import sys
import cv2
import numpy as np
import time

from ids_peak.ids_peak import DeviceDescriptor
from threading import Thread, Event, Lock

import ids_peak_common
from ids_peak import ids_peak
import ids_peak_icv


_NODE_TYPES = {
    0: "Integer",
    1: "Boolean",
    2: "Command",
    3: "Float",
    4: "String",
    5: "Register",
    6: "Category",
    7: "Enumeration",
    8: "EnumerationEntry",
}


class GenICamVideoCapture:
    """
    OpenCV-ähnlicher Wrapper für IDS peak (GenICam) – USB3 & GigE.
    API: isOpened(), read(), async_read(), set(), get(), release(), start_async(), stop_async()

    - robustes Acquisition-Handling (TLParamsLocked, Start/Stop, Flush/Revoke)
    - generische set_node/get_node (wie PeakCamera)
    - optionaler Device-Timestamp über GenICam ChunkData
    """

    def __init__(self, index_or_serial=0, buf_timeout_ms=5000, buffers_min=10, enable_device_timestamp=True):
        ids_peak.Library.Initialize()

        self.device = None
        self.datastream = None
        self.nodemap = None
        self.local_nodemap = None

        # gecachte Nodes
        self._n_tl_params_locked = None
        self._n_acq_start = None
        self._n_acq_stop = None
        self._n_trigger_sw = None

        self._buf_timeout = int(buf_timeout_ms)
        self._acq_running = False

        # Async state
        self._thread = None
        self._stop_event = Event()
        self._new_frame_event = Event()
        self._frame_lock = Lock()
        self._latest_frame_bgr = None
        self._latest_ts_host = 0.0
        self._latest_ts_dev = None

        self._buffers_min = int(buffers_min)
        self._enable_device_timestamp = bool(enable_device_timestamp)

        self.open(index_or_serial)

        self.pipeline = ids_peak_icv.pipeline.DefaultPipeline()
        self.pipeline.output_pixel_format = ids_peak_common.PixelFormat.BGR_8
        self.pipeline.color_correction.matrix = self.get_ccm()
        self.pipeline.gamma.value = 2.2

    def __del__(self):
        try:
            self.release()
        except Exception as e:
            print(e)
            pass

    def get_ccm(self):
        ccm = np.identity(3, dtype=np.float32)
        if self.has_node('ColorCorrectionMatrix') and self.has_node('ColorCorrectionMatrixValueSelector'):
            for x in range(3):
                for y in range(3):
                    self.set_node('ColorCorrectionMatrixValueSelector', f"Gain{x}{y}")
                    ccm[x, y] = self.get_node('ColorCorrectionMatrixValue')
        return ccm

    # ---------------------- Node convenience ----------------------
    def has_node(self, name):
        try:
            self.nodemap.FindNode(name)
            return True
        except ids_peak.NotFoundException:
            return False
        except Exception as e:
            print(e)
            return False

    def set_node(self, node: str, value=None) -> bool:
        if self.nodemap is None:
            print("Nodemap missing.")
            return False
        try:
            n = self.nodemap.FindNode(node)
            t = _NODE_TYPES.get(n.Type(), "Unknown")

            if t == "Integer" and isinstance(value, (int, float)):
                n.SetValue(int(value))
            elif t == "Boolean" and isinstance(value, (bool, int)):
                n.SetValue(bool(value))
            elif t == "Float" and isinstance(value, (float, int)):
                n.SetValue(float(value))
            elif t == "String" and isinstance(value, str):
                n.SetValue(value)
            elif t == "Enumeration" and isinstance(value, str):
                n.SetCurrentEntry(value)
            elif t == "Command" and value is None:
                n.Execute()
                n.WaitUntilDone()
                if value is not None:
                    raise ValueError(f"Command node(s) {node} do not accept values.")
            elif t == "Register":
                if value is None:
                    raise ValueError("Register write requires a value.")
                n.Write(value)
            else:
                print(f"Unsupported node type or invalid value: node='{node}' type='{t}' value='{value}'")
                return False
            return True
        except Exception as e:
            print(f"Could not set '{node}' to {value}: {e}")
            return False

    def get_node(self, node: str):
        if self.nodemap is None:
            print("Nodemap missing.")
            return None
        try:
            n = self.nodemap.FindNode(node)
            t = _NODE_TYPES.get(n.Type(), "Unknown")

            if t in ["Integer", "Boolean", "Float", "String"]:
                return n.Value()
            elif t == "Enumeration":
                return n.CurrentEntry().StringValue()
            elif t == "EnumerationEntry":
                return n.StringValue()
            elif t in ["Command", "Register", "Category"]:
                print(f"Convenience getter for node type {t} not implemented")
                return None
            else:
                return None
        except Exception as e:
            print(f"Could not get '{node}' value: {e}")
            return None

    # ---------------------- Acquisition ----------------------
    def _enable_chunk_timestamp(self):
        if not self._enable_device_timestamp or self.nodemap is None:
            return
        try:
            if self.has_node("ChunkModeActive"):
                self.set_node("ChunkModeActive", True)
            if self.has_node("ChunkSelector"):
                sel = self.nodemap.FindNode("ChunkSelector")
                # "Timestamp" Eintrag auswählen, falls vorhanden
                entries = [e.SymbolicValue() for e in sel.Entries()]
                if "Timestamp" in entries:
                    sel.SetCurrentEntry("Timestamp")
                    if self.has_node("ChunkEnable"):
                        self.set_node("ChunkEnable", True)
        except Exception as e:
            print(f"Enabling chunk timestamp failed (ignored): {e}")

    def _acq_start(self):
        if self._acq_running:
            print("Acquisition already running.")
            return True
        try:
            # TLParams entsperren
            if self.has_node("TLParamsLocked"):
                self._n_tl_params_locked = self.nodemap.FindNode("TLParamsLocked")
                self._n_tl_params_locked.SetValue(0)

            # Bufferpool aufsetzen
            payload_size = self.get_node("PayloadSize")
            buf_req = max(self.datastream.NumBuffersAnnouncedMinRequired(), self._buffers_min)
            for _ in range(buf_req):
                self.datastream.QueueBuffer(self.datastream.AllocAndAnnounceBuffer(payload_size))

            # wieder sperren (während der Aufnahme)
            if self._n_tl_params_locked is not None:
                self._n_tl_params_locked.SetValue(1)

            # Datastream + AcquisitionStart
            self.datastream.StartAcquisition()
            if self._n_acq_start is None and self.has_node("AcquisitionStart"):
                self._n_acq_start = self.nodemap.FindNode("AcquisitionStart")
            if self._n_acq_start is not None:
                self._n_acq_start.Execute()
                self._n_acq_start.WaitUntilDone()

            # Optional SW-Trigger Node (nur wenn in SW-Trigger-Modus verwendet)
            if self.has_node("TriggerSoftware"):
                self._n_trigger_sw = self.nodemap.FindNode("TriggerSoftware")

            self._acq_running = True
            return True
        except Exception as e:
            print(f"Acquisition start failed: {e}")
            return False

    def _acq_stop(self):
        if not self._acq_running:
            print("Acquisition already stopped.")
            return True
        try:
            if self._n_acq_stop is None and self.has_node("AcquisitionStop"):
                self._n_acq_stop = self.nodemap.FindNode("AcquisitionStop")
            if self._n_acq_stop is not None:
                self._n_acq_stop.Execute()
                self._n_acq_stop.WaitUntilDone()

            # Datastream stoppen + Flush/Unannounce
            self.datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            self.datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for b in self.datastream.AnnouncedBuffers():
                self.datastream.RevokeBuffer(b)

            self._acq_running = False

            if self._n_tl_params_locked is not None:
                self._n_tl_params_locked.SetValue(0)

            self._n_trigger_sw = None
            return True
        except Exception as e:
            print(f"Acquisition stop failed: {e}")
            return False

    @staticmethod
    def get_device_list():
        dm = ids_peak.DeviceManager.Instance()
        dm.Update()
        devices = dm.Devices()
        if devices.empty():
            raise RuntimeError("Keine GenICam/IDS-Kamera gefunden.")
        dev_list = []
        for d in devices:
            dev_list.append({
                "is_openable": d.IsOpenable(),
                "serial": d.SerialNumber(),
                "model": d.ModelName(),
            })
        return dev_list

    def open(self, index_or_serial=0):
        dm = ids_peak.DeviceManager.Instance()
        dm.Update()
        devices = dm.Devices()
        if devices.empty():
            raise RuntimeError("Keine GenICam/IDS-Kamera gefunden.")

        # Auswahl: per Index oder per Seriennummer (String-Substring erlaubt, analog __init__.py)
        chosen = None
        if isinstance(index_or_serial, int):
            if index_or_serial < 0 or index_or_serial >= devices.size():
                raise IndexError("Kamera-Index außerhalb des gültigen Bereichs.")
            chosen: DeviceDescriptor = devices[index_or_serial]
        else:
            for d in devices:
                try:
                    if str(index_or_serial) in d.SerialNumber():
                        chosen = d
                        break
                except Exception as e:
                    print(e)
        if chosen is None:
            raise RuntimeError(f"Keine Kamera mit Serial '{index_or_serial}' gefunden.")

        # Gerät öffnen
        if not chosen.IsOpenable():
            print(f"Device '{chosen.SerialNumber()}' is in use")
            sys.exit(1)
        self.device = chosen.OpenDevice(ids_peak.DeviceAccessType_Control)
        self.nodemap = self.device.RemoteDevice().NodeMaps()[0]
        self.local_nodemap = self.device.RemoteDevice().LocalDevice().NodeMaps()[0]

        # Datenstrom öffnen
        streams = self.device.DataStreams()
        if streams.empty():
            raise RuntimeError("Device hat keinen DataStream.")
        self.datastream = streams[0].OpenDataStream()

        # gecachte Command-Knoten
        if self.has_node("AcquisitionStart"):
            self._n_acq_start = self.nodemap.FindNode("AcquisitionStart")
        if self.has_node("AcquisitionStop"):
            self._n_acq_stop = self.nodemap.FindNode("AcquisitionStop")
        if self.has_node("TLParamsLocked"):
            self._n_tl_params_locked = self.nodemap.FindNode("TLParamsLocked")

        # optional: Chunk-Timestamp aktivieren
        self._enable_chunk_timestamp()

        # Aufnahme starten
        started = self._acq_start()
        if not started:
            raise RuntimeError("Acquisition konnte nicht gestartet werden.")

    def release(self):
        if not self.isOpened():
            return
        try:
            self.stop_async()
        except Exception as e:
            print(e)
            pass
        try:
            self._acq_stop()
        except Exception as e:
            print(e)
        finally:
            self.device = None
            self.datastream = None
            self.nodemap = None
            self.local_nodemap = None
            try:
                ids_peak.Library.Close()
            except Exception as e:
                print(e)
                pass

    # ----------- sync read -----------
    def isOpened(self):
        return self.device is not None and self.datastream is not None

    def read(self, as_gray=False):
        if not self.isOpened() or not self._acq_running:
            return False, None

        ok, buffer, frame = False, None, None
        try:
            buffer = self.datastream.WaitForFinishedBuffer(self._buf_timeout)
            frame = self.pipeline.process(buffer.ToImageView()).to_numpy_array(True)
            if as_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            ok = True
        except Exception as e:
            print(e)
        finally:
            if buffer is not None:
                self.datastream.QueueBuffer(buffer)
            return ok, frame

    # ----------- async API -----------
    def start_async(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._grab_loop, name="GenICamGrabLoop", daemon=True)
        self._thread.start()

    def stop_async(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        self._new_frame_event.clear()

    def async_read(self, timeout_ms=200, return_dict=False):
        """
        Liefert das jüngste Frame aus dem Hintergrund-Thread.
        - timeout_ms: Wartezeit auf neues Frame
        - return_dict: {"img":..., "timestamp_device":..., "timestamp_host":...}
        - as_rgb: BGR->RGB
        """
        if self._thread is None or not self._thread.is_alive():
            self.start_async()

        got = self._new_frame_event.wait(timeout=timeout_ms / 1000.0)
        if not got:
            raise TimeoutError(f"async_read timeout after {timeout_ms} ms")

        with self._frame_lock:
            frame = None if self._latest_frame_bgr is None else self._latest_frame_bgr.copy()
            ts_host = self._latest_ts_host
            ts_dev = self._latest_ts_dev
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError("Internal: event set but no frame available.")

        if return_dict:
            return {
                "img": frame,
                "timestamp_device": ts_dev,
                "timestamp_host": ts_host,
            }
        return frame

    # ----------------- Worker -----------------
    def _grab_loop(self):
        time.sleep(0.01)
        while not self._stop_event.is_set():
            # nur wenn Akquisition läuft
            if not self._acq_running:
                time.sleep(0.01)

            buffer = None
            try:
                buffer = self.datastream.WaitForFinishedBuffer(self._buf_timeout)
                frame = self.pipeline.process(buffer.ToImageView())
                ts_host = time.time()
                ts_dev = frame.metadata.get_value_by_key(ids_peak_common.MetadataKey.DEVICE_TIMESTAMP)
                frame = frame.to_numpy_array(True)
                with self._frame_lock:
                    self._latest_frame_bgr = frame
                    self._latest_ts_host = ts_host
                    self._latest_ts_dev = ts_dev
                    self._new_frame_event.set()
            except Exception as e:
                print(e)
            finally:
                if buffer is not None:
                    self.datastream.QueueBuffer(buffer)

    # ----------- OpenCV-style property bridge -----------

    def _ensure_legal_node_value(self, name: str, value, autoclip=True):
        n = self.nodemap.FindNode(name)
        _min, _max = n.Minimum(), n.Maximum()
        if n.IncrementType() == ids_peak.NodeIncrementType_FixedIncrement:
            _inc = n.Increment()
        elif n.IncrementType() == ids_peak.NodeIncrementType_ListIncrement:
            raise NotImplementedError
        else:
            _inc = 0
        if not _min < value < _max and not autoclip:
            raise ValueError(f"Node '{name}' value {value} is out of range [{_min}, {_max}].")
        if _inc > 0:
            value = _min + _inc * np.floor((value - _min) / _inc + 0.5)
        if autoclip:
            value = np.clip(value, _min, _max)
        return value

    def set(self, prop, value):
        if not self.isOpened():
            return False
        try:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self._acq_stop()
                value = self._ensure_legal_node_value('Width', value)
                ok = self.set_node('Width', value)
                self._acq_start()
                return ok

            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self._acq_stop()
                value = self._ensure_legal_node_value('Height', value)
                ok = self.set_node('Height', value)
                self._acq_start()
                return ok

            if prop == cv2.CAP_PROP_FPS:
                value = self._ensure_legal_node_value('AcquisitionFrameRate', value)
                ok = self.set_node('AcquisitionFrameRate', value)
                return ok

            if prop == cv2.CAP_PROP_GAIN:
                if self.get_node("GainAuto") != 'Off':
                    self.set_node('GainAuto', 'Off')
                gain_modes = [e.StringValue() for e in self.nodemap.FindNode('GainSelector').AvailableEntries()]
                if 'AnalogAll' in gain_modes:
                    self.set_node('GainSelector', 'AnalogAll')
                elif 'All' in gain_modes:
                    self.set_node('GainSelector', 'All')
                n = self.nodemap.FindNode('Gain')
                n.SetValue(self._ensure_legal_node_value('Gain', value))
                maxgain = n.Maximum()
                if value > maxgain:
                    print(f"could not set gain to {value}x. {maxgain}x is the maximum")
                value = self._ensure_legal_node_value('Gain', value)
                ok = self.set_node('Gain', value)
                return ok

            if prop == cv2.CAP_PROP_EXPOSURE:
                if self.has_node("ExposureAuto"):
                    self.set_node("ExposureAuto", "Off")
                if self.has_node("ExposureTime"):
                    self.set_node("ExposureTime", float(value))
                    return True
                return False

            if prop == cv2.CAP_PROP_AUTO_EXPOSURE and self.has_node("ExposureAuto"):
                return self.set_node("ExposureAuto", "Continuous" if value else "Off")

            if prop == cv2.CAP_PROP_AUTO_WB and self.has_node("BalanceWhiteAuto"):
                return self.set_node("BalanceWhiteAuto", "Continuous" if value else "Off")

            if prop == cv2.CAP_PROP_AUTOFOCUS and self.has_node("FocusAuto"):
                return self.set_node('FocusAuto', 'Continuous' if value else "Off")

            else:
                raise NotImplementedError(prop)
        except Exception as e:
            print(e)
            return None

    def get(self, prop):
        if not self.isOpened():
            return 0
        try:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return self.get_node("Width")
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return self.get_node("Height")
            if prop == cv2.CAP_PROP_FPS and self.has_node("AcquisitionFrameRate"):
                return self.get_node("AcquisitionFrameRate")
            if prop == cv2.CAP_PROP_EXPOSURE and self.has_node("ExposureTime"):
                return self.get_node("ExposureTime")
            if prop == cv2.CAP_PROP_GAIN and self.has_node("Gain"):
                gain_modes = [e.StringValue() for e in self.nodemap.FindNode('GainSelector').AvailableEntries()]
                if 'AnalogAll' in gain_modes:
                    self.set_node('GainSelector', 'AnalogAll')
                elif 'All' in gain_modes:
                    self.set_node('GainSelector', 'All')
                return self.get_node("Gain")
            else:
                raise NotImplementedError(prop)
        except Exception as e:
            print(e)
            return None


if __name__ == '__main__':
    cap = GenICamVideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1501)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1501)
    # cap.set(cv2.CAP_PROP_GAIN, 5)
    cap.set(cv2.CAP_PROP_AUTO_WB, True)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, True)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, True)

    # cap.start_async()
    try:
        cv2.namedWindow("display")
        id = 0
        while True:
            # frame = cap.async_read(timeout_ms=200, return_dict=False)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.setWindowTitle("display", f"frame {id:03d}")
                id += 1
                cv2.imshow("display", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.stop_async()
        cap.release()
        cv2.destroyAllWindows()
