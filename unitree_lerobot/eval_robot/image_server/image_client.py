import cv2
import zmq
import numpy as np
import time
import struct
import threading
from collections import deque
from multiprocessing import shared_memory


class ImageClient:
    def __init__(
        self,
        tv_img_shape=None,
        tv_img_shm_name=None,
        wrist_img_shape=None,
        wrist_img_shm_name=None,
        image_show=False,
        server_address="192.168.123.164",
        port=5555,
        Unit_Test=False,
    ):
        """
        tv_img_shape: User's expected head camera resolution shape (H, W, C). It should match the output of the image service terminal.
        tv_img_shm_name: Shared memory is used to easily transfer images across processes to the Vuer.
        wrist_img_shape: User's expected wrist camera resolution shape (H, W, C). It should maintain the same shape as tv_img_shape.
        wrist_img_shm_name: Shared memory is used to easily transfer images.
        image_show: Whether to display received images in real time.
        server_address: The ip address to execute the image server script.
        port: The port number to bind to. It should be the same as the image server.
        Unit_Test: When both server and client are True, it can be used to test the image transfer latency, \
                   network jitter, frame loss rate and other information.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=self.tv_image_shm.buf)
            self.tv_enable_shm = True

        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        # Performance evaluation parameters
        self._enable_performance_eval = Unit_Test
        self._init_performance_metrics()
        self._header_decode_warning_printed = False
        self._header_fallback_warning_printed = False

    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID

        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window

        # Data transmission quality metrics
        self._latency_samples = deque()  # (receive_time, latency_s) in current window
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs
        self._last_receive_time = None
        self._last_source_timestamp = None
        self._stats_lock = threading.Lock()

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # Update latency (only when sender provides timestamps)
        if timestamp is not None:
            latency = receive_time - timestamp
            self._latency_samples.append((receive_time, latency))
            self._last_source_timestamp = timestamp
        else:
            latency = None

        # Remove latencies outside the time window
        while self._latency_samples and self._latency_samples[0][0] < receive_time - self._time_window:
            self._latency_samples.popleft()

        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        # Update frame counts for lost frame calculation
        if frame_id is not None:
            expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
            if frame_id != expected_frame_id:
                lost = frame_id - expected_frame_id
                if lost < 0:
                    print(f"[Image Client] Received out-of-order frame ID: {frame_id}")
                else:
                    self._lost_frames += lost
                    print(
                        f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}"
                    )
            self._last_frame_id = frame_id
            self._total_frames = frame_id + 1

        self._frame_count += 1
        self._last_receive_time = receive_time

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # Calculate real-time FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # Calculate latency metrics
            latencies = [lat for _, lat in self._latency_samples]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            if latencies:
                print(
                    f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency * 1000:.2f} ms, Max Latency: {max_latency * 1000:.2f} ms, \
                      Min Latency: {min_latency * 1000:.2f} ms, Jitter: {jitter * 1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%"
                )
            else:
                print(
                    f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Latency: unavailable (no sender timestamp header), "
                    f"Lost Frame Rate: {lost_frame_rate:.2f}%"
                )

    def get_runtime_stats(self):
        now = time.time()
        with self._stats_lock:
            last_receive_time = self._last_receive_time
            last_source_timestamp = self._last_source_timestamp
            frame_count = self._frame_count
            recent_frame_count = len(self._frame_times)
            latencies = [lat for _, lat in self._latency_samples]
            lost_frames = self._lost_frames
            total_frames = self._total_frames

        age_s = (now - last_receive_time) if last_receive_time is not None else None
        avg_latency_s = (sum(latencies) / len(latencies)) if latencies else None
        lost_frame_rate = (lost_frames / total_frames) if total_frames > 0 else 0.0
        return {
            "frame_count": frame_count,
            "recv_fps_1s": float(recent_frame_count) / float(self._time_window),
            "last_frame_age_s": age_s,
            "avg_latency_s": avg_latency_s,
            "last_source_timestamp": last_source_timestamp,
            "lost_frame_rate": lost_frame_rate,
        }

    def _close(self):
        self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        print("Image client has been closed.")

    def receive_process(self):
        # Set up ZeroMQ context and socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print("\nImage client has started, waiting to receive data...")
        try:
            while self.running:
                # Receive message
                message = self._socket.recv()
                receive_time = time.time()
                timestamp = None
                frame_id = None
                jpg_bytes = message
                if self._enable_performance_eval:
                    header_size = struct.calcsize("dI")
                    if len(message) > header_size:
                        raw_jpeg_soi = len(message) >= 2 and message[0] == 0xFF and message[1] == 0xD8
                        if not raw_jpeg_soi:
                            try:
                                header = message[:header_size]
                                candidate_timestamp, candidate_frame_id = struct.unpack("dI", header)
                                candidate_jpg = message[header_size:]
                                candidate_jpg_soi = (
                                    len(candidate_jpg) >= 2 and candidate_jpg[0] == 0xFF and candidate_jpg[1] == 0xD8
                                )
                                timestamp_plausible = 946684800.0 <= float(candidate_timestamp) <= receive_time + 60.0
                                frame_id_plausible = int(candidate_frame_id) >= 0
                                if candidate_jpg_soi and timestamp_plausible and frame_id_plausible:
                                    timestamp = candidate_timestamp
                                    frame_id = candidate_frame_id
                                    jpg_bytes = candidate_jpg
                                elif not self._header_fallback_warning_printed:
                                    print(
                                        "[Image Client] Sender appears to publish raw JPEG without timestamp header; "
                                        "using raw mode (latency unavailable)."
                                    )
                                    self._header_fallback_warning_printed = True
                            except struct.error as e:
                                if not self._header_decode_warning_printed:
                                    print(
                                        "[Image Client] Failed to decode sender header once; "
                                        "falling back to raw JPEG mode (latency unavailable). "
                                        f"Error: {e}"
                                    )
                                    self._header_decode_warning_printed = True
                # Decode image
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if current_image is None:
                    print("[Image Client] Failed to decode image.")
                    continue

                if self.tv_enable_shm:
                    tv_frame = np.array(current_image[:, : self.tv_img_shape[1]])
                    # Teleimager in Isaac Sim may publish per-camera 640x480 streams; resize to expected
                    # shared-memory shape to avoid broadcaster crashes in cross-machine setups.
                    if tv_frame.shape != self.tv_img_array.shape:
                        tv_frame = cv2.resize(tv_frame, (self.tv_img_shape[1], self.tv_img_shape[0]))
                    np.copyto(self.tv_img_array, tv_frame)

                if self.wrist_enable_shm:
                    wrist_frame = np.array(current_image[:, -self.wrist_img_shape[1] :])
                    if wrist_frame.shape != self.wrist_img_array.shape:
                        wrist_frame = cv2.resize(wrist_frame, (self.wrist_img_shape[1], self.wrist_img_shape[0]))
                    np.copyto(self.wrist_img_array, wrist_frame)

                if self._image_show:
                    height, width = current_image.shape[:2]
                    resized_image = cv2.resize(current_image, (width // 2, height // 2))
                    cv2.imshow("Image Client Stream", resized_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

                with self._stats_lock:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                if self._enable_performance_eval:
                    self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()


if __name__ == "__main__":
    # example1
    # tv_img_shape = (480, 1280, 3)
    # img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    # img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    # img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    # img_client.receive_process()

    # example2
    # Initialize the client with performance evaluation enabled
    # client = ImageClient(image_show = True, server_address='127.0.0.1', Unit_Test=True) # local test
    client = ImageClient(image_show=True, server_address="192.168.123.164", Unit_Test=False)  # deployment test
    client.receive_process()
