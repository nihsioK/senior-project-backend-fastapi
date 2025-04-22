import collections
import enum
import time
import tensorflow as tf

import argparse
import asyncio
import datetime
import json
import cv2
import numpy as np
import logging
from aiohttp import ClientSession, TCPConnector
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
from av import VideoFrame
from fractions import Fraction
import websockets

# from gesture_recognition import predict_from_frame

logging.basicConfig(level=logging.INFO, format="[Jetson] %(asctime)s - %(message)s")

ice_servers = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="turn:senior-backend.xyz:3478", username="testuser", credential="supersecretpassword"),
    RTCIceServer(urls="turns:senior-backend.xyz:5349", username="testuser", credential="supersecretpassword"),
]


latest_frame = None


class VideoCaptureTrack(VideoStreamTrack):
    def __init__(self, device=0, fps=60):
        super().__init__()
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with index {device}")

        self.fps = fps
        self.time_base = Fraction(1, self.fps)
        self.next_pts = 0
        self.running = False

    async def recv(self):
        global latest_frame
        while not self.running:
            await asyncio.sleep(0.1)

        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        if not ret:
            logging.error("Failed to read frame from camera")
            raise RuntimeError("Camera read failure")

        latest_frame = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = self.next_pts
        video_frame.time_base = self.time_base
        self.next_pts += 1

        return video_frame

    async def start_stream(self):
        if not self.running:
            self.running = True
            logging.info("Streaming started.")

    async def stop_stream(self):
        if self.running:
            self.running = False
            logging.info("Streaming stopped.")


async def recognition_loop(device_id, ws_url, interval=0.5):
    global latest_frame
    retry_attempts = 0
    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                logging.info(f"Connected to WebSocket: {ws_url}")
                retry_attempts = 0  # reset after successful connect

                interpreter = tf.lite.Interpreter(model_path="model.tflite")
                interpreter.allocate_tensors()

                interpreter_act = tf.lite.Interpreter(model_path="bin/act_movenet.tflite")
                interpreter_act.allocate_tensors()

                def movenet(input_image):
                    """Runs detection on an input image.

                    Args:
                      input_image: A [1, height, width, 3] tensor represents the input image
                        pixels. Note that the height/width should already be resized and match the
                        expected input resolution of the model before passing into this function.

                    Returns:
                      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
                      coordinates and scores.
                    """
                    # TF Lite format expects tensor type of uint8.
                    input_image = tf.cast(input_image, dtype=tf.uint8)
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
                    # Invoke inference.
                    interpreter.invoke()
                    # Get the model prediction.
                    keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
                    return keypoints_with_scores

                def act(keypoints):
                    input_details = interpreter_act.get_input_details()
                    output_details = interpreter_act.get_output_details()
                    interpreter_act.set_tensor(input_details[0]["index"], keypoints.astype(np.float32)[None, ...])
                    interpreter_act.invoke()
                    return interpreter_act.get_tensor(output_details[0]["index"])

                def predict_action(pose_buffer):
                    pose_input = preprocess(pose_buffer)
                    pose_dist = act(pose_input)[0]
                    pose_dist = np.exp(pose_dist) / np.sum(np.exp(pose_dist))

                    top_indices = np.argsort(pose_dist)[-2:][::-1]
                    top_classes = [CLASSES[idx] for idx in top_indices]
                    top_probs = [pose_dist[idx] for idx in top_indices]

                    confidence = top_probs[0]
                    second = top_probs[1]

                    if confidence - second < 0.5:
                        action = "I'm not sure"
                    else:
                        action = top_classes[0]

                    return action, confidence

                n = 0
                sum_process_time = 0
                sum_inference_time = 0
                n_frames = 30
                # fps_counter = avg_fps_counter(n_frames)
                inference_time = 0

                crop_region = init_crop_region(INPUT_SIZE, INPUT_SIZE)

                text_template = "{label}, {conf:.2f}%"

                pose_buffer = None  # stores rolling pose history
                prev_len = 0

                while True:

                    frame = latest_frame
                    action = "none detected"
                    confidence = 0.0

                    if frame is None or frame.size == 0:
                        logging.warning("Empty frame received. Skipping...")
                        await asyncio.sleep(0.05)
                        continue

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_height, image_width, _ = frame.shape

                    input_image = tf.expand_dims(frame, axis=0)
                    input_image = crop_and_resize(input_image, crop_region, crop_size=SRC_SIZE)

                    outputs = movenet(input_image)

                    # Map coordinates back to original frame
                    for idx in range(17):
                        outputs[0, 0, idx, 0] = (
                            crop_region["y_min"] * image_height
                            + crop_region["height"] * image_height * outputs[0, 0, idx, 0]
                        ) / image_height
                        outputs[0, 0, idx, 1] = (
                            crop_region["x_min"] * image_width
                            + crop_region["width"] * image_width * outputs[0, 0, idx, 1]
                        ) / image_width

                    start_time = time.monotonic()

                    poses = []
                    dets = []
                    for pose in outputs:
                        score = pose[:, -1].mean()
                        if score > 0.3:
                            poses.append(pose)
                            dets.append(
                                [
                                    np.min(pose[:, 0]),
                                    np.min(pose[:, 1]),
                                    np.max(pose[:, 0]),
                                    np.max(pose[:, 1]),
                                    score,
                                ]
                            )

                    # logging.info(f"Pose detection: {len(poses)} poses detected")
                    # logging.info(f"Pose detection: {poses}")
                    if poses:
                        best_pose = poses[0]
                        pose_sequence = np.squeeze(best_pose[None, ...])  # shape: (17, 3)
                        pose_sequence = pose_sequence[None, ...]  # shape: (1, 17, 3)

                        if pose_buffer is None:
                            pose_buffer = pose_sequence
                        else:
                            pose_buffer = np.concatenate((pose_buffer, pose_sequence), axis=0)

                        # Always trim the buffer to max size n_frames
                        if pose_buffer.shape[0] > n_frames:
                            pose_buffer = pose_buffer[-n_frames:]

                        print("pose_buffer shape:", pose_buffer.shape)

                        if pose_buffer.shape[0] == n_frames:
                            logging.info("Pose buffer full, running prediction...")
                            t = time.process_time()

                            action, confidence = predict_action(pose_buffer)
                            print(f"Action: {action}, Confidence: {confidence:.2f}")

                            inference_time = time.process_time() - t

                    end_time = time.monotonic()
                    sum_process_time += 1000 * (end_time - start_time)
                    sum_inference_time += inference_time * 1000
                    n += 1

                    avg_inference_time = sum_inference_time / n
                    text_line = "MoveNet: %.1fms (%.2f fps) TrueFPS: Null Nposes %d" % (
                        avg_inference_time,
                        1000 / (avg_inference_time + 1e-6),
                        len(outputs),
                    )

                    if action != "I'm not sure" and action != "none detected":
                        # connvert confidence from float32 to json serializable
                        confidence = float(confidence)
                        payload = {
                            "camera_id": device_id,
                            "action": action,
                            "confidence": confidence,
                        }
                        await ws.send(json.dumps(payload))
                        logging.info(f"Sent recognition: {payload}")

                    if action == "jumping":
                        alert_payload = {
                            "type": "alert",
                            "alert_level": "warning",
                            "camera_id": device_id,
                            "message": f"High confidence gesture: {action}",
                            "details": {
                                "action": action,
                                "confidence": confidence,
                                "timestamp": str(datetime.datetime.now()),
                            },
                        }
                        await ws.send(json.dumps(alert_payload))

                    await asyncio.sleep(0.1)

        except Exception as e:
            logging.warning(f"WebSocket error: {e}")
            retry_attempts += 1
            await asyncio.sleep(min(2**retry_attempts, 30))  # exponential backoff


async def register_camera(device_id, server_url):
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/create/", json={"name": f"Camera_{device_id}", "camera_id": device_id}
            )
            return response.status in [200, 201]
        except Exception as e:
            logging.error(f"Camera registration failed: {e}")
            return False


async def set_connection(device_id, server_url, connected):
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/connect/", json={"camera_id": device_id, "connected": connected}
            )
            return response.status in [200, 201]
        except Exception as e:
            logging.error(f"Set connection failed: {e}")
            return False


async def set_stream(device_id, server_url, stream):
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/set_stream/", json={"camera_id": device_id, "stream": stream}
            )
            return response.status in [200, 201]
        except Exception as e:
            logging.error(f"Set stream failed: {e}")
            return False


async def control_stream(device_id, video_track, server_url):
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        while True:
            try:
                async with session.get(f"{server_url}/cameras/get/{device_id}") as resp:
                    cmd = await resp.json()
                    if cmd.get("stream"):
                        await video_track.start_stream()
                    else:
                        await video_track.stop_stream()
            except Exception as e:
                logging.warning(f"Polling stream state error: {e}")
            await asyncio.sleep(1)


async def run(pc, session, cloud_server_url, camera_device, device_id):
    pc.configuration = RTCConfiguration(iceServers=ice_servers)

    if not await register_camera(device_id, cloud_server_url):
        logging.error("Camera registration failed.")
        return

    if not await set_connection(device_id, cloud_server_url, True):
        logging.error("Connection setup failed.")
        return

    video_track = VideoCaptureTrack(device=camera_device)
    pc.addTrack(video_track)

    @pc.on("icecandidate")
    def on_ice_candidate(candidate):
        if candidate:
            logging.info(f"New ICE Candidate: {candidate}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        logging.info(f"ICE state: {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "disconnected"]:
            await pc.close()

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await asyncio.sleep(2)

    response = await session.post(
        f"{cloud_server_url}/offer",
        json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "role": "publisher",
            "device_id": device_id,
        },
    )

    if response.status != 200:
        logging.error(f"Offer failed: {response.status}")
        return

    answer = await response.json()
    if answer.get("error"):
        logging.error(f"SDP Error: {answer['error']}")
        return

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    logging.info("SDP handshake complete.")

    if not await set_stream(device_id, cloud_server_url, True):

        logging.error("Stream setup failed.")
        return

    asyncio.create_task(control_stream(device_id, video_track, cloud_server_url))


async def cleanup(device_id, cloud_server_url):
    logging.info("Cleaning up before exit...")
    try:
        await set_connection(device_id, cloud_server_url, False)
        await set_stream(device_id, cloud_server_url, False)
    except Exception as e:
        logging.warning(f"Cleanup error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Jetson Publisher with WebRTC + WebSocket")
    parser.add_argument("--server", type=str, default="https://senior-backend.xyz", help="Backend server URL")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    print("HELLO")
    # parser.add_argument("--camera", type=str, default="0",
    #                 help="Camera index (0,1,…) or path to video file")("--device_id", type=str, required=True, help="Unique camera ID")
    parser.add_argument("--device_id", type=str, required=True, help="Unique camera ID")
    args = parser.parse_args()
    print("HELLO")

    pc = RTCPeerConnection()
    server_url = args.server
    ws_protocol = "wss" if server_url.startswith("https") else "ws"
    ws_host = server_url.removeprefix("https://").removeprefix("http://")
    ws_url = f"{ws_protocol}://{ws_host}/ws/jetson"

    print("HELLO")

    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        try:
            await run(pc, session, server_url, args.camera, args.device_id)
            asyncio.create_task(recognition_loop(args.device_id, ws_url))
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await cleanup(args.device_id, server_url)


def torso_visible(keypoints):
    """Checks whether there are enough torso keypoints.

    This function checks whether the model is confident at predicting one of the
    shoulders/hips which is required to determine a good crop region.
    """
    return (
        keypoints[0, 0, KEYPOINT_DICT["left_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
        or keypoints[0, 0, KEYPOINT_DICT["right_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
    ) and (
        keypoints[0, 0, KEYPOINT_DICT["left_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
        or keypoints[0, 0, KEYPOINT_DICT["right_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
    )


def determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x):
    """Calculates the maximum distance from each keypoints to the center location.

    The function returns the maximum distances from the two sets of keypoints:
    full 17 keypoints and 4 torso keypoints. The returned information will be
    used to determine the crop size. See determineCropRegion for more detail.
    """
    torso_joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y

        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


def init_crop_region(image_height, image_width):
    """Defines the default crop region.

    The function provides the initial crop region (pads the full image from both
    sides to make it a square image) when the algorithm cannot reliably determine
    the crop region from the previous frame.
    """
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        "y_min": y_min,
        "x_min": x_min,
        "y_max": y_min + box_height,
        "x_max": x_min + box_width,
        "height": box_height,
        "width": box_width,
    }


def determine_crop_region(keypoints, image_height, image_width):
    """Determines the region to crop the image for the model to run inference on.

    The algorithm uses the detected joints from the previous frame to estimate
    the square region that encloses the full body of the target person and
    centers at the midpoint of two hip joints. The crop size is determined by
    the distances between each joints and the center point.
    When the model is not confident with the four torso joint predictions, the
    function returns a default crop which is the full image padded to square.
    """
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width,
        ]

    if torso_visible(keypoints):
        center_y = (target_keypoints["left_hip"][0] + target_keypoints["right_hip"][0]) / 2
        center_x = (target_keypoints["left_hip"][1] + target_keypoints["right_hip"][1]) / 2

        (max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
            keypoints, target_keypoints, center_y, center_x
        )

        crop_length_half = np.amax(
            [
                max_torso_xrange * 1.9,
                max_torso_yrange * 1.9,
                max_body_yrange * 1.2,
                max_body_xrange * 1.2,
            ]
        )

        tmp = np.array([center_x, image_width - center_x, center_y, image_height - center_y])
        crop_length_half = np.amin([crop_length_half, np.amax(tmp)])

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2
            return {
                "y_min": crop_corner[0] / image_height,
                "x_min": crop_corner[1] / image_width,
                "y_max": (crop_corner[0] + crop_length) / image_height,
                "x_max": (crop_corner[1] + crop_length) / image_width,
                "height": (crop_corner[0] + crop_length) / image_height - crop_corner[0] / image_height,
                "width": (crop_corner[1] + crop_length) / image_width - crop_corner[1] / image_width,
            }
    else:
        return init_crop_region(image_height, image_width)


def crop_and_resize(image, crop_region, crop_size):
    """Crops and resize the image to prepare for the model input."""
    boxes = [
        [
            crop_region["y_min"],
            crop_region["x_min"],
            crop_region["y_max"],
            crop_region["x_max"],
        ]
    ]
    output_image = tf.image.crop_and_resize(image, box_indices=[0], boxes=boxes, crop_size=crop_size)
    return output_image


C1 = 1
C2 = 2
M1 = 7
M2 = 8
H = [0, 1, 2, 3, 4]
RF = [15]
LF = [16]

INPUT_SIZE = 256
SRC_SIZE = (INPUT_SIZE, INPUT_SIZE)

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

CLASSES = [
    "standing",
    "check watch",
    "cross arms",
    "scratch head",
    "sit down",
    "get up",
    "turn around",
    "walking",
    "wave1",
    "boxing",
    "kicking",
    "pointing",
    "pick up",
    "bending",
    "hands clapping",
    "wave2",
    "jogging",
    "jumping",
    "pjump",
    "running",
]

MIN_CROP_KEYPOINT_SCORE = 0.1


def remove_confidence(X):
    Xr = X[..., :-1]
    return Xr


def invert_xy(X):
    Xi = X[..., ::-1]
    return Xi


def reduce_keypoints(X):
    to_prune = []
    for group in [H, RF, LF]:
        if len(group) > 1:
            to_prune.append(group[1:])
    to_prune = [item for sublist in to_prune for item in sublist]
    X[:, H[0], :] = np.true_divide(X[:, H].sum(1), (X[:, H] != 0).sum(1) + 1e-9)
    X[:, RF[0], :] = np.true_divide(X[:, RF].sum(1), (X[:, RF] != 0).sum(1) + 1e-9)
    X[:, LF[0], :] = np.true_divide(X[:, LF].sum(1), (X[:, LF] != 0).sum(1) + 1e-9)
    Xr = np.delete(X, to_prune, 1)
    return Xr


def add_velocity(X):
    T, K, C = X.shape
    v1, v2 = np.zeros((T + 1, K, C)), np.zeros((T + 1, K, C))
    v1[1:] = X
    v2[:T] = X
    vel = (v2 - v1)[:-1]
    Xv = np.concatenate((X, vel), axis=-1)
    return Xv


def scale_and_center(X):
    pose_list = []
    for pose in X:
        zero_point = (pose[C1, :2] + pose[C2, :2]) / 2
        module_keypoint = (pose[M1, :2] + pose[M2, :2]) / 2
        scale_mag = np.linalg.norm(zero_point - module_keypoint)
        if scale_mag < 1:
            scale_mag = 1
        pose[:, :2] = (pose[:, :2] - zero_point) / scale_mag
        pose_list.append(pose)
    Xn = np.stack(pose_list)
    return Xn


def flatten_features(X):
    Xf = X.reshape(X.shape[0], -1)
    return Xf


def preprocess(X):
    X = remove_confidence(X)
    X = invert_xy(X)
    X = reduce_keypoints(X)
    X = add_velocity(X)
    X = scale_and_center(X)
    X = flatten_features(X)
    return X


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
