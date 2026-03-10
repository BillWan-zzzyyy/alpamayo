"""加载 nuScenes 数据集用于 Alpamayo-R1 模型推理。

直接读取 nuScenes JSON 元数据文件（不依赖 nuscenes-devkit），
输出格式与 load_physical_aiavdataset() 完全一致。
"""

import json
import os
from typing import Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

# nuScenes 摄像头 -> Physical AI AV 摄像头索引映射（最佳近似）
NUSCENES_CAMERA_MAPPING = [
    ("CAM_FRONT_LEFT", 0),   # ~ CAMERA_CROSS_LEFT_120FOV
    ("CAM_FRONT", 1),        # ~ CAMERA_FRONT_WIDE_120FOV
    ("CAM_FRONT_RIGHT", 2),  # ~ CAMERA_CROSS_RIGHT_120FOV
    ("CAM_FRONT", 6),        # ~ CAMERA_FRONT_TELE_30FOV（无直接对应，复用 CAM_FRONT）
]


class NuScenesDataInterface:
    """轻量级 nuScenes 元数据接口，初始化时加载 JSON 并构建索引。

    Args:
        dataroot: nuScenes 数据集根目录
                  （如 "/data/dataset/nuscenes/v1.0-trainval"）
    """

    def __init__(self, dataroot: str = "/data/dataset/nuscenes/v1.0-trainval"):
        self.dataroot = dataroot
        meta_dir = os.path.join(dataroot, "v1.0-trainval")

        self._scene = self._load_json(meta_dir, "scene.json")
        self._sample = self._load_json(meta_dir, "sample.json")
        self._sample_data = self._load_json(meta_dir, "sample_data.json")
        self._ego_pose = self._load_json(meta_dir, "ego_pose.json")
        self._calibrated_sensor = self._load_json(meta_dir, "calibrated_sensor.json")
        self._sensor = self._load_json(meta_dir, "sensor.json")

        self._build_indices()

    @staticmethod
    def _load_json(meta_dir: str, filename: str) -> list[dict]:
        with open(os.path.join(meta_dir, filename)) as f:
            return json.load(f)

    def _build_indices(self):
        """构建基于 token 的查找表。"""
        self.scene_by_token = {r["token"]: r for r in self._scene}
        self.sample_by_token = {r["token"]: r for r in self._sample}
        self.sd_by_token = {r["token"]: r for r in self._sample_data}
        self.ego_pose_by_token = {r["token"]: r for r in self._ego_pose}
        self.cs_by_token = {r["token"]: r for r in self._calibrated_sensor}
        self.sensor_by_token = {r["token"]: r for r in self._sensor}

        self.scene_by_name = {r["name"]: r for r in self._scene}

        self._channel_by_cs_token: dict[str, str] = {}
        for cs in self._calibrated_sensor:
            self._channel_by_cs_token[cs["token"]] = self.sensor_by_token[
                cs["sensor_token"]
            ]["channel"]

        # (sample_token, channel) -> 关键帧 sample_data 记录
        self._keyframe_sd: dict[tuple[str, str], dict] = {}
        for sd in self._sample_data:
            if sd["is_key_frame"]:
                channel = self._channel_by_cs_token[sd["calibrated_sensor_token"]]
                self._keyframe_sd[(sd["sample_token"], channel)] = sd

    def list_scenes(self) -> list[dict[str, Any]]:
        """列出所有可用场景。

        Returns:
            包含 name, token, nbr_samples, description 的字典列表
        """
        return [
            {
                "name": s["name"],
                "token": s["token"],
                "nbr_samples": s["nbr_samples"],
                "description": s["description"],
            }
            for s in self._scene
        ]

    def get_scene_ego_poses(
        self, scene_token: str, buffer_us: int = 10_000_000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """通过 LIDAR_TOP 链表收集场景内所有 ego_pose。

        Args:
            scene_token: 场景 token
            buffer_us: 场景首尾时间戳的缓冲区（微秒），确保插值覆盖范围足够

        Returns:
            timestamps: (N,) int64，时间戳（微秒）
            translations: (N, 3) float64，世界坐标系位置
            rotations_wxyz: (N, 4) float64，四元数 (w,x,y,z) 格式
        """
        scene = self.scene_by_token[scene_token]
        first_sample = self.sample_by_token[scene["first_sample_token"]]
        last_sample = self.sample_by_token[scene["last_sample_token"]]

        bound_start = first_sample["timestamp"] - buffer_us
        bound_end = last_sample["timestamp"] + buffer_us

        lidar_sd = self._keyframe_sd.get((first_sample["token"], "LIDAR_TOP"))
        if lidar_sd is None:
            raise ValueError(f"场景 {scene['name']} 缺少 LIDAR_TOP 数据")

        current = lidar_sd
        while current["prev"]:
            prev_sd = self.sd_by_token[current["prev"]]
            if prev_sd["timestamp"] < bound_start:
                break
            current = prev_sd

        timestamps, translations, rotations = [], [], []
        while current:
            if current["timestamp"] > bound_end:
                break
            ep = self.ego_pose_by_token[current["ego_pose_token"]]
            timestamps.append(ep["timestamp"])
            translations.append(ep["translation"])
            rotations.append(ep["rotation"])
            current = (
                self.sd_by_token.get(current["next"]) if current["next"] else None
            )

        return (
            np.array(timestamps, dtype=np.int64),
            np.array(translations, dtype=np.float64),
            np.array(rotations, dtype=np.float64),
        )

    def build_ego_interpolator(
        self,
        timestamps_us: np.ndarray,
        translations: np.ndarray,
        rotations_wxyz: np.ndarray,
    ):
        """构建 ego 位姿插值器（平移线性插值 + 旋转 Slerp 插值）。

        Args:
            timestamps_us: (N,) 时间戳（微秒）
            translations: (N, 3) 位置
            rotations_wxyz: (N, 4) 四元数，nuScenes (w,x,y,z) 格式

        Returns:
            可调用对象：输入 query_us (int64 数组) -> (translations, Rotation)
        """
        order = np.argsort(timestamps_us)
        timestamps_us = timestamps_us[order]
        translations = translations[order]
        rotations_wxyz = rotations_wxyz[order]

        _, unique_idx = np.unique(timestamps_us, return_index=True)
        timestamps_us = timestamps_us[unique_idx]
        translations = translations[unique_idx]
        rotations_wxyz = rotations_wxyz[unique_idx]

        # nuScenes (w,x,y,z) -> scipy (x,y,z,w)
        rotations_xyzw = rotations_wxyz[:, [1, 2, 3, 0]]

        ts_sec = timestamps_us.astype(np.float64) * 1e-6
        trans_interp = interp1d(ts_sec, translations, axis=0, kind="linear")
        rot_slerp = Slerp(ts_sec, Rotation.from_quat(rotations_xyzw))

        t_min, t_max = timestamps_us[0], timestamps_us[-1]

        def interpolate(
            query_us: np.ndarray,
        ) -> tuple[np.ndarray, Rotation]:
            if query_us.min() < t_min or query_us.max() > t_max:
                raise ValueError(
                    f"查询时间范围 [{query_us.min()}, {query_us.max()}] "
                    f"超出 ego_pose 覆盖范围 [{t_min}, {t_max}]，"
                    f"请调整 t0 或选择其他场景"
                )
            q_sec = query_us.astype(np.float64) * 1e-6
            return trans_interp(q_sec), rot_slerp(q_sec)

        return interpolate

    def find_nearest_camera_images(
        self,
        scene_token: str,
        channel: str,
        target_timestamps_us: np.ndarray,
    ) -> tuple[list[str], np.ndarray]:
        """查找最接近目标时间戳的摄像头图像。

        Args:
            scene_token: 场景 token
            channel: 摄像头通道名（如 "CAM_FRONT"）
            target_timestamps_us: (M,) 目标时间戳（微秒）

        Returns:
            file_paths: M 个绝对路径
            actual_timestamps: (M,) 实际图像时间戳
        """
        scene = self.scene_by_token[scene_token]
        first_sample = self.sample_by_token[scene["first_sample_token"]]
        last_sample = self.sample_by_token[scene["last_sample_token"]]

        kf_sd = self._keyframe_sd.get((first_sample["token"], channel))
        if kf_sd is None:
            raise ValueError(f"场景 {scene['name']} 缺少 {channel} 数据")

        bound_start = first_sample["timestamp"] - 5_000_000
        current = kf_sd
        while current["prev"]:
            prev_sd = self.sd_by_token[current["prev"]]
            if prev_sd["timestamp"] < bound_start:
                break
            current = prev_sd

        bound_end = last_sample["timestamp"] + 5_000_000
        entries: list[dict] = []
        while current:
            if current["timestamp"] > bound_end:
                break
            entries.append(current)
            current = (
                self.sd_by_token.get(current["next"]) if current["next"] else None
            )

        sd_timestamps = np.array([e["timestamp"] for e in entries], dtype=np.int64)

        file_paths = []
        actual_ts = []
        for target in target_timestamps_us:
            idx = int(np.argmin(np.abs(sd_timestamps - target)))
            sd = entries[idx]
            file_paths.append(os.path.join(self.dataroot, sd["filename"]))
            actual_ts.append(sd["timestamp"])

        return file_paths, np.array(actual_ts, dtype=np.int64)


def load_nuscenes(
    scene_name: str,
    t0_us: int | None = None,
    t0_offset_s: float = 5.0,
    ndi: NuScenesDataInterface | None = None,
    dataroot: str = "/data/dataset/nuscenes/v1.0-trainval",
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_mapping: list[tuple[str, int]] | None = None,
    num_frames: int = 4,
) -> dict[str, Any]:
    """从 nuScenes 加载一个样本用于 Alpamayo-R1 推理。

    输出格式与 load_physical_aiavdataset() 完全一致。

    Args:
        scene_name: nuScenes 场景名（如 "scene-0001"）
        t0_us: 绝对 t0 时间戳（微秒）。为 None 时自动计算为
               first_keyframe + t0_offset_s * 1e6
        t0_offset_s: 从首个关键帧算起的偏移秒数（仅当 t0_us 为 None 时使用）
        ndi: 预初始化的 NuScenesDataInterface（避免重复加载元数据）
        dataroot: nuScenes 数据集根目录
        num_history_steps: 历史轨迹步数（默认 16 = 1.6s @ 10Hz）
        num_future_steps: 未来轨迹步数（默认 64 = 6.4s @ 10Hz）
        time_step: 时间步长（秒，默认 0.1 = 10Hz）
        camera_mapping: (nuScenes通道, 摄像头索引) 列表，默认 NUSCENES_CAMERA_MAPPING
        num_frames: 每个摄像头的帧数（默认 4）

    Returns:
        字典，包含 image_frames, camera_indices, ego_history_xyz,
        ego_history_rot, ego_future_xyz, ego_future_rot, relative_timestamps,
        absolute_timestamps, t0_us, scene_name

    Example:
        ndi = NuScenesDataInterface()
        data = load_nuscenes("scene-0061", ndi=ndi)
    """
    if ndi is None:
        ndi = NuScenesDataInterface(dataroot)

    if camera_mapping is None:
        camera_mapping = NUSCENES_CAMERA_MAPPING

    scene = ndi.scene_by_name[scene_name]
    scene_token = scene["token"]

    if t0_us is None:
        first_sample = ndi.sample_by_token[scene["first_sample_token"]]
        t0_us = first_sample["timestamp"] + int(t0_offset_s * 1_000_000)

    # ---- ego 轨迹 ----
    ts_raw, trans_raw, rot_raw = ndi.get_scene_ego_poses(scene_token)
    ego_interp = ndi.build_ego_interpolator(ts_raw, trans_raw, rot_raw)

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_ts = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_ts = t0_us + future_offsets_us

    ego_hist_xyz, ego_hist_rot = ego_interp(history_ts)
    ego_fut_xyz, ego_fut_rot = ego_interp(future_ts)

    ego_hist_quat = ego_hist_rot.as_quat()  # scipy (x,y,z,w)
    ego_fut_quat = ego_fut_rot.as_quat()

    # 变换到 t0 时刻的局部坐标系
    t0_xyz = ego_hist_xyz[-1].copy()
    t0_rot = Rotation.from_quat(ego_hist_quat[-1])
    t0_rot_inv = t0_rot.inv()

    hist_xyz_local = t0_rot_inv.apply(ego_hist_xyz - t0_xyz)
    fut_xyz_local = t0_rot_inv.apply(ego_fut_xyz - t0_xyz)
    hist_rot_local = (t0_rot_inv * Rotation.from_quat(ego_hist_quat)).as_matrix()
    fut_rot_local = (t0_rot_inv * Rotation.from_quat(ego_fut_quat)).as_matrix()

    ego_h_xyz = torch.from_numpy(hist_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_h_rot = torch.from_numpy(hist_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_f_xyz = torch.from_numpy(fut_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_f_rot = torch.from_numpy(fut_rot_local).float().unsqueeze(0).unsqueeze(0)

    # ---- 摄像头图像 ----
    img_ts = np.array(
        [
            t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000)
            for i in range(num_frames)
        ],
        dtype=np.int64,
    )

    frames_list, idx_list, ts_list = [], [], []
    for ns_channel, cam_idx in camera_mapping:
        paths, actual_ts = ndi.find_nearest_camera_images(
            scene_token, ns_channel, img_ts
        )
        imgs = np.stack([np.array(Image.open(p).convert("RGB")) for p in paths])
        frames_tensor = rearrange(torch.from_numpy(imgs), "t h w c -> t c h w")

        frames_list.append(frames_tensor)
        idx_list.append(cam_idx)
        ts_list.append(torch.from_numpy(actual_ts))

    image_frames = torch.stack(frames_list, dim=0)
    camera_indices = torch.tensor(idx_list, dtype=torch.int64)
    all_timestamps = torch.stack(ts_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_h_xyz,
        "ego_history_rot": ego_h_rot,
        "ego_future_xyz": ego_f_xyz,
        "ego_future_rot": ego_f_rot,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": t0_us,
        "scene_name": scene_name,
    }
