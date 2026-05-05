"""
Point cloud from MuJoCo MjModel/MjData (Gymnasium / mujoco>=3) using offscreen Renderer.
Compatible with depth+RGB pipeline used by FlowPolicy (Open3D + same extrinsic fix as mujoco_point_cloud).
"""
from __future__ import annotations

import math
from typing import List

import mujoco
import numpy as np
import open3d as o3d

from flow_policy_3d.gym_util.mujoco_point_cloud import (
    cammat2o3d,
    posRotMat2Mat,
    quat2Mat,
    rotMatList2NPRotMat,
)


class NativeMuJoCoPointCloudGenerator:
    """Build fused world-frame point clouds from named MuJoCo cameras."""

    def __init__(self, model: mujoco.MjModel, cam_names: List[str], img_size: int = 128):
        self.model = model
        self.cam_names = cam_names
        self.img_width = img_size
        self.img_height = img_size
        self._renderers: list[mujoco.Renderer] = []
        for _ in cam_names:
            r = mujoco.Renderer(model, height=img_size, width=img_size)
            self._renderers.append(r)

        self.cam_mats = []
        for name in cam_names:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            fovy = math.radians(model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(
                ((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1))
            )
            self.cam_mats.append(cam_mat)

    def close(self):
        for r in self._renderers:
            r.close()
        self._renderers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _depth_to_open3d(self, depth: np.ndarray) -> np.ndarray:
        """MuJoCo Renderer depth is distance along the view ray (meters); clip invalid sky."""
        depth = np.flipud(depth).copy()
        depth = np.where(np.isfinite(depth), depth, 0.0)
        depth = np.where(depth < 120.0, depth, 0.0)
        return depth

    def generate(self, data: mujoco.MjData, use_rgb: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Returns (N, 3) world points and stacked depth maps (one per camera)."""
        mujoco.mj_forward(self.model, data)
        o3d_clouds = []
        depths = []

        b2w_r = quat2Mat([0, 1, 0, 0])

        for cam_i, name in enumerate(self.cam_names):
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            cam_pos = data.cam_xpos[cam_id].copy()
            c2b_r = rotMatList2NPRotMat(data.cam_xmat[cam_id])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)

            ren = self._renderers[cam_i]
            ren.enable_depth_rendering()
            ren.update_scene(data, camera=name)
            depth_raw = ren.render()
            depth = self._depth_to_open3d(depth_raw)

            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth.astype(np.float32))
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
            transformed_cloud = o3d_cloud.transform(c2w)
            o3d_clouds.append(transformed_cloud)
            depths.append(depth)
            _ = use_rgb  # reserved if RGB–depth fusion is needed later

        combined = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined += cloud
        pts = np.asarray(combined.points).astype(np.float32)
        depth_stack = np.array(depths).squeeze()
        return pts, depth_stack
