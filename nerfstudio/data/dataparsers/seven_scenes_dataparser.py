'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2025-05-27 12:04:59
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-12-19 14:50:21
FilePath: /nerfstudio/nerfstudio/data/dataparsers/seven_scenes_dataparser.py
Description: Data parser for 7scenes dataset
'''

"""Data parser for 7scenes dataset"""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import imageio
import numpy as np
import torch
from loguru import logger
from math import floor

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class SevenScenesDataParserConfig(DataParserConfig):
    """7Scenes dataset parser config"""

    _target: Type = field(default_factory=lambda: SevenScenes)
    """target class to instantiate"""
    data: Path = Path("/opt/data/private/datasets/NVS/7scenes_chess")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """Scale of the scene, used to define the scene box."""
    patch_size: int = 16
    """Patch size for the images, used for training."""
    vggt_flag: bool = False
    """Whether to use VGGT input settings."""


def convert_opencv_to_opengl(opencv_mat, transform_type='world2cam'):
    # If the input matrix is cam2world, invert it to get world2cam
    if transform_type == 'cam2world':
        opencv_mat = np.linalg.inv(opencv_mat)

    # This matrix converts between opengl (x right, y up, z back) and cv-style (x right, y down, z forward) coordinates
    # For nerfstudio, we want opengl coordinates
    coord_transform = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])

    # NB for the following we expect the mat to be in world2cam format
    opengl_mat = coord_transform @ opencv_mat

    if transform_type == 'cam2world':
        opengl_mat = np.linalg.inv(opengl_mat)

    return opengl_mat


@dataclass
class SevenScenes(DataParser):
    """7Scenes Dataset
    """
    config: SevenScenesDataParserConfig # type: ignore

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs) -> DataparserOutputs:
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        if split == "val": split = "test"  # 7Scenes does not have a validation set, so we use the test set instead.
        meta = load_from_json(self.config.data / f"transforms_{split}.json")
        focal_length = meta["f"]
        image_filenams = []
        poses = []
        for frame in meta["frames"]:
            fname = self.config.data / Path(frame["file_path"])
            assert fname.exists(), f"Image file {fname} does not exist."
            image_filenams.append(str(fname))
            pose_temp = np.array(frame["transform_matrix"], dtype=np.float32)  # 4 x 4, suppose camera to world transform
            # pose_temp = np.linalg.inv(pose_temp)  # convert from camera to world to world to camera
            pose_temp = convert_opencv_to_opengl(pose_temp, transform_type='cam2world')  # convert to OpenGL coordinates
            poses.append(pose_temp)  # 4 x 4,
        
        poses = np.array(poses, dtype=np.float32)

        img_0 = imageio.imread(image_filenams[0])
        image_height, image_width = img_0.shape[:2]

        cx = image_width / 2.0
        cy = image_height / 2.0
        
        c2w = torch.from_numpy(poses[:, :3])

        c2w[..., 3] *= self.config.scale_factor
        
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            camera_type=CameraType.PERSPECTIVE,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_to_worlds=c2w,
        )


        #@EasonZhang: 250603: use vggt input settings if vggt_flag is set
        if self.config.vggt_flag:
            w_temp = cameras.width[0]
            h_temp = cameras.height[0]
            target_width = 518
            patch_size = 14
            w_new = target_width
            h_new = floor(h_temp * (w_new / w_temp) / patch_size) * patch_size

            w_factor = float(w_new / w_temp)
            h_factor = float(h_new / h_temp)

            # logger.critical(f"Rescaling cameras from {w_temp}x{h_temp} to {w_new}x{h_new} with factors {w_factor}x{h_factor}")

            cameras.rescale_output_resolution_uv(
                scaling_factor_w=w_factor,
                scaling_factor_h=h_factor,
            )

            assert cameras.width[0] % patch_size == 0, f"width {cameras.width[0]} is not divisible by {patch_size}"
            assert cameras.height[0] % patch_size == 0, f"height {cameras.height[0]} is not divisible by {patch_size}"


        # @EasonZhang: 250520
        elif self.config.patch_size > 1:
            # make the image size and camera intrinsics devisible by patch_size
            w_temp = cameras.width[0]
            h_temp = cameras.height[0]
            w_new = w_temp // self.config.patch_size * self.config.patch_size
            h_new = h_temp // self.config.patch_size * self.config.patch_size
            w_factor = w_new / w_temp
            h_factor = h_new / h_temp
            w_factor = float(w_factor)
            h_factor = float(h_factor)
            cameras.rescale_output_resolution_uv(
                scaling_factor_w=w_factor,
                scaling_factor_h=h_factor,
            )

            assert cameras.width[0] % self.config.patch_size == 0, f"width {cameras.width[0]} is not divisible by {self.config.patch_size}"
            assert cameras.height[0] % self.config.patch_size == 0, f"height {cameras.height[0]} is not divisible by {self.config.patch_size}"


        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenams,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale= self.config.scale_factor,
        )

        return dataparser_outputs