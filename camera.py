"""
Camera — wraps a Transform and owns the projection pipeline.

A Camera is NOT a scene-graph node (it has no parent/children); it is a
first-class object that a Scene references.  This keeps the common case
(one free-flying camera) simple while leaving room to attach a camera to
an object by reading that object's world_matrix.

Public surface that a PyGame developer touches
----------------------------------------------
  cam = Camera(fov_y=60, aspect=16/9, near=0.1, far=1000)
  cam.position = (0, 2, -5)
  cam.look_at((0, 0, 0))

  # Per-frame (called by Scene):
  vp = cam.view_projection_matrix   # 4×4, recomputed lazily

  # Picking (on mouse click):
  ray_origin, ray_dir = cam.screen_to_ray(mouse_x, mouse_y, screen_w, screen_h)
"""

from __future__ import annotations
import numpy as np
from transform import Transform


class Camera:
    """
    Perspective camera with lazy view-projection matrix caching.

    Projection convention
    ---------------------
    Depth is mapped to [0, 1] (reverse-Z friendly, Qwen's projection).
    NDC → PyGame pixels is handled by the ScreenMapper in rendering/.

    Dirty-flag strategy
    -------------------
    Camera tracks its own _view_dirty and _proj_dirty booleans.
    view_projection_matrix re-bakes both halves only when needed.
    """

    def __init__(
        self,
        fov_y: float = 60.0,
        aspect: float = 16 / 9,
        near: float = 0.1,
        far: float = 1000.0,
    ) -> None:
        self._transform = Transform()      # internal; not in scene graph
        self._fov_y    = float(fov_y)
        self._aspect   = float(aspect)
        self._near     = float(near)
        self._far      = float(far)

        self._view_matrix      = np.eye(4, dtype=np.float32)
        self._proj_matrix      = np.eye(4, dtype=np.float32)
        self._view_proj_matrix = np.eye(4, dtype=np.float32)

        self._view_dirty = True
        self._proj_dirty = True

    # ------------------------------------------------------------------
    # internal Transform
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return self._transform.position

    @position.setter
    def position(self, value: tuple[float, float, float] | np.ndarray) -> None:
        self._transform.position = value
        self._view_dirty = True

    @property
    def rotation(self) -> np.ndarray:
        """Quaternion [w, x, y, z]."""
        return self._transform.rotation

    @rotation.setter
    def rotation(self, value: tuple[float, float, float, float] | np.ndarray) -> None:
        self._transform.rotation = value
        self._view_dirty = True

    # ------------------------------------------------------------------
    # projection parameters
    # ------------------------------------------------------------------

    @property
    def fov_y(self) -> float:
        return self._fov_y

    @fov_y.setter
    def fov_y(self, value: float) -> None:
        self._fov_y = float(value)
        self._proj_dirty = True

    @property
    def aspect(self) -> float:
        return self._aspect

    @aspect.setter
    def aspect(self, value: float) -> None:
        self._aspect = float(value)
        self._proj_dirty = True

    @property
    def near(self) -> float:
        return self._near

    @near.setter
    def near(self, value: float) -> None:
        self._near = float(value)
        self._proj_dirty = True

    @property
    def far(self) -> float:
        return self._far

    @far.setter
    def far(self, value: float) -> None:
        self._far = float(value)
        self._proj_dirty = True

    # ------------------------------------------------------------------
    # orientation helpers
    # ------------------------------------------------------------------

    def look_at(
        self,
        target: tuple[float, float, float] | np.ndarray,
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        """
        Rotate the camera so it faces *target*.
        Delegates quaternion math to math_core (Qwen's domain).
        """
        from math_core import look_at_quaternion  # type: ignore[import]
        q = look_at_quaternion(
            np.asarray(self._transform.position, dtype=np.float32),
            np.asarray(target,                   dtype=np.float32),
            np.asarray(up,                       dtype=np.float32),
        )
        self._transform.rotation = q
        self._view_dirty = True

    def orbit(self, d_yaw: float, d_pitch: float, pivot: tuple = (0, 0, 0)) -> None:
        """
        Orbit around *pivot* by delta yaw / pitch in degrees.
        Useful for editor-style mouse-drag rotation.
        """
        from math_core import (  # type: ignore[import]
            quaternion_from_axis_angle,
            quaternion_multiply,
        )
        pivot_arr = np.array(pivot, dtype=np.float32)
        pos       = self._transform.position - pivot_arr

        q_yaw   = quaternion_from_axis_angle((0, 1, 0), np.radians(d_yaw))
        q_pitch = quaternion_from_axis_angle((1, 0, 0), np.radians(d_pitch))
        q_delta = quaternion_multiply(q_yaw, q_pitch)

        from math_core import quaternion_rotate_vector  # type: ignore[import]
        new_pos = quaternion_rotate_vector(q_delta, pos) + pivot_arr
        self._transform.position = new_pos

        new_rot = quaternion_multiply(q_delta, self._transform.rotation)
        self._transform.rotation = new_rot
        self._view_dirty = True

    def pan(self, dx: float, dy: float) -> None:
        """Translate in camera-local XY plane (right / up)."""
        from math_core import quaternion_rotate_vector  # type: ignore[import]
        q     = self._transform.rotation
        right = quaternion_rotate_vector(q, np.array([1, 0, 0], np.float32))
        up    = quaternion_rotate_vector(q, np.array([0, 1, 0], np.float32))
        self._transform.position = (
            self._transform.position + right * dx + up * dy
        )
        self._view_dirty = True

    def zoom(self, dz: float) -> None:
        """Move along the camera's forward axis."""
        from math_core import quaternion_rotate_vector  # type: ignore[import]
        q       = self._transform.rotation
        forward = quaternion_rotate_vector(q, np.array([0, 0, -1], np.float32))
        self._transform.position = self._transform.position + forward * dz
        self._view_dirty = True

    # ------------------------------------------------------------------
    # Matrix properties
    # ------------------------------------------------------------------

    @property
    def view_matrix(self) -> np.ndarray:
        if self._view_dirty:
            self._rebuild_view()
        return self._view_matrix

    @property
    def projection_matrix(self) -> np.ndarray:
        if self._proj_dirty:
            self._rebuild_proj()
        return self._proj_matrix

    @property
    def view_projection_matrix(self) -> np.ndarray:
        if self._view_dirty:
            self._rebuild_view()
        if self._proj_dirty:
            self._rebuild_proj()
        if self._view_dirty or self._proj_dirty:
            self._view_proj_matrix = self._proj_matrix @ self._view_matrix
        return self._view_proj_matrix

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def screen_to_ray(
        self,
        screen_x: int,
        screen_y: int,
        screen_w: int,
        screen_h: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Unproject a screen pixel into a world-space ray.
        Returns (origin, direction) both as float32 ndarray.

        The screen_mapper Y-flip is the inverse of what we applied during
        rendering, so we just reverse it here.
        """
        from math_core import unproject_ray  # type: ignore[import]
        return unproject_ray(
            screen_x, screen_y, screen_w, screen_h,
            self.view_matrix,
            self.projection_matrix,
        )

    # ------------------------------------------------------------------
    # Frustum planes (used by Scene culling)
    # ------------------------------------------------------------------

    @property
    def frustum_planes(self) -> np.ndarray:
        """
        Six planes as (normal_x, normal_y, normal_z, d) in world space.
        Recomputed whenever the VP matrix changes.
        """
        from math_core import extract_frustum_planes  # type: ignore[import]
        return extract_frustum_planes(self.view_projection_matrix)

    # ------------------------------------------------------------------
    # Window-resize helper
    # ------------------------------------------------------------------

    def set_aspect_from_size(self, width: int, height: int) -> None:
        self.aspect = width / max(height, 1)

    # ------------------------------------------------------------------
    # Internal rebuild methods
    # ------------------------------------------------------------------

    def _rebuild_view(self) -> None:
        from math_core import view_matrix_from_transform  # type: ignore[import]
        self._view_matrix = view_matrix_from_transform(
            self._transform.position,
            self._transform.rotation,
        )
        self._view_dirty = False
        # Row-major: v_clip = v @ View @ Proj, so VP = View @ Proj
        self._view_proj_matrix = self._view_matrix @ self._proj_matrix

    def _rebuild_proj(self) -> None:
        from math_core import perspective_matrix  # type: ignore[import]
        self._proj_matrix = perspective_matrix(
            np.radians(self._fov_y),
            self._aspect,
            self._near,
            self._far,
        )
        self._proj_dirty = False
        self._view_proj_matrix = self._view_matrix @ self._proj_matrix

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        p = self._transform.position
        return (
            f"Camera(pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}), "
            f"fov={self._fov_y}°, near={self._near}, far={self._far})"
        )