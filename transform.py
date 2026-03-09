"""
Transform — position, rotation (quaternion), scale with dirty-flag propagation.

This class is the backbone of the scene graph.  Every WireframeObject owns
exactly one Transform.  Dirty flags mean we only recompute world matrices when
something actually changed, keeping the render loop cache-friendly.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np

from math_core import trs_matrix


class Transform:
    """
    Local-space transform with lazy world-matrix computation.

    Coordinate convention
    ---------------------
    * Right-handed, Y-up (matching OpenGL / most 3D literature).
    * Stored as: local_position (vec3), local_rotation (unit quaternion [w,x,y,z]),
      local_scale (vec3).
    * World matrix is recomputed lazily whenever _dirty is True.

    Dirty-flag propagation
    ----------------------
    Whenever a local property changes:
      1. This transform marks itself dirty.
      2. It recurses into all registered children and marks them dirty too,
         because their world matrix depends on the parent's.
    Children register themselves via _add_child / _remove_child (called
    automatically by WireframeObject when re-parenting).
    """

    __slots__ = (
        "_local_position",
        "_local_rotation",   # [w, x, y, z]
        "_local_scale",
        "_local_matrix",     # 4×4 float32, local TRS
        "_world_matrix",     # 4×4 float32, accumulated from root
        "_dirty",
        "_parent",           # Transform | None
        "_children",         # list[Transform]
    )

    def __init__(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self._local_position = np.array(position, dtype=np.float32)
        self._local_rotation = np.array(rotation, dtype=np.float32)  # w,x,y,z
        self._local_scale    = np.array(scale,    dtype=np.float32)
        self._local_matrix   = np.eye(4, dtype=np.float32)
        self._world_matrix   = np.eye(4, dtype=np.float32)
        self._dirty          = True
        self._parent: Optional[Transform] = None
        self._children: List[Transform] = []

    # ------------------------------------------------------------------
    # Public property accessors — each setter propagates the dirty flag
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return self._local_position.copy()

    @position.setter
    def position(self, value: tuple[float, float, float] | np.ndarray) -> None:
        self._local_position[:] = value
        self._mark_dirty()

    @property
    def rotation(self) -> np.ndarray:
        """Quaternion [w, x, y, z]."""
        return self._local_rotation.copy()

    @rotation.setter
    def rotation(self, value: tuple[float, float, float, float] | np.ndarray) -> None:
        self._local_rotation[:] = value
        # Normalise in-place to guard against float drift
        n = np.linalg.norm(self._local_rotation)
        if n > 1e-8:
            self._local_rotation /= n
        self._mark_dirty()

    @property
    def scale(self) -> np.ndarray:
        return self._local_scale.copy()

    @scale.setter
    def scale(self, value: tuple[float, float, float] | np.ndarray) -> None:
        self._local_scale[:] = value
        self._mark_dirty()

    # ------------------------------------------------------------------
    # Matrix accessors — recompute lazily
    # ------------------------------------------------------------------

    @property
    def local_matrix(self) -> np.ndarray:
        if self._dirty:
            self._recompute()
        return self._local_matrix

    @property
    def world_matrix(self) -> np.ndarray:
        if self._dirty:
            self._recompute()
        return self._world_matrix

    # ------------------------------------------------------------------
    # Convenience mutators (chainable)
    # ------------------------------------------------------------------

    def translate(self, dx: float, dy: float, dz: float) -> "Transform":
        self._local_position += (dx, dy, dz)
        self._mark_dirty()
        return self

    def set_scale_uniform(self, s: float) -> "Transform":
        self._local_scale[:] = (s, s, s)
        self._mark_dirty()
        return self

    def set_euler_degrees(self, pitch: float, yaw: float, roll: float) -> "Transform":
        """
        Convenience wrapper: converts Euler angles (degrees) to a quaternion.
        """
        from math_core import euler_to_quaternion
        self.rotation = euler_to_quaternion(
            np.radians(pitch), np.radians(yaw), np.radians(roll)
        )
        return self

    # ------------------------------------------------------------------
    # Scene-graph wiring (called by WireframeObject, not by user code)
    # ------------------------------------------------------------------

    def _add_child(self, child: "Transform") -> None:
        if child not in self._children:
            self._children.append(child)
            child._parent = self
            child._mark_dirty()

    def _remove_child(self, child: "Transform") -> None:
        if child in self._children:
            self._children.remove(child)
            child._parent = None
            child._mark_dirty()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_dirty(self) -> None:
        """Mark this node and all descendants dirty (BFS)."""
        # Use stack for BFS to avoid recursion depth issues
        stack = [self]
        while stack:
            node = stack.pop()
            if not node._dirty:
                node._dirty = True
                stack.extend(node._children)

    def _recompute(self) -> None:
        """Rebuild local_matrix from TRS, then chain with parent world_matrix."""
        # Build local TRS matrix using math_core
        self._local_matrix = trs_matrix(
            self._local_position,
            self._local_rotation,
            self._local_scale,
        )

        # Chain with parent
        if self._parent is not None:
            parent_world = self._parent.world_matrix   # parent already clean
            self._world_matrix = parent_world @ self._local_matrix
        else:
            self._world_matrix = self._local_matrix.copy()

        self._dirty = False

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        p = self._local_position
        return (
            f"Transform(pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}), "
            f"dirty={self._dirty})"
        )