"""
WireframeObject — a named, drawable node in the scene graph.

Design goals
------------
* Owns a Transform (local position/rotation/scale).
* Owns raw mesh data: vertices (Nx3 float32) + edges (Mx2 int indices).
* Can be parented to another WireframeObject; world transform is then the
  composition of all ancestors' local transforms.
* Exposes a clean API — no matrix math visible to the caller.
* Bounding sphere is computed once and cached (invalidated on vertex change).

Typical usage
-------------
    cube = WireframeObject.from_box(2, 2, 2, name="MyCube")
    cube.transform.position = (0, 1, 0)
    cube.visible = True
    cube.color   = (0, 255, 128)

    child = WireframeObject.from_box(0.5, 0.5, 0.5, name="SmallCube")
    child.set_parent(cube)              # child moves with cube
    child.transform.position = (1.5, 0, 0)   # relative to cube
"""

from __future__ import annotations
from typing import Optional, Iterator
import numpy as np

from transform import Transform


class WireframeObject:
    """
    Scene-graph node that carries renderable wireframe geometry.

    Attributes (read/write by user)
    --------------------------------
    name    : str           — human-readable identifier
    visible : bool          — skip rendering when False
    color   : tuple[int,int,int] — RGB, 0-255
    line_width : int        — pixel thickness of edges (if renderer supports it)
    pickable   : bool       — include in picking tests
    """

    # -------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------

    def __init__(
        self,
        vertices: np.ndarray,   # shape (N, 3), float32
        edges:    np.ndarray,   # shape (M, 2), int32  — index pairs
        name: str = "Object",
    ) -> None:
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N,3), got {vertices.shape}")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must be (M,2), got {edges.shape}")

        self.name       = name
        self.visible    = True
        self.color:     tuple[int, int, int] = (255, 255, 255)
        self.line_width = 1
        self.pickable   = True

        # Geometry (world-space vertices are computed by the renderer, not here)
        self._vertices: np.ndarray = vertices.astype(np.float32, copy=False)
        self._edges:    np.ndarray = edges.astype(np.int32,   copy=False)

        # Bounding sphere in local space: (centre_x,y,z, radius)
        self._bsphere: Optional[np.ndarray] = None   # lazily computed

        # Transform and scene graph
        self.transform: Transform   = Transform()
        self._parent:   Optional["WireframeObject"] = None
        self._children: list["WireframeObject"]     = []

        # Optional metadata bag — anything a user wants to store per-object
        self.user_data: dict = {}

    # -------------------------------------------------------------------
    # Factory helpers — common primitives
    # -------------------------------------------------------------------

    @classmethod
    def from_box(
        cls,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        name: str = "Box",
    ) -> "WireframeObject":
        """Axis-aligned box centred at the origin."""
        w, h, d = width / 2, height / 2, depth / 2
        verts = np.array([
            [-w, -h, -d], [ w, -h, -d], [ w,  h, -d], [-w,  h, -d],
            [-w, -h,  d], [ w, -h,  d], [ w,  h,  d], [-w,  h,  d],
        ], dtype=np.float32)
        edges = np.array([
            [0,1],[1,2],[2,3],[3,0],   # back face
            [4,5],[5,6],[6,7],[7,4],   # front face
            [0,4],[1,5],[2,6],[3,7],   # connecting edges
        ], dtype=np.int32)
        return cls(verts, edges, name=name)

    @classmethod
    def from_axes(cls, length: float = 1.0, name: str = "Axes") -> "WireframeObject":
        """Three-axis gizmo (X=red, Y=green, Z=blue via color assignment)."""
        verts = np.array([
            [0, 0, 0], [length, 0, 0],
            [0, 0, 0], [0, length, 0],
            [0, 0, 0], [0, 0, length],
        ], dtype=np.float32)
        edges = np.array([[0,1],[2,3],[4,5]], dtype=np.int32)
        return cls(verts, edges, name=name)

    @classmethod
    def from_grid(
        cls,
        size: float = 10.0,
        divisions: int = 10,
        name: str = "Grid",
    ) -> "WireframeObject":
        """Flat XZ grid centred at origin."""
        step   = size / divisions
        half   = size / 2
        verts, edges = [], []
        idx = 0
        for i in range(divisions + 1):
            x = -half + i * step
            verts += [[ x, 0, -half], [ x, 0,  half]]
            edges.append([idx, idx + 1]); idx += 2
            verts += [[-half, 0, x],   [ half, 0, x]]
            edges.append([idx, idx + 1]); idx += 2
        return cls(
            np.array(verts, dtype=np.float32),
            np.array(edges, dtype=np.int32),
            name=name,
        )

    @classmethod
    def from_sphere_approx(
        cls,
        radius: float = 1.0,
        lat_lines: int = 8,
        lon_lines: int = 12,
        name: str = "Sphere",
    ) -> "WireframeObject":
        """UV sphere wireframe (latitude + longitude rings)."""
        verts, edges = [], []
        idx = 0
        # Latitude rings
        for i in range(1, lat_lines):
            phi   = np.pi * i / lat_lines
            ring_start = idx
            for j in range(lon_lines):
                theta = 2 * np.pi * j / lon_lines
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.cos(phi)
                z = radius * np.sin(phi) * np.sin(theta)
                verts.append([x, y, z])
                next_j = ring_start + (j + 1) % lon_lines
                edges.append([idx, next_j])
                idx += 1
        # Longitude lines (connect matching vertices on adjacent rings)
        n_rings = lat_lines - 1
        for j in range(lon_lines):
            for i in range(n_rings - 1):
                edges.append([i * lon_lines + j, (i + 1) * lon_lines + j])
        return cls(
            np.array(verts, dtype=np.float32),
            np.array(edges, dtype=np.int32),
            name=name,
        )

    # -------------------------------------------------------------------
    # Scene-graph parenting
    # -------------------------------------------------------------------

    def set_parent(self, parent: Optional["WireframeObject"]) -> None:
        """
        Reparent this object.  Pass None to detach from current parent.
        Transforms the local position so the world position stays the same.
        """
        if self._parent is parent:
            return
        # Detach from old parent
        if self._parent is not None:
            self._parent._children.remove(self)
            self._parent.transform._remove_child(self.transform)

        self._parent = parent

        if parent is not None:
            parent._children.append(self)
            parent.transform._add_child(self.transform)

    @property
    def parent(self) -> Optional["WireframeObject"]:
        return self._parent

    @property
    def children(self) -> list["WireframeObject"]:
        return list(self._children)

    # -------------------------------------------------------------------
    # Geometry accessors
    # -------------------------------------------------------------------

    @property
    def vertices(self) -> np.ndarray:
        """Local-space vertices, shape (N, 3)."""
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray) -> None:
        self._vertices = np.asarray(value, dtype=np.float32)
        self._bsphere  = None   # invalidate cached bounding sphere
        self.transform._mark_dirty()

    @property
    def edges(self) -> np.ndarray:
        """Edge index pairs, shape (M, 2)."""
        return self._edges

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # -------------------------------------------------------------------
    # Bounding sphere (local space) — cached
    # -------------------------------------------------------------------

    @property
    def local_bounding_sphere(self) -> tuple[np.ndarray, float]:
        """Returns (centre_vec3, radius) in local space."""
        if self._bsphere is None:
            self._bsphere = self._compute_bsphere()
        c = self._bsphere[:3]
        r = float(self._bsphere[3])
        return c, r

    def _compute_bsphere(self) -> np.ndarray:
        centre = self._vertices.mean(axis=0)
        radius = float(np.linalg.norm(self._vertices - centre, axis=1).max())
        return np.array([*centre, radius], dtype=np.float32)

    # -------------------------------------------------------------------
    # Scene-graph traversal utilities
    # -------------------------------------------------------------------

    def iter_subtree(self) -> Iterator["WireframeObject"]:
        """Depth-first iteration over self and all descendants."""
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node._children))

    # -------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WireframeObject(name={self.name!r}, "
            f"verts={self.vertex_count}, edges={self.edge_count}, "
            f"visible={self.visible})"
        )
