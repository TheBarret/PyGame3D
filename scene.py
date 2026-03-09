"""
Scene — owns the object list, drives culling, picking, and render dispatch.

The Scene is the single point of contact between user game-loop code and the
rendering pipeline.  A typical PyGame main loop looks like:

    scene.camera.orbit(d_yaw, d_pitch)
    scene.render(surface)

    # On click:
    hit = scene.pick(mouse_x, mouse_y)

Architecture
------------
Scene  ──owns──►  Camera
       ──owns──►  list[WireframeObject]   (flat, order = draw order)
       ──uses──►  Renderer                (injected; defaults to WireframeRenderer)
       ──uses──►  Picker                  (stateless helper, created lazily)

Object management
-----------------
scene.add(obj)     — adds to scene (no parent change)
scene.remove(obj)  — removes; children stay in scene unless also removed
scene.clear()      — wipe everything

Render pipeline per frame
--------------------------
1. Collect visible objects
2. Frustum-cull against camera planes (sphere test)
3. Per surviving object: transform vertices → clip-space
4. Clip edges (homogeneous Liang-Barsky)
5. Map to screen (NDC → PyGame pixels, Y-flip)
6. Draw to surface

Picking pipeline (on demand)
-----------------------------
1. Build ray from pixel
2. For each pickable object: two-phase test (bsphere first, then edges)
3. Return closest hit or None
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np

from camera import Camera
from wireframe_object import WireframeObject


if TYPE_CHECKING:
    import pygame


class PickResult:
    """Returned by Scene.pick(); holds the hit object and hit distance."""
    __slots__ = ("object", "distance", "ray_t")

    def __init__(self, obj: WireframeObject, distance: float, ray_t: float) -> None:
        self.object   = obj
        self.distance = distance   # world-space closest approach distance
        self.ray_t    = ray_t      # parameter along the ray at closest point

    def __repr__(self) -> str:
        return f"PickResult(object={self.object.name!r}, dist={self.distance:.3f})"


class Scene:
    """
    Container for all renderable objects plus the active camera.

    Parameters
    ----------
    camera : Camera | None
        If None, a default Camera is created (fov=60, 16:9 aspect).
    renderer : object | None
        Renderer instance.  If None, WireframeRenderer is used with
        default settings.  Inject your own to customise drawing.
    background_color : tuple[int,int,int]
        RGB colour used to fill the surface before each frame.
    """

    def __init__(
        self,
        camera: Optional[Camera] = None,
        renderer=None,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.camera           = camera or Camera()
        self.background_color = background_color

        # Flat list preserving insertion (draw) order.
        self._objects: list[WireframeObject] = []

        # Injected renderer — import default lazily to avoid circular deps.
        if renderer is None:
            from ..rendering.renderer import WireframeRenderer
            renderer = WireframeRenderer()
        self._renderer = renderer

        # Stats updated each render call (optional HUD use)
        self.stats: dict = {
            "objects_total":    0,
            "objects_culled":   0,
            "objects_drawn":    0,
            "edges_clipped":    0,
            "edges_drawn":      0,
            "frame_ms":         0.0,
        }

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def add(self, *objects: WireframeObject) -> "Scene":
        """Add one or more objects.  Returns self for chaining."""
        for obj in objects:
            if obj not in self._objects:
                self._objects.append(obj)
        return self

    def remove(self, *objects: WireframeObject) -> "Scene":
        """Remove objects (does not change parent/child relationships)."""
        for obj in objects:
            if obj in self._objects:
                self._objects.remove(obj)
        return self

    def clear(self) -> "Scene":
        """Remove all objects."""
        self._objects.clear()
        return self

    def get(self, name: str) -> Optional[WireframeObject]:
        """Return first object with the given name, or None."""
        for obj in self._objects:
            if obj.name == name:
                return obj
        return None

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self):
        return iter(self._objects)

    # ------------------------------------------------------------------
    # Window resize
    # ------------------------------------------------------------------

    def on_resize(self, width: int, height: int) -> None:
        """Call whenever the PyGame window is resized."""
        self.camera.set_aspect_from_size(width, height)
        self._renderer.on_resize(width, height)

    # ------------------------------------------------------------------
    # Main render entry-point
    # ------------------------------------------------------------------

    def render(self, surface: "pygame.Surface") -> None:
        """
        Draw all visible, non-culled objects to *surface*.

        Call once per game-loop iteration after updating transforms.
        """
        import time
        t0 = time.perf_counter()

        # 1. Clear background
        surface.fill(self.background_color)

        # 2. Snapshot camera matrices (one VP matrix for the frame)
        vp          = self.camera.view_projection_matrix
        frustum     = self.camera.frustum_planes
        screen_size = surface.get_size()

        # 3. Collect + cull
        visible  = [o for o in self._objects if o.visible]
        drawable = self._cull(visible, frustum)

        # 4. Dispatch each surviving object to the renderer
        edges_drawn   = 0
        edges_clipped = 0
        for obj in drawable:
            ed, ec = self._renderer.draw_object(surface, obj, vp, screen_size)
            edges_drawn   += ed
            edges_clipped += ec

        # 5. Update stats
        t1 = time.perf_counter()
        self.stats.update(
            objects_total   = len(self._objects),
            objects_culled  = len(visible) - len(drawable),
            objects_drawn   = len(drawable),
            edges_drawn     = edges_drawn,
            edges_clipped   = edges_clipped,
            frame_ms        = (t1 - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def pick(
        self,
        screen_x: int,
        screen_y: int,
        screen_w: int | None = None,
        screen_h: int | None = None,
        max_distance: float = 0.1,
    ) -> Optional[PickResult]:
        """
        Return the closest pickable object under the cursor, or None.

        Parameters
        ----------
        screen_x, screen_y : pixel coordinates (PyGame convention, Y-down).
        screen_w, screen_h : surface dimensions; inferred from renderer if None.
        max_distance       : world-space threshold for edge proximity.
        """
        w = screen_w or self._renderer.screen_width
        h = screen_h or self._renderer.screen_height

        ray_origin, ray_dir = self.camera.screen_to_ray(screen_x, screen_y, w, h)

        candidates = [o for o in self._objects if o.visible and o.pickable]
        return self._pick_closest(candidates, ray_origin, ray_dir, max_distance)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cull(
        self,
        objects: list[WireframeObject],
        frustum_planes: np.ndarray,
    ) -> list[WireframeObject]:
        """
        Sphere-based frustum culling.
        Objects that pass the cull test are returned in original order.
        """
        from math_core import sphere_in_frustum  # type: ignore[import]

        surviving = []
        for obj in objects:
            centre_local, radius_local = obj.local_bounding_sphere
            # Row-major: v_world = v_local @ world_matrix
            wm     = obj.transform.world_matrix
            c4     = np.array([*centre_local, 1.0], dtype=np.float32) @ wm
            centre_world = c4[:3]
            # In row-major, scale is encoded in the ROWS (not columns)
            sx, sy, sz = (
                np.linalg.norm(wm[0, :3]),
                np.linalg.norm(wm[1, :3]),
                np.linalg.norm(wm[2, :3]),
            )
            world_radius = radius_local * max(sx, sy, sz)

            if sphere_in_frustum(frustum_planes, centre_world, world_radius):
                surviving.append(obj)
        return surviving

    def _pick_closest(
        self,
        candidates: list[WireframeObject],
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        max_dist: float,
    ) -> Optional[PickResult]:
        """Two-phase pick: bsphere early-out, then per-edge test."""
        from math_core import (    # type: ignore[import]
            ray_sphere_intersect,
            ray_segment_distance,
        )

        best: Optional[PickResult] = None

        for obj in candidates:
            wm = obj.transform.world_matrix

            # Phase 1 — bounding sphere
            centre_local, radius_local = obj.local_bounding_sphere
            # Row-major: v_world = v_local @ world_matrix
            c4 = np.array([*centre_local, 1.0], dtype=np.float32) @ wm
            centre_world  = c4[:3]
            world_radius  = radius_local * max(
                np.linalg.norm(wm[0, :3]),
                np.linalg.norm(wm[1, :3]),
                np.linalg.norm(wm[2, :3]),
            )
            if not ray_sphere_intersect(ray_origin, ray_dir,
                                        centre_world, world_radius + max_dist):
                continue

            # Phase 2 — per-edge
            # Row-major batch: verts_world = verts_h @ world_matrix
            verts_h = np.ones((obj.vertex_count, 4), dtype=np.float32)
            verts_h[:, :3] = obj.vertices
            world_verts = (verts_h @ wm)[:, :3]

            for e in obj.edges:
                a, b = world_verts[e[0]], world_verts[e[1]]
                dist, t_ray = ray_segment_distance(ray_origin, ray_dir, a, b)
                if dist < max_dist:
                    if best is None or dist < best.distance:
                        best = PickResult(obj, dist, t_ray)

        return best