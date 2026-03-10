"""
Renderer — per-object draw pipeline.

Responsibilities (in order)
----------------------------
1. Homogeneous transform: local → clip space  (vertices × VP)
2. Per-edge clipping: homogeneous Liang-Barsky  (Qwen's clipper)
3. Perspective divide: clip → NDC
4. Screen mapping: NDC → PyGame pixels with Y-flip  (Qwen's screen_mapper)
5. pygame.draw.line / aaline for each surviving edge

The renderer is stateless with respect to scene data; it only stores
screen-size and rendering preferences so the Scene can call draw_object()
repeatedly without re-injecting configuration every call.

"""

from __future__ import annotations
from typing import Optional
import numpy as np


class WireframeRenderer:
    """
    Draw wireframe objects one at a time onto a pygame.Surface.

    Parameters
    ----------
    antialiased : bool
        Use pygame.draw.aaline instead of draw.line.  Slower but smoother.
    depth_sort : bool
        Sort edges back-to-front before drawing (painter's algorithm).
    """

    def __init__(
        self,
        screen_width:  int = 800,
        screen_height: int = 600,
        antialiased:   bool = False,
        depth_sort:    bool = False,
    ) -> None:
        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.antialiased   = antialiased
        self.depth_sort    = depth_sort

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_resize(self, width: int, height: int) -> None:
        self.screen_width  = width
        self.screen_height = height

    # ------------------------------------------------------------------
    # Main draw method  (called once per object per frame by Scene)
    # ------------------------------------------------------------------

    def draw_object(
        self,
        surface,               # pygame.Surface
        obj,                   # WireframeObject
        view_proj: np.ndarray, # 4×4 combined VP matrix
        screen_size: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Project, clip, and draw *obj* onto *surface*.

        Returns
        -------
        (edges_drawn, edges_clipped) : int, int
        """
        import pygame
        from math_core import (      # type: ignore[import]
            clip_edge_homogeneous,
            ndc_to_screen,
        )

        w, h = screen_size

        # ---- Step 1: Homogeneous transform ----------------------------
        # Row-major convention: v_clip = v @ World @ VP = v @ MVP
        # MVP = World @ ViewProj  (chain left-to-right)
        world_matrix = obj.transform.world_matrix
        mvp          = world_matrix @ view_proj              # row-major: W @ VP

        verts_h = np.ones((obj.vertex_count, 4), dtype=np.float32)
        verts_h[:, :3] = obj.vertices
        clip_verts = verts_h @ mvp                           # (N,4) @ (4,4) = (N,4)

        # ---- Step 2–5: Per-edge clip + project + draw -----------------
        edges_drawn   = 0
        edges_clipped = 0

        color = obj.color

        # Optional: collect edges for depth sorting
        draw_list: list[tuple[float, tuple, tuple, tuple]] = []  # (z, pA, pB, col)

        for edge in obj.edges:
            a_clip = clip_verts[edge[0]]
            b_clip = clip_verts[edge[1]]

            # Clip in homogeneous space
            clipped = clip_edge_homogeneous(a_clip, b_clip)
            if clipped is None:
                edges_clipped += 1
                continue

            a_clip_c, b_clip_c = clipped   # clipped clip-space endpoints

            # Perspective divide → NDC
            wa, wb = a_clip_c[3], b_clip_c[3]
            if abs(wa) < 1e-7 or abs(wb) < 1e-7:
                edges_clipped += 1
                continue

            ndc_a = a_clip_c[:3] / wa
            ndc_b = b_clip_c[:3] / wb

            # Screen mapping (Y-flip happens inside ndc_to_screen)
            sx_a, sy_a = ndc_to_screen(ndc_a, w, h)
            sx_b, sy_b = ndc_to_screen(ndc_b, w, h)
            edge_color = color

            pA = (int(sx_a), int(sy_a))
            pB = (int(sx_b), int(sy_b))

            if self.depth_sort:
                mid_z = float((ndc_a[2] + ndc_b[2]) * 0.5)
                draw_list.append((mid_z, pA, pB, edge_color))
            else:
                self._draw_line(surface, pA, pB, edge_color, obj.line_width)
                edges_drawn += 1

        # Depth-sorted pass (far-to-near)
        if self.depth_sort and draw_list:
            draw_list.sort(key=lambda x: -x[0])   # descending Z = far first
            for _z, pA, pB, col in draw_list:
                self._draw_line(surface, pA, pB, col, obj.line_width)
                edges_drawn += 1

        return edges_drawn, edges_clipped

    # ------------------------------------------------------------------
    # Internal draw primitive
    # ------------------------------------------------------------------

    def _draw_line(
        self,
        surface,
        pA: tuple[int, int],
        pB: tuple[int, int],
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        import pygame
        if self.antialiased and width == 1:
            pygame.draw.aaline(surface, color, pA, pB)
        else:
            pygame.draw.line(surface, color, pA, pB, width)