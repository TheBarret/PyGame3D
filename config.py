from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """All tuneable constants for the demo application."""

    # ── Window ──────────────────────────────────────────────────────
    window_width:   int   = 920
    window_height:  int   = 720
    window_title:   str   = "Wireframe"
    target_fps:     int   = 60
    resizable:      bool  = True

    # ── Camera ──────────────────────────────────────────────────────
    camera_fov_y:   float = 60.0          # degrees
    camera_near:    float = 0.1
    camera_far:     float = 500.0
    camera_spawn:   tuple = (0.0, 6.0, -12.0)
    camera_target:  tuple = (0.0, 0.0,   0.0)

    # ── Renderer ────────────────────────────────────────────────────
    renderer_antialiased: bool  = True
    renderer_depth_sort:  bool  = True

    # ── Scene ───────────────────────────────────────────────────────
    scene_background: tuple = (1, 1, 1)

    # ── Input ───────────────────────────────────────────────────────
    input_orbit_sensitivity:  float = 0.15   # degrees per pixel
    input_pan_sensitivity:    float = 0.02   # world-units per pixel
    input_dolly_sensitivity:  float = 0.05
    input_zoom_sensitivity:   float = 0.5
    input_smoothing:          float = 0.18   # 0=instant  ~0.2=floaty  ~0.4=heavy

    # ── HUD ─────────────────────────────────────────────────────────
    hud_font_name:  str   = "monospace"
    hud_font_size:  int   = 12
    hud_color:      tuple = (180, 220, 180)
    hud_margin:     int   = 8
    hud_line_height: int  = 16

    # ── Picking ─────────────────────────────────────────────────────
    pick_max_distance: float = 0.15       # world-space threshold
