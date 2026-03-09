"""
Constructs and wires all engine objects from an AppConfig.

The factory has one job: read config, build objects, return a ready-to-use
bundle.  No game logic, no event handling, no animation — just construction.

Usage
-----
    from config       import AppConfig
    from scene_factory import SceneFactory

    cfg     = AppConfig()
    bundle  = SceneFactory.create(cfg)

    # Everything you need is on the bundle:
    bundle.scene      # pygame3d Scene
    bundle.camera     # Camera  (also reachable as bundle.scene.camera)
    bundle.renderer   # WireframeRenderer
    bundle.handler    # CameraInputHandler

    # Add your own objects before the loop:
    bundle.scene.add(my_object)

    # Or use the fluent builder if you prefer:
    bundle = (
        SceneFactory.builder(cfg)
            .add_grid(size=20, divisions=20, color=(40, 40, 60))
            .add_object(my_cube)
            .build()
    )
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pygame

from config import AppConfig
from scene import Scene
from camera import Camera
from renderer import WireframeRenderer
from wireframe_object import WireframeObject
from input_handler import CameraInputHandler


# ---------------------------------------------------------------------------
# AppBundle — the assembled result
# ---------------------------------------------------------------------------

@dataclass
class AppBundle:
    """
    Everything main() needs, assembled and ready to run.
    All fields are fully constructed — no further init required.
    """
    cfg:      AppConfig
    scene:    Scene
    camera:   Camera
    renderer: WireframeRenderer
    handler:  CameraInputHandler
    screen:   pygame.Surface
    clock:    pygame.time.Clock
    font:     pygame.font.Font

    # Convenience passthrough so callers don't need to import Camera
    @property
    def cam(self) -> Camera:
        return self.camera

    def reset_camera(self) -> None:
        """Restore camera to its configured spawn position."""
        self.camera.position = self.cfg.camera_spawn
        self.camera.look_at(self.cfg.camera_target)

    def toggle_antialiasing(self) -> None:
        self.renderer.antialiased = not self.renderer.antialiased

    def on_resize(self, width: int, height: int) -> None:
        self.scene.on_resize(width, height)


# ---------------------------------------------------------------------------
# SceneFactory
# ---------------------------------------------------------------------------

class SceneFactory:
    """
    Static factory.  Call SceneFactory.create(cfg) for the common case.
    Call SceneFactory.builder(cfg) for a fluent object-population API.
    """

    @staticmethod
    def create(cfg: Optional[AppConfig] = None) -> AppBundle:
        """
        Build a complete AppBundle from config.
        Calls pygame.init() internally — do not call it beforehand.
        """
        if cfg is None:
            cfg = AppConfig()

        # ── pygame bootstrap ────────────────────────────────────────
        pygame.init()
        flags  = pygame.RESIZABLE if cfg.resizable else 0
        screen = pygame.display.set_mode((cfg.window_width, cfg.window_height), flags)
        pygame.display.set_caption(cfg.window_title)
        clock  = pygame.time.Clock()
        font   = pygame.font.SysFont(cfg.hud_font_name, cfg.hud_font_size)

        # ── Engine objects ───────────────────────────────────────────
        camera = SceneFactory._make_camera(cfg)
        renderer = SceneFactory._make_renderer(cfg)
        scene    = SceneFactory._make_scene(cfg, camera, renderer)
        handler  = SceneFactory._make_handler(cfg)

        return AppBundle(
            cfg      = cfg,
            scene    = scene,
            camera   = camera,
            renderer = renderer,
            handler  = handler,
            screen   = screen,
            clock    = clock,
            font     = font,
        )

    @staticmethod
    def builder(cfg: Optional[AppConfig] = None) -> "SceneBuilder":
        """Return a fluent builder for populating scene objects."""
        bundle = SceneFactory.create(cfg)
        return SceneBuilder(bundle)

    # ── Private constructors ─────────────────────────────────────────

    @staticmethod
    def _make_camera(cfg: AppConfig) -> Camera:
        cam = Camera(
            fov_y  = cfg.camera_fov_y,
            aspect = cfg.window_width / max(cfg.window_height, 1),
            near   = cfg.camera_near,
            far    = cfg.camera_far,
        )
        cam.position = cfg.camera_spawn
        cam.look_at(cfg.camera_target)
        return cam

    @staticmethod
    def _make_renderer(cfg: AppConfig) -> WireframeRenderer:
        return WireframeRenderer(
            screen_width  = cfg.window_width,
            screen_height = cfg.window_height,
            antialiased   = cfg.renderer_antialiased,
            depth_sort    = cfg.renderer_depth_sort,
        )

    @staticmethod
    def _make_scene(cfg: AppConfig, camera: Camera, renderer: WireframeRenderer) -> Scene:
        return Scene(
            camera           = camera,
            renderer         = renderer,
            background_color = cfg.scene_background,
        )

    @staticmethod
    def _make_handler(cfg: AppConfig) -> CameraInputHandler:
        return CameraInputHandler(
            orbit_sensitivity  = cfg.input_orbit_sensitivity,
            pan_sensitivity    = cfg.input_pan_sensitivity,
            dolly_sensitivity  = cfg.input_dolly_sensitivity,
            zoom_sensitivity   = cfg.input_zoom_sensitivity,
            smoothing          = cfg.input_smoothing,
        )


# ---------------------------------------------------------------------------
# SceneBuilder — fluent API for adding objects
# ---------------------------------------------------------------------------

class SceneBuilder:
    """
    Fluent builder returned by SceneFactory.builder().

    Example
    -------
        bundle = (
            SceneFactory.builder(cfg)
                .add_grid(size=20, divisions=20, color=(40,40,60), pickable=False)
                .add_box(2, 2, 2, name="Cube", color=(80,200,255), position=(0,1,0))
                .build()
        )
    """

    def __init__(self, bundle: AppBundle) -> None:
        self._bundle = bundle

    # ── Primitive helpers ─────────────────────────────────────────────

    def add_grid(
        self,
        size:       float = 10.0,
        divisions:  int   = 10,
        color:      tuple = (40, 40, 60),
        pickable:   bool  = False,
        name:       str   = "Grid",
    ) -> "SceneBuilder":
        obj = WireframeObject.from_grid(size=size, divisions=divisions, name=name)
        obj.color    = color
        obj.pickable = pickable
        self._bundle.scene.add(obj)
        return self

    def add_axes(
        self,
        length:   float = 3.0,
        color:    tuple = (255, 80, 80),
        pickable: bool  = False,
        name:     str   = "Axes",
    ) -> "SceneBuilder":
        obj = WireframeObject.from_axes(length=length, name=name)
        obj.color    = color
        obj.pickable = pickable
        self._bundle.scene.add(obj)
        return self

    def add_box(
        self,
        w: float = 1.0, h: float = 1.0, d: float = 1.0,
        name:       str           = "Box",
        color:      tuple         = (255, 255, 255),
        position:   tuple         = (0.0, 0.0, 0.0),
        line_width: int           = 1,
        parent:     Optional[WireframeObject] = None,
    ) -> "SceneBuilder":
        obj = WireframeObject.from_box(w, h, d, name=name)
        obj.color      = color
        obj.line_width = line_width
        obj.transform.position = position
        if parent is not None:
            obj.set_parent(parent)
        self._bundle.scene.add(obj)
        return self

    def add_sphere(
        self,
        radius:     float = 1.0,
        lat_lines:  int   = 8,
        lon_lines:  int   = 12,
        name:       str   = "Sphere",
        color:      tuple = (255, 255, 255),
        position:   tuple = (0.0, 0.0, 0.0),
        parent:     Optional[WireframeObject] = None,
    ) -> "SceneBuilder":
        obj = WireframeObject.from_sphere_approx(
            radius=radius, lat_lines=lat_lines, lon_lines=lon_lines, name=name
        )
        obj.color = color
        obj.transform.position = position
        if parent is not None:
            obj.set_parent(parent)
        self._bundle.scene.add(obj)
        return self

    def add_object(self, obj: WireframeObject) -> "SceneBuilder":
        """Add a fully pre-configured WireframeObject directly."""
        self._bundle.scene.add(obj)
        return self

    def build(self) -> AppBundle:
        """Finalise and return the bundle."""
        return self._bundle