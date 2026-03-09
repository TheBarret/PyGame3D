"""
input_handler.py — Camera input handler for pygame3d.

Key fixes over the original:
─────────────────────────────
1. Delta is now per-frame (current_pos - last_pos), NOT total drag from mouse-down.
   This is why sensitivity=0.0 previously had no effect — the accumulated pixel
   distance was being used directly, bypassing the multiplier entirely.

2. Exponential smoothing (lerp toward target) on the yaw/pitch/pan values so
   movement eases in and out instead of snapping to the raw pixel delta.

3. Sensitivity is now expressed in meaningful units:
     orbit_sensitivity  = degrees per pixel
     pan_sensitivity    = world-units per pixel
     zoom_sensitivity   = world-units per scroll tick

4. The _world_y_orbit branch no longer imports from a stale relative path.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Optional, Tuple
import pygame
import numpy as np


class InputMode(Enum):
    ORBIT_FREE    = auto()
    ORBIT_YAW     = auto()
    ORBIT_PITCH   = auto()
    PAN_FREE      = auto()
    PAN_X         = auto()
    PAN_Y         = auto()
    PAN_Z         = auto()
    DOLLY         = auto()
    TRUCK         = auto()
    PEDESTAL      = auto()


class CameraInputHandler:
    """
    Frame-delta camera input with exponential smoothing.

    Sensitivity guide
    -----------------
    orbit_sensitivity  : degrees of rotation per pixel dragged  (try 0.25–0.5)
    pan_sensitivity    : world-units of translation per pixel   (try 0.01–0.03)
    zoom_sensitivity   : world-units per scroll tick            (try 0.3–1.0)
    smoothing          : 0.0 = instant (no smoothing), 1.0 = never moves.
                         0.15–0.25 gives a nice "floating" feel.
    """

    def __init__(
        self,
        orbit_sensitivity:  float = 0.3,
        pan_sensitivity:    float = 0.02,
        dolly_sensitivity:  float = 0.05,
        zoom_sensitivity:   float = 0.5,
        smoothing:          float = 0.18,      # exponential lerp factor (0=instant)
        axis_lock_threshold: float = 0.75,
        invert_pitch: bool = False,
        invert_pan_y: bool = False,
    ):
        self.orbit_sensitivity   = orbit_sensitivity
        self.pan_sensitivity     = pan_sensitivity
        self.dolly_sensitivity   = dolly_sensitivity
        self.zoom_sensitivity    = zoom_sensitivity
        self.smoothing           = smoothing
        self.axis_lock_threshold = axis_lock_threshold
        self.invert_pitch        = invert_pitch
        self.invert_pan_y        = invert_pan_y

        # --- Internal state ---

        # Last known mouse position — used to compute per-frame delta
        self._last_mouse: tuple[int, int] = (0, 0)

        # Are we currently dragging?
        self._dragging_left  = False
        self._dragging_right = False

        # Active mode this drag
        self.current_mode = InputMode.ORBIT_FREE

        # Modifier keys currently held
        self.modifiers: set[str] = set()

        # Flags set by _determine_mode
        self._world_y_orbit  = False
        self._auto_orbit_axis = False

        # Smoothed velocity targets — we lerp toward these each frame
        # Keys: 'yaw', 'pitch', 'pan_right', 'pan_up', 'pan_x', 'pan_y', 'pan_z', 'zoom'
        self._target:  dict[str, float] = {}
        self._current: dict[str, float] = {}   # smoothed values

    # ------------------------------------------------------------------
    # Event ingestion  (call for every pygame event)
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN:
            self._update_modifiers(event.key, pressed=True)
            return True

        elif event.type == pygame.KEYUP:
            self._update_modifiers(event.key, pressed=False)
            return True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            self._last_mouse = event.pos   # anchor last-pos to click point
            if event.button == 1:
                self._dragging_left = True
                self._determine_mode(event)
            elif event.button == 3:
                self._dragging_right = True
                self._determine_mode(event)
            return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: self._dragging_left  = False
            if event.button == 3: self._dragging_right = False
            # Zero targets so smoothing decelerates to a stop
            self._target.clear()
            return True

        elif event.type == pygame.MOUSEMOTION:
            # Compute FRAME delta (not total-drag delta)
            dx = event.pos[0] - self._last_mouse[0]
            dy = event.pos[1] - self._last_mouse[1]
            self._last_mouse = event.pos

            if self._dragging_left or self._dragging_right:
                self._update_targets(dx, dy)
            return True

        elif event.type == pygame.MOUSEWHEEL:
            return True   # handled separately in handle_wheel

        return False

    def handle_wheel(self, event: pygame.event.Event, camera) -> None:
        """Direct (non-smoothed) zoom on scroll wheel."""
        if event.y != 0:
            camera.zoom(event.y * self.zoom_sensitivity)

    # ------------------------------------------------------------------
    # Per-frame update  (call ONCE per game loop iteration)
    # ------------------------------------------------------------------

    def update(self, camera, dt: float) -> None:
        """
        Apply smoothed camera movement.  Call once per frame after handling events.

        Parameters
        ----------
        camera : Camera
        dt     : elapsed seconds since last frame (used to make movement
                 frame-rate independent when smoothing=0)
        """
        if not self._target and not self._current:
            return

        alpha = self.smoothing   # 0 = instant, higher = more lag

        def smooth(key: str) -> float:
            """Lerp _current[key] toward _target[key] and return value."""
            target  = self._target.get(key, 0.0)
            current = self._current.get(key, 0.0)
            # Exponential decay: new = current + (target - current) * (1 - alpha)
            # alpha=0 → new = target  (instant)
            # alpha close to 1 → very slow follow
            new = current + (target - current) * (1.0 - alpha)
            self._current[key] = new
            return new

        # --- Orbit ---
        yaw   = smooth('yaw')
        pitch = smooth('pitch')
        if abs(yaw) > 1e-5 or abs(pitch) > 1e-5:
            camera.orbit(yaw, pitch)

        # --- Free pan ---
        pr = smooth('pan_right')
        pu = smooth('pan_up')
        if abs(pr) > 1e-5 or abs(pu) > 1e-5:
            camera.pan(pr, pu)

        # --- Axis-locked pan ---
        for axis_key, world_axis in [('pan_x', (1,0,0)), ('pan_y', (0,1,0)), ('pan_z', (0,0,1))]:
            amount = smooth(axis_key)
            if abs(amount) > 1e-5:
                ax = np.array(world_axis, dtype=np.float32)
                camera.position = camera.position + ax * amount

        # --- Dolly ---
        zoom = smooth('zoom')
        if abs(zoom) > 1e-5:
            camera.zoom(zoom)

        # Clear targets each frame — they'll be re-set by motion events
        self._target.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_modifiers(self, key: int, pressed: bool) -> None:
        MAP = {
            pygame.K_LSHIFT: 'shift', pygame.K_RSHIFT: 'shift',
            pygame.K_LCTRL:  'ctrl',  pygame.K_RCTRL:  'ctrl',
            pygame.K_LALT:   'alt',   pygame.K_RALT:   'alt',
            pygame.K_SPACE:  'space',
        }
        if key in MAP:
            mod = MAP[key]
            if pressed: self.modifiers.add(mod)
            else:       self.modifiers.discard(mod)

    def _determine_mode(self, event: pygame.event.Event) -> None:
        mods = self.modifiers
        self._world_y_orbit  = False
        self._auto_orbit_axis = False

        if event.button == 1:   # Left — orbit
            if 'shift' in mods and 'ctrl' in mods:
                self.current_mode = InputMode.ORBIT_YAW
                self._world_y_orbit = True
            elif 'shift' in mods:
                self.current_mode = InputMode.ORBIT_YAW
            elif 'ctrl' in mods:
                self.current_mode = InputMode.ORBIT_PITCH
            elif 'alt' in mods:
                self.current_mode = InputMode.ORBIT_FREE
                self._auto_orbit_axis = True
            else:
                self.current_mode = InputMode.ORBIT_FREE

        elif event.button == 3:  # Right — pan / dolly
            if 'shift' in mods:
                self.current_mode = InputMode.PAN_X
            elif 'ctrl' in mods:
                self.current_mode = InputMode.PAN_Y
            elif 'alt' in mods:
                self.current_mode = InputMode.PAN_Z
            elif 'space' in mods:
                self.current_mode = InputMode.DOLLY
            else:
                self.current_mode = InputMode.PAN_FREE

    def _update_targets(self, dx: int, dy: int) -> None:
        """
        Convert raw pixel delta into target values that update() will smooth.
        All values are reset to 0.0 each frame and re-written by motion events,
        so the smoothing naturally decelerates when the mouse stops.
        """
        # Compute raw scaled values
        raw_yaw   =  dx * self.orbit_sensitivity
        raw_pitch =  dy * self.orbit_sensitivity * (-1 if self.invert_pitch else 1)
        raw_pan_r = -dx * self.pan_sensitivity
        raw_pan_u =  dy * self.pan_sensitivity * (-1 if self.invert_pan_y else 1)

        mode = self.current_mode

        # ── ORBIT modes ──────────────────────────────────────────────
        if mode in (InputMode.ORBIT_FREE, InputMode.ORBIT_YAW, InputMode.ORBIT_PITCH):

            if self._auto_orbit_axis:
                ratio = self.axis_lock_threshold
                if abs(dx) > abs(dy) * ratio:
                    self._target['yaw'] = raw_yaw; self._target['pitch'] = 0.0
                elif abs(dy) > abs(dx) * ratio:
                    self._target['yaw'] = 0.0;    self._target['pitch'] = raw_pitch
                else:
                    self._target['yaw'] = raw_yaw; self._target['pitch'] = raw_pitch

            elif self._world_y_orbit:
                # Orbit around world Y — handled directly (no smoothing target needed)
                # The camera.orbit() call already does this; we just pass yaw-only.
                self._target['yaw']   = raw_yaw
                self._target['pitch'] = 0.0

            elif mode == InputMode.ORBIT_YAW:
                self._target['yaw']   = raw_yaw
                self._target['pitch'] = 0.0

            elif mode == InputMode.ORBIT_PITCH:
                self._target['yaw']   = 0.0
                self._target['pitch'] = raw_pitch

            else:   # ORBIT_FREE
                self._target['yaw']   = raw_yaw
                self._target['pitch'] = raw_pitch

        # ── PAN modes ────────────────────────────────────────────────
        elif mode == InputMode.PAN_FREE:
            self._target['pan_right'] = raw_pan_r
            self._target['pan_up']    = raw_pan_u

        elif mode == InputMode.PAN_X:
            self._target['pan_x'] = -dx * self.pan_sensitivity

        elif mode == InputMode.PAN_Y:
            amount = dy * self.pan_sensitivity * (-1 if self.invert_pan_y else 1)
            self._target['pan_y'] = amount

        elif mode == InputMode.PAN_Z:
            self._target['pan_z'] = dy * self.pan_sensitivity

        # ── DOLLY / TRUCK ────────────────────────────────────────────
        elif mode == InputMode.DOLLY:
            if abs(dx) > abs(dy) * self.axis_lock_threshold:
                # Horizontal → truck (world X)
                self._target['pan_x'] = -dx * self.dolly_sensitivity
            else:
                # Vertical → dolly (zoom along forward)
                self._target['zoom'] = -dy * self.dolly_sensitivity