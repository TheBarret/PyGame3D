from enum import Enum, auto
from typing import Optional, Tuple
import pygame
import numpy as np

class InputMode(Enum):
    """Different interaction modes for camera control."""
    ORBIT_FREE = auto()      # Free orbit (both axes)
    ORBIT_YAW = auto()       # Orbit constrained to yaw (horizontal)
    ORBIT_PITCH = auto()     # Orbit constrained to pitch (vertical)
    PAN_FREE = auto()        # Free pan in camera plane
    PAN_X = auto()           # Pan locked to world X axis
    PAN_Y = auto()           # Pan locked to world Y axis
    PAN_Z = auto()           # Pan locked to world Z axis
    DOLLY = auto()           # Dolly (move in/out)
    TRUCK = auto()           # Truck (move left/right)
    PEDESTAL = auto()        # Pedestal (move up/down)


class CameraInputHandler:
    """
    Handles camera input with modifier keys for precise axis control.
    
    Default behavior:
    - Left drag: Free orbit (both axes)
    - Right drag: Free pan (in camera plane)
    
    With modifiers for ORBIT (Left drag):
    - Shift + Left: Yaw only (horizontal orbit)
    - Ctrl + Left: Pitch only (vertical orbit)
    - Alt + Left: Auto-detect based on drag direction
    - Shift+Ctrl + Left: Orbit around world Y axis (maintains horizon)
    
    With modifiers for PAN (Right drag):
    - Shift + Right: Pan along world X axis
    - Ctrl + Right: Pan along world Y axis
    - Alt + Right: Pan along world Z axis
    - Space + Right: Auto-detect dolly vs truck
    
    Configurable sensitivity for each action.
    """
    
    def __init__(
        self,
        orbit_sensitivity: float = 0.4,
        pan_sensitivity: float = 0.02,
        dolly_sensitivity: float = 0.03,
        zoom_sensitivity: float = 0.5,
        axis_lock_threshold: float = 0.7,  # Ratio for automatic axis selection
        invert_pitch: bool = False,         # Invert vertical orbit
        invert_pan_y: bool = False,         # Invert vertical pan
    ):
        self.orbit_sensitivity = orbit_sensitivity
        self.pan_sensitivity = pan_sensitivity
        self.dolly_sensitivity = dolly_sensitivity
        self.zoom_sensitivity = zoom_sensitivity
        self.axis_lock_threshold = axis_lock_threshold
        self.invert_pitch = invert_pitch
        self.invert_pan_y = invert_pan_y
        
        # State
        self.mouse_pos = (0, 0)
        self.mouse_down_pos = (0, 0)
        self.mouse_down_time = 0
        self.drag_started = False
        self.current_mode = InputMode.ORBIT_FREE
        self.modifiers = set()
        self.last_delta = (0, 0)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Process input event. Returns True if event was handled.
        """
        if event.type == pygame.KEYDOWN:
            self._update_modifiers(event.key, pressed=True)
            return True
        elif event.type == pygame.KEYUP:
            self._update_modifiers(event.key, pressed=False)
            return True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_down_pos = event.pos
            self.mouse_down_time = pygame.time.get_ticks()
            self.drag_started = True
            self._determine_mode(event)
            return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.drag_started = False
            self.last_delta = (0, 0)
            return True
        elif event.type == pygame.MOUSEMOTION:
            self.mouse_pos = event.pos
            return True
        elif event.type == pygame.MOUSEWHEEL:
            return True
        return False
    
    def _update_modifiers(self, key: int, pressed: bool) -> None:
        """Track modifier key states."""
        modifiers_map = {
            pygame.K_LSHIFT: 'shift',
            pygame.K_RSHIFT: 'shift',
            pygame.K_LCTRL: 'ctrl',
            pygame.K_RCTRL: 'ctrl',
            pygame.K_LALT: 'alt',
            pygame.K_RALT: 'alt',
            pygame.K_SPACE: 'space',
        }
        if key in modifiers_map:
            mod = modifiers_map[key]
            if pressed:
                self.modifiers.add(mod)
            else:
                self.modifiers.discard(mod)
    
    def _determine_mode(self, event: pygame.event.Event) -> None:
        """Determine input mode based on mouse button and modifiers."""
        mods = self.modifiers
        
        if event.button == 1:  # Left button - ORBIT modes
            if 'shift' in mods and 'ctrl' in mods:
                # Shift+Ctrl = World Y axis orbit (maintains horizon)
                self.current_mode = InputMode.ORBIT_YAW  # Will treat specially
                self._world_y_orbit = True
            elif 'shift' in mods:
                self.current_mode = InputMode.ORBIT_YAW
                self._world_y_orbit = False
            elif 'ctrl' in mods:
                self.current_mode = InputMode.ORBIT_PITCH
            elif 'alt' in mods:
                self.current_mode = InputMode.ORBIT_FREE
                self._auto_orbit_axis = True  # Will determine based on drag
            else:
                self.current_mode = InputMode.ORBIT_FREE
                self._auto_orbit_axis = False
                
        elif event.button == 3:  # Right button - PAN/DOLLY modes
            if 'shift' in mods:
                self.current_mode = InputMode.PAN_X
            elif 'ctrl' in mods:
                self.current_mode = InputMode.PAN_Y
            elif 'alt' in mods:
                self.current_mode = InputMode.PAN_Z
            elif 'space' in mods:
                self.current_mode = InputMode.DOLLY  # Will determine dolly vs truck
            else:
                self.current_mode = InputMode.PAN_FREE
    
    def get_camera_delta(self, camera, current_pos: Tuple[int, int]) -> dict:
        """
        Calculate camera movement based on current mode and drag delta.
        Returns a dict with movement parameters.
        """
        if not self.drag_started:
            return {}
        
        dx = current_pos[0] - self.mouse_down_pos[0]
        dy = current_pos[1] - self.mouse_down_pos[1]
        
        # If no movement, return empty
        if abs(dx) < 1 and abs(dy) < 1:
            return {}
        
        # Apply inversion settings
        orbit_yaw = dx * self.orbit_sensitivity
        orbit_pitch = dy * self.orbit_sensitivity
        if self.invert_pitch:
            orbit_pitch = -orbit_pitch
        
        result = {}
        
        # ---- ORBIT MODES ----
        if self.current_mode in [InputMode.ORBIT_FREE, InputMode.ORBIT_YAW, InputMode.ORBIT_PITCH]:
            
            # Auto-detect axis based on drag direction (Alt modifier)
            if hasattr(self, '_auto_orbit_axis') and self._auto_orbit_axis:
                if abs(dx) > abs(dy) * self.axis_lock_threshold:
                    # Horizontal drag = yaw only
                    result['orbit'] = {
                        'yaw': orbit_yaw,
                        'pitch': 0
                    }
                elif abs(dy) > abs(dx) * self.axis_lock_threshold:
                    # Vertical drag = pitch only
                    result['orbit'] = {
                        'yaw': 0,
                        'pitch': orbit_pitch
                    }
                else:
                    # Diagonal = free orbit
                    result['orbit'] = {
                        'yaw': orbit_yaw,
                        'pitch': orbit_pitch
                    }
            
            # World Y-axis orbit (Shift+Ctrl) - maintains horizon
            elif hasattr(self, '_world_y_orbit') and self._world_y_orbit:
                # Get camera's current right vector to orbit around world Y
                import numpy as np
                from math_core import quaternion_rotate_vector
                
                # Calculate orbit around world Y axis
                q_yaw = self._quaternion_from_axis_angle((0, 1, 0), np.radians(orbit_yaw))
                
                # Get current position relative to look target (assuming look_at target is (0,0,0))
                # You might need to store the orbit target separately
                target = np.array([0, 0, 0], dtype=np.float32)
                pos = camera.position - target
                
                # Rotate position around world Y
                new_pos = quaternion_rotate_vector(q_yaw, pos)
                camera.position = new_pos + target
                
                # Update rotation to keep looking at target
                camera.look_at(target)
                return {}  # Return early since we handled specially
            
            # Standard constrained orbit
            elif self.current_mode == InputMode.ORBIT_YAW:
                result['orbit'] = {
                    'yaw': orbit_yaw,
                    'pitch': 0
                }
            elif self.current_mode == InputMode.ORBIT_PITCH:
                result['orbit'] = {
                    'yaw': 0,
                    'pitch': orbit_pitch
                }
            else:  # ORBIT_FREE
                result['orbit'] = {
                    'yaw': orbit_yaw,
                    'pitch': orbit_pitch
                }
        
        # ---- PAN MODES ----
        elif self.current_mode == InputMode.PAN_FREE:
            # Free pan in camera plane
            pan_right = -dx * self.pan_sensitivity
            pan_up = dy * self.pan_sensitivity
            if self.invert_pan_y:
                pan_up = -pan_up
            
            result['pan'] = {
                'right': pan_right,
                'up': pan_up
            }
            
        elif self.current_mode == InputMode.PAN_X:
            # Pan locked to world X axis
            result['pan_axis'] = {
                'axis': (1.0, 0.0, 0.0),
                'amount': -dx * self.pan_sensitivity * 10
            }
            
        elif self.current_mode == InputMode.PAN_Y:
            # Pan locked to world Y axis
            amount = dy * self.pan_sensitivity * 10
            if self.invert_pan_y:
                amount = -amount
            result['pan_axis'] = {
                'axis': (0.0, 1.0, 0.0),
                'amount': amount
            }
            
        elif self.current_mode == InputMode.PAN_Z:
            # Pan locked to world Z axis
            result['pan_axis'] = {
                'axis': (0.0, 0.0, 1.0),
                'amount': dy * self.pan_sensitivity * 10
            }
            
        elif self.current_mode == InputMode.DOLLY:
            # Determine if we're doing dolly (in/out) or truck (left/right)
            if abs(dx) > abs(dy) * self.axis_lock_threshold:
                # Horizontal drag = truck (left/right)
                result['pan_axis'] = {
                    'axis': (1.0, 0.0, 0.0),  # X axis
                    'amount': -dx * self.dolly_sensitivity * 10
                }
            else:
                # Vertical drag = dolly (in/out)
                result['zoom'] = -dy * self.zoom_sensitivity
        
        return result
    
    def _quaternion_from_axis_angle(self, axis, angle_rad):
        """Helper to create quaternion from axis-angle."""
        import numpy as np
        axis = np.asarray(axis, dtype=np.float32)
        n = np.linalg.norm(axis)
        if n < 1e-8:
            return np.array([1, 0, 0, 0], dtype=np.float32)
        axis /= n
        h = angle_rad * 0.5
        return np.array([np.cos(h), axis[0]*np.sin(h), axis[1]*np.sin(h), axis[2]*np.sin(h)], dtype=np.float32)
    
    def apply_to_camera(self, camera, delta: dict) -> None:
        """Apply the calculated movement to the camera."""
        if 'orbit' in delta:
            camera.orbit(
                delta['orbit']['yaw'],
                delta['orbit']['pitch']
            )
            
        if 'pan' in delta:
            camera.pan(
                delta['pan']['right'],
                delta['pan']['up']
            )
            
        if 'pan_axis' in delta:
            # Apply translation along a specific world axis
            import numpy as np
            axis = np.array(delta['pan_axis']['axis'], dtype=np.float32)
            amount = delta['pan_axis']['amount']
            camera.position = camera.position + axis * amount
            
        if 'zoom' in delta:
            camera.zoom(delta['zoom'])
    
    def handle_wheel(self, event: pygame.event.Event, camera) -> None:
        """Handle mouse wheel for zoom."""
        if event.y != 0:
            camera.zoom(event.y * self.zoom_sensitivity)

