import sys
import pygame
import numpy as np
import random
from scene              import Scene
from camera             import Camera
from renderer           import WireframeRenderer
from primitive          import WireframeObject
from input_handler      import CameraInputHandler
from config             import AppConfig
from factory            import SceneFactory

# ---------------------------------------------------------------------------
# Scene population
# ---------------------------------------------------------------------------

def create_scene(bundle) -> None:
    c1 = WireframeObject.from_box(1, 1, 1, name="C1")
    c1.color = (90, 90, 90)
    c1.transform.position = (0, 0, 0)

    c2 = WireframeObject.from_sphere_approx(radius=0.5, lat_lines=2, lon_lines=3, name="C2")
    c2.color = (255, 0, 0)
    c2.transform.position = (0, 1, 0)
    c2.set_parent(c1)

    c3 = WireframeObject.from_sphere_approx(radius=0.5, lat_lines=3, lon_lines=3, name="C3")
    c3.color = (0, 255, 0)
    c3.transform.position = (0, 2, 0)
    c3.set_parent(c1)
    
    c4 = WireframeObject.from_sphere_approx(radius=0.5, lat_lines=4, lon_lines=3, name="C4")
    c4.color = (0, 0, 255)
    c4.transform.position = (0, 3, 0)
    c4.set_parent(c1)
    
    c5 = WireframeObject.from_sphere_approx(radius=0.5, lat_lines=6, lon_lines=3, name="C5")
    c5.color = (255, 255, 255)
    c5.transform.position = (0, 4, 0)
    c5.set_parent(c1)
    
    bundle.scene.add(
        create_grid(size=10, divisions=10, color=(40, 40, 60)),
        create_anchor(length=4, color=(255, 80, 80),parent=c1),
        c1, c2, c3, c4, c5
        )

def create_grid(size, divisions, color):
    obj = WireframeObject.from_grid(size=size, divisions=divisions, name="Grid")
    obj.color    = color
    obj.pickable = False
    return obj

def create_anchor(length, color, parent):
    obj = WireframeObject.from_axes(length=length, name="WorldAxes")
    obj.color    = color
    obj.pickable = False
    obj.set_parent(parent)
    return obj

# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_hud(bundle) -> None:
    cfg   = bundle.cfg
    stats = bundle.scene.stats

    lines = [
        f"FPS       : {bundle.clock.get_fps():.0f}",
        f"Frame     : {stats['frame_ms']:.1f} ms",
        f"Objects   : {stats['objects_drawn']} / {stats['objects_total']} / culled {stats['objects_culled']}",
        f"Edges     : {stats['edges_drawn']} / clipped: {stats['edges_clipped']}",
        f"Mode      : {bundle.handler.current_mode.name}",
        "R: Reset",
    ]

    x, y = cfg.hud_margin, cfg.hud_margin
    for line in lines:
        surf = bundle.font.render(line, True, cfg.hud_color)
        bundle.screen.blit(surf, (x, y))
        y += cfg.hud_line_height


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    cfg    = AppConfig()
    bundle = SceneFactory.create(cfg)
    
    create_scene(bundle)

    # Runtime state
    _angle  = 0.0
    running     = True
    base = bundle.scene.get("C1")

    while running:
        dt = bundle.clock.tick(cfg.target_fps) / 1000.0

        # ── 1. Events ────────────────────────────────────────────────
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                bundle.handler.handle_event(event)
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    bundle.reset_camera()
                elif event.key == pygame.K_a:
                    bundle.toggle_antialiasing()

            elif event.type == pygame.KEYUP:
                bundle.handler.handle_event(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                bundle.handler.handle_event(event)
                if event.button == 1:
                    w, h = bundle.screen.get_size()
                    result = bundle.scene.pick(
                        *event.pos, w, h,
                        max_distance=cfg.pick_max_distance,
                    )
                    print(f"Selected: {result.object.name}" if result else "Selected: nothing")

            elif event.type == pygame.MOUSEBUTTONUP:
                bundle.handler.handle_event(event)

            elif event.type == pygame.MOUSEMOTION:
                bundle.handler.handle_event(event)

            elif event.type == pygame.MOUSEWHEEL:
                bundle.handler.handle_wheel(event, bundle.camera)

            elif event.type == pygame.VIDEORESIZE:
                bundle.on_resize(event.w, event.h)

        # ── 2. Update ────────────────────────────────────────────────
        bundle.handler.update(bundle.camera, dt)

        
        if base:
            _angle += 30 * dt
            base.transform.set_euler_degrees(_angle, _angle, -_angle)

        # ── 3. Render ────────────────────────────────────────────────
        bundle.scene.render(bundle.screen)
        
        draw_hud(bundle)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()