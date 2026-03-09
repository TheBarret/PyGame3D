import sys
import pygame
import numpy as np
import random
from scene              import Scene
from camera             import Camera
from renderer           import WireframeRenderer
from wireframe_object   import WireframeObject
from input_handler      import CameraInputHandler
from config             import AppConfig
from factory            import SceneFactory

# ---------------------------------------------------------------------------
# Scene population
# ---------------------------------------------------------------------------

def create_scene(bundle) -> None:
    cube = WireframeObject.from_box(2, 2, 2, name="Cube")
    cube.color = (80, 200, 255)
    cube.transform.position = (0, 1, 0)

    #satellite = WireframeObject.from_sphere_approx(radius=0.4, lat_lines=6, lon_lines=8, name="Satellite")
    #satellite.color = (255, 180, 80)
    #satellite.set_parent(cube)
    #satellite.transform.position = (2, 0, 0)

    bundle.scene.add(
        create_grid(size=20, divisions=20, color=(40, 40, 60)),
        create_anchor(length=3, color=(255, 80, 80)),
        cube
        )


def create_grid(size, divisions, color):
    obj = WireframeObject.from_grid(size=size, divisions=divisions, name="Grid")
    obj.color    = color
    obj.pickable = False
    return obj


def create_anchor(length, color):
    obj = WireframeObject.from_axes(length=length, name="WorldAxes")
    obj.color    = color
    obj.pickable = False
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
    cube_angle  = 0.0
    running     = True

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
                    print(f"Picked: {result.object.name}" if result else "Picked: nothing")

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

        cube = bundle.scene.get("Cube")
        if cube:
            cube_angle += cfg.cube_spin_speed * dt
            cube.transform.set_euler_degrees(cube_angle, cube_angle, -cube_angle)

        # ── 3. Render ────────────────────────────────────────────────
        bundle.scene.render(bundle.screen)
        draw_hud(bundle)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
