
import sys
import pygame
import numpy as np

from scene import Scene
from camera import Camera
from renderer import WireframeRenderer
from wireframe_object import WireframeObject

# ---------------------------------------------------------------------------
# Build scene content
# ---------------------------------------------------------------------------

def build_scene(scene: Scene) -> None:
    """Populate the scene with a few demo objects."""

    # Ground grid
    grid = WireframeObject.from_grid(size=20, divisions=20, name="Grid")
    grid.color     = (40, 40, 60)
    grid.pickable  = False
    scene.add(grid)

    # Central cube (parent)
    cube = WireframeObject.from_box(2, 2, 2, name="CentralCube")
    cube.color = (80, 200, 255)
    cube.transform.position = (0, 1, 0)
    scene.add(cube)

    # Satellite sphere — parented to cube
    satellite = WireframeObject.from_sphere_approx(
        radius=0.4, lat_lines=6, lon_lines=8, name="Satellite"
    )
    satellite.color = (255, 180, 80)
    satellite.set_parent(cube)
    satellite.transform.position = (2, 0, 0)   # 2 units right of cube centre
    scene.add(satellite)

    # Three axis gizmos at origin
    axes = WireframeObject.from_axes(length=3, name="WorldAxes")
    axes.color    = (255, 80, 80)    # all red in this simple demo
    axes.pickable = False
    scene.add(axes)

    # A few scattered boxes
    import random; random.seed(42)
    for i in range(8):
        box = WireframeObject.from_box(
            random.uniform(0.3, 1.5),
            random.uniform(0.3, 1.5),
            random.uniform(0.3, 1.5),
            name=f"Box_{i}",
        )
        box.color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )
        box.transform.position = (
            random.uniform(-8, 8),
            random.uniform(0.2, 2.0),
            random.uniform(-8, 8),
        )
        scene.add(box)


# ---------------------------------------------------------------------------
# HUD helper
# ---------------------------------------------------------------------------

def draw_hud(surface: pygame.Surface, scene: Scene, font: pygame.font.Font, clock: pygame.time.Clock) -> None:
    stats = scene.stats
    lines = [
        f"FPS: {clock.get_fps():.0f}",
        f"Frame: {stats['frame_ms']:.1f} ms",
        f"Objects: {stats['objects_drawn']} / {stats['objects_total']}  "
        f"(culled {stats['objects_culled']})",
        f"Edges drawn: {stats['edges_drawn']}  clipped: {stats['edges_clipped']}",
        "",
        "Drag: orbit | Right-drag: pan | Scroll: zoom",
        "Click: pick | A: AA | F: fog | R: reset | ESC: quit",
    ]
    y = 8
    for line in lines:
        surf = font.render(line, True, (180, 220, 180))
        surface.blit(surf, (8, y))
        y += 16


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    WIDTH, HEIGHT = 920, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("pygame3d wireframe demo")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)

    # ---- Camera setup ----
    cam = Camera(fov_y=60, aspect=WIDTH / HEIGHT, near=0.1, far=500)
    cam.position = (0, 6, -12)
    cam.look_at((0, 0, 0))

    # ---- Renderer (inject custom settings) ----
    renderer = WireframeRenderer(
        screen_width  = WIDTH,
        screen_height = HEIGHT,
        antialiased   = False,
        depth_sort    = True,
    )

    # ---- Scene ----
    scene = Scene(camera=cam, renderer=renderer, background_color=(8, 8, 16))
    build_scene(scene)

    # ---- State ----
    mouse_down_left  = False
    mouse_down_right = False
    last_mouse       = (0, 0)
    cube_angle       = 0.0   # for animating the central cube
    fog_enabled      = False

    running = True
    while running:
        dt = clock.tick(60) / 1000.0   # seconds

        # -- Events --
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    cam.position = (0, 6, -12)
                    cam.look_at((0, 0, 0))
                elif event.key == pygame.K_a:
                    renderer.antialiased = not renderer.antialiased
                elif event.key == pygame.K_f:
                    fog_enabled = not fog_enabled
                    renderer.fog_color = (8, 8, 16) if fog_enabled else None
                    renderer.fog_near  = 0.6
                    renderer.fog_far   = 0.98

            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
                scene.on_resize(WIDTH, HEIGHT)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down_left = True
                    # Picking
                    w, h = screen.get_size()
                    result = scene.pick(*event.pos, w, h, max_distance=0.15)
                    if result:
                        print(f"Picked: {result.object.name}  dist={result.distance:.3f}")
                    else:
                        print("Picked: nothing")
                elif event.button == 3:
                    mouse_down_right = True
                last_mouse = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: mouse_down_left  = False
                if event.button == 3: mouse_down_right = False

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                dx = mx - last_mouse[0]
                dy = my - last_mouse[1]
                if mouse_down_left:
                    cam.orbit(dx * 0.4, dy * 0.4)
                if mouse_down_right:
                    cam.pan(-dx * 0.02, dy * 0.02)
                last_mouse = event.pos

            elif event.type == pygame.MOUSEWHEEL:
                cam.zoom(event.y * 0.5)

        # -- Animate: rotate the central cube (and drag the satellite with it) --
        cube_angle += 45 * dt   # 45 deg/sec
        cube = scene.get("CentralCube")
        if cube:
            # In a real project this calls euler_to_quaternion from math_core;
            # here we show the API shape.
            cube.transform.set_euler_degrees(0, cube_angle, 0)

        # -- Render --
        scene.render(screen)
        draw_hud(screen, scene, font, clock)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
