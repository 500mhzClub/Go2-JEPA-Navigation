"""Demonstration ("hero") video of one navigation run, for presentation only: nothing here feeds an experiment.

    python -m go2nav.eval.hero_video RUN_DIR OUT.mp4 [--ego EGO_DIR] [--still SECONDS PNG]

Composition (1920x1080, 30 fps, 2x simulated time; stretches longer than 2 s without translation play at 6x with a
fast-forward badge, and the mission clock keeps simulated time):
- **Main view:** a smoothed drone-style follow camera, re-rendered kinematically from the run's physics trace (base
  pose and the 12 joint angles; no physics) in a presentation scene: the run's wall boxes with CC0 textures
  (assets/textures), a textured floor, soft light, and markers for home and the goal. The textures and markers are
  presentation choices, not what the robot saw, except for a run with demonstration surfaces (`--demo-textures`):
  then the scene draws the very brick and parquet meshes the robot saw (`go2nav.sim.demo_appearance`), and the video
  says the models never saw them in training.
- **Inset:** the egocentric frames the controller consumed, from a verified replay (EGO_DIR, e.g. the replay/ego_frames
  of `go2nav.eval.video`; without --ego the replay runs first, into OUT's folder).
- **Minimap:** the walls, the trail, home and goal, and the latest decision's forecast for each candidate action
  (planning.json `forecast_xy_m`; for an original run, `motion_correction.raw_forecast_xy_m`), the chosen one
  highlighted; a second panel zooms in on the forecasts in the robot's frame.
- **Status bar:** phase, mission time, distance, speed and decision count; an end card gives the result, contacts and
  closest approach. Labelled PRELIMINARY.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from go2nav.planning.actions import ACTIONS

REPO = Path(__file__).resolve().parents[2]
FLOOR_TEXTURE = REPO/'assets/textures/floor/WoodFloor043.jpg'
WALL_TEXTURE = REPO/'assets/textures/wall/PaintedPlaster017.jpg'
W, H, FPS, SPEED, FAST = 1920, 1080, 30, 2., 6.
ACCENT, GOAL, HOME, TEXT, CHOSEN = (90, 200, 255), (255, 150, 40), (80, 150, 255), (240, 244, 248), (120, 255, 160)
FONTS = {'Regular': ('Lato-Regular', 'DejaVuSans'), 'Bold': ('Lato-Bold', 'DejaVuSans-Bold'),
         'Black': ('Lato-Black', 'DejaVuSans-Bold'), 'Italic': ('Lato-Italic', 'DejaVuSans-Oblique')}
ACTION_NAMES = dict(forward='forward', left_arc='arc left', right_arc='arc right', left_turn='turn left',
                    right_turn='turn right', hold='hold')


def font(weight, size):
    """Lato if installed, else DejaVu Sans."""
    for name in FONTS[weight]:
        for folder in ('lato', 'dejavu'):
            path = Path('/usr/share/fonts/truetype')/folder/f'{name}.ttf'
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def rotation_2d(angle):
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


class Run:
    def __init__(self, run: Path, ego: Path):
        load = lambda name: json.loads((run/name).read_text())  # noqa: E731
        config = load('config.json')
        self.jepa = config.get('arm', config.get('controller')) == 'C3'
        self.textured = bool(config.get('demo_textures'))
        with np.load(run/'native/physics_trace.npz') as z:
            self.t, self.pose, self.joints = z['timestamp_s'].copy(), z['base_pose_world'].copy(), z['joint_position'].copy()
        self.walls = load('specification.json')['geometry']['wall_boxes']
        episode = load('episode.json')
        self.home, self.goal, self.maze = np.asarray(episode['home_se2_world'][:2]), np.asarray(episode['beacon_xy_world']), episode['maze_id']
        requests = load('requests.json')
        self.t0, self.t1 = requests[0]['simulator_ns']/1e9, requests[-1]['simulator_ns']/1e9
        self.phases = [(row['measured_ns']/1e9, row['phase']) for row in load('mission.json')]
        self.decisions = []
        for row in load('planning.json'):
            forecast = row.get('forecast_xy_m') or (row.get('motion_correction') or {}).get('raw_forecast_xy_m')
            if 'selection' in row and forecast is not None:
                self.decisions.append((row['measured_ns']/1e9, np.asarray(forecast), int(row['selection']['action_index'])))
        evaluation = load('episode_evaluation.json')
        self.success = bool(evaluation['round_trip_success'])
        self.contacts = int(evaluation['disallowed_contact_samples'])
        self.clearance = float(evaluation['safety']['hard']['minimum_separation_lower_m'])
        self.ego, self.ego_count = ego, len(list(ego.glob('*.png')))
        xs, ys = [w['centre_xyz'][0] for w in self.walls], [w['centre_xyz'][1] for w in self.walls]
        self.bounds = (min(xs)-.8, max(xs)+.8, min(ys)-.8, max(ys)+.8)
        self.fast = self._fast_forward()

    def _fast_forward(self):
        """10-Hz flags: True inside stretches longer than 2 s whose 1-s translation stays under 3 cm (from 1 s after the
        stretch starts to 0.5 s before it ends)."""
        grid = np.arange(self.t0, self.t1, .1)
        xy = self.pose[np.searchsorted(self.t, grid).clip(0, len(self.t)-1), :2]
        speed = np.zeros(len(grid))
        speed[5:-5] = np.linalg.norm(xy[10:]-xy[:-10], axis=1)
        slow, fast, k = speed < .03, np.zeros(len(grid), bool), 0
        while k < len(slow):
            e = k
            while e < len(slow) and slow[e] == slow[k]:
                e += 1
            if slow[k] and (e-k)*.1 > 2.:
                fast[k+10:max(k+10, e-5)] = True
            k = e
        return fast

    def fast_at(self, t):
        return bool(self.fast[int(np.clip((t-self.t0)/.1, 0, len(self.fast)-1))])

    def playback(self):
        times, t = [], self.t0
        while t < self.t1:
            times.append(t)
            t += (FAST if self.fast_at(t) else SPEED)/FPS
        return times+[self.t1]

    def index(self, t):
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t)-1))

    def phase_at(self, t):
        current = self.phases[0][1]
        for start, phase in self.phases:
            if start > t:
                break
            current = phase
        return current

    def decision_at(self, t):
        best = None
        for decision in self.decisions:
            if decision[0] > t:
                break
            best = decision
        return best

    def ego_frame(self, t):
        k = int(np.clip((t-self.t0)/.1, 0, self.ego_count-1))
        return Image.open(self.ego/f'{k:04d}.png').convert('RGB')


def build_scene(run):
    import genesis as gs
    from go2nav.corpus.textures import cached_box_obj
    from go2nav.sim.scene import LEG_JOINTS, go2_urdf
    gs.init(backend=gs.cpu, logging_level='warning')
    light = gs.options.vis.DirectionalLight(dir=(-.45, -.35, -1.), color=(1., .96, .9), intensity=3.2)
    vis = gs.options.VisOptions(shadow=True, ambient_light=(.42, .42, .46), background_color=(.86, .9, .95),
                                plane_reflection=False, lights=[light])
    scene = gs.Scene(show_viewer=False, vis_options=vis, renderer=gs.renderers.Rasterizer())
    x0, x1, y0, y1 = run.bounds

    def textured(path):
        return gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(path), encoding='srgb'))
    if run.textured:  # the brick and parquet meshes the robot's camera saw
        from go2nav.sim.demo_appearance import surfaces
        for name, mesh in surfaces(run.walls):
            scene.add_entity(gs.morphs.MeshSet(files=[mesh], fixed=True, collision=False, visualization=True, decimate=False,
                                               convexify=False, align=False, file_meshes_are_zup=True), name=name)
    else:
        scene.add_entity(gs.morphs.Mesh(file=cached_box_obj((x1-x0+4., y1-y0+4., .04), tiles_per_m=.6),
                                        pos=((x0+x1)/2, (y0+y1)/2, -.02), fixed=True, collision=False,
                                        file_meshes_are_zup=True), surface=textured(FLOOR_TEXTURE))
        for wall in run.walls:
            scene.add_entity(gs.morphs.Mesh(file=cached_box_obj(tuple(wall['size_xyz']), tiles_per_m=.8),
                                            pos=tuple(wall['centre_xyz']), euler=(0, 0, math.degrees(wall['yaw_rad'])),
                                            fixed=True, collision=False, file_meshes_are_zup=True),
                             surface=textured(WALL_TEXTURE))
    for morph, surface in (
            (gs.morphs.Cylinder(radius=.035, height=.75, pos=(*run.goal, .375)), gs.surfaces.Default(color=(.95, .95, .95))),
            (gs.morphs.Sphere(radius=.11, pos=(*run.goal, .82)), gs.surfaces.Emission(color=(1., .55, .12))),
            (gs.morphs.Cylinder(radius=.32, height=.012, pos=(*run.goal, .006)), gs.surfaces.Default(color=(1., .6, .2))),
            (gs.morphs.Cylinder(radius=.34, height=.012, pos=(*run.home, .006)), gs.surfaces.Default(color=(.3, .58, 1.)))):
        morph.fixed, morph.collision = True, False
        scene.add_entity(morph, surface=surface)
    robot = scene.add_entity(gs.morphs.URDF(file=str(go2_urdf()), pos=tuple(run.pose[0, :3]), fixed=False, collision=False))
    camera = scene.add_camera(res=(W, H), pos=(0., 0., 4.), lookat=(0., 0., 0.), fov=48, GUI=False, far=60.)
    scene.build()
    joints = {joint.name: joint for joint in robot.joints}
    dofs = [int(np.asarray(joints[name].dofs_idx_local).reshape(-1)[0]) for name in LEG_JOINTS]
    return scene, robot, camera, dofs


class Follow:
    """A drone-style camera 1.5 m behind and 3.5 m above the robot, looking 0.3 m ahead of it, exponentially smoothed."""

    def __init__(self):
        self.position = self.target = self.heading = None

    def update(self, pose, alpha=.09):
        yaw = yaw_of(pose[3:])
        h = np.array([math.cos(yaw), math.sin(yaw)])
        self.heading = h if self.heading is None else (1-alpha*.6)*self.heading+alpha*.6*h
        heading = self.heading/np.linalg.norm(self.heading)
        position, target = np.array([*(pose[:2]-1.5*heading), 3.5]), np.array([*(pose[:2]+.3*heading), .2])
        self.position = position if self.position is None else (1-alpha)*self.position+alpha*position
        self.target = target if self.target is None else (1-alpha)*self.target+alpha*target
        return self.position, self.target


def render_view(run, scene, robot, camera, dofs, follow, t):
    i = run.index(t)
    pose = run.pose[i]
    robot.set_pos(pose[:3])
    robot.set_quat([pose[6], pose[3], pose[4], pose[5]])
    robot.set_dofs_position(run.joints[i], dofs)
    position, target = follow.update(pose, alpha=.09*(FAST/SPEED if run.fast_at(t) else 1.))
    camera.set_pose(pos=position, lookat=target, up=(0., 0., 1.))
    scene.visualizer.update(force=True)  # kinematic posing does not step the scene, so push the visuals explicitly
    rgb = camera.render(rgb=True, depth=False, segmentation=False, normal=False)[0]
    return Image.fromarray(np.asarray(rgb).reshape(H, W, 3).astype(np.uint8)), i


def rounded_panel(size, radius=18, fill=(12, 16, 22, 185)):
    panel = Image.new('RGBA', size, (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle((0, 0, size[0]-1, size[1]-1), radius=radius, fill=fill)
    return panel


def minimap(run, i, t, size=430):
    x0, x1, y0, y1 = run.bounds
    scale = (size-40)/max(x1-x0, y1-y0)
    ox, oy = 20+(size-40-(x1-x0)*scale)/2, 20+(size-40-(y1-y0)*scale)/2

    def to(p):
        return ox+(p[0]-x0)*scale, size-(oy+(p[1]-y0)*scale)
    panel = rounded_panel((size, size))
    d = ImageDraw.Draw(panel)
    for wall in run.walls:
        centre, half, R = np.asarray(wall['centre_xyz'][:2]), np.asarray(wall['size_xyz'][:2])/2, rotation_2d(wall['yaw_rad'])
        d.polygon([to(centre+R@np.array([sx*half[0], sy*half[1]])) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
                  fill=(225, 230, 238, 255))
    trail = [to(p) for p in run.pose[:i+1:50, :2]]
    for k in range(1, len(trail)):
        f = k/len(trail)
        d.line([trail[k-1], trail[k]], fill=(int(60+40*f), int(150+50*f), 255, int(110+145*f)), width=4)
    for centre, colour in ((run.home, HOME), (run.goal, GOAL)):
        x, y = to(centre)
        d.ellipse((x-9, y-9, x+9, y+9), fill=(*colour, 255), outline=(255, 255, 255, 255), width=2)
    decision = run.decision_at(t)
    if decision is not None:
        decided, forecast, selected = decision
        j = run.index(decided)
        start, R = run.pose[j, :2], rotation_2d(yaw_of(run.pose[j, 3:]))
        for k in range(forecast.shape[0]):
            chosen = k == selected
            d.line([to(start)]+[to(start+R@xy) for xy in forecast[k]], fill=(*CHOSEN, 255) if chosen else (255, 255, 255, 110),
                   width=4 if chosen else 2)
    pose = run.pose[i]
    x, y = to(pose[:2])
    yaw = yaw_of(pose[3:])
    d.polygon([(x+16*math.cos(yaw), y-16*math.sin(yaw)), (x+9*math.cos(yaw+2.5), y-9*math.sin(yaw+2.5)),
               (x+9*math.cos(yaw-2.5), y-9*math.sin(yaw-2.5))], fill=(*ACCENT, 255), outline=(255, 255, 255, 255))
    d.text((18, 12), 'MAP', font=font('Bold', 18), fill=(*TEXT, 230))
    lx = 72
    for colour, label in ((CHOSEN, 'chosen forecast'), ((255, 255, 255), 'candidates'), (GOAL, 'goal'), (HOME, 'home')):
        d.rectangle((lx, 17, lx+12, 29), fill=(*colour, 255))
        d.text((lx+16, 12), label, font=font('Regular', 14), fill=(*TEXT, 210))
        lx += 26+int(d.textlength(label, font=font('Regular', 14)))
    return panel


def forecast_panel(run, t, size=(430, 300)):
    """The latest decision's candidate forecasts (8 ticks of 100 ms) in the robot's frame, forward up, zoomed."""
    panel = rounded_panel(size)
    d = ImageDraw.Draw(panel)
    d.text((18, 12), f'{"WORLD-MODEL " if run.jepa else ""}FORECAST  ·  next 0.8 s', font=font('Bold', 16), fill=(*TEXT, 230))
    origin, scale = (size[0]//2, size[1]-46), 1500.  # px per metre
    d.ellipse((origin[0]-7, origin[1]-7, origin[0]+7, origin[1]+7), fill=(*ACCENT, 255))
    decision = run.decision_at(t)
    if decision is None:
        return panel
    _, forecast, selected = decision
    for k in sorted(range(forecast.shape[0]), key=lambda k: k == selected):
        points = [origin]+[(origin[0]-scale*xy[1], origin[1]-scale*xy[0]) for xy in forecast[k]]
        chosen = k == selected
        d.line(points, fill=(*CHOSEN, 255) if chosen else (255, 255, 255, 120), width=5 if chosen else 2)
        end = points[-1]
        d.ellipse((end[0]-4, end[1]-4, end[0]+4, end[1]+4), fill=(*CHOSEN, 255) if chosen else (255, 255, 255, 150))
    d.text((18, size[1]-30), f'chosen: {ACTION_NAMES[ACTIONS[selected]]}', font=font('Bold', 18), fill=(*CHOSEN, 255))
    d.text((size[0]-18, size[1]-30), '10 cm', font=font('Regular', 14), fill=(*TEXT, 170), anchor='ra')
    d.line((size[0]-18-scale*.1, size[1]-12, size[0]-18, size[1]-12), fill=(*TEXT, 170), width=2)
    return panel


def compose(run, view, i, t, decisions_so_far, distance):
    frame = view.convert('RGBA')
    vignette = Image.new('L', (W, H), 0)
    ImageDraw.Draw(vignette).rectangle((0, 0, W, 170), fill=110)
    frame = Image.composite(Image.new('RGBA', (W, H), (8, 10, 14, 255)), frame, vignette.filter(ImageFilter.GaussianBlur(60)))
    d = ImageDraw.Draw(frame)
    title, subtitle = (('JEPA world-model navigation', 'a V-JEPA world model forecasts each candidate move from camera images  ·  ')
                       if run.jepa else ('Go2 navigation', ''))
    d.text((48, 34), title, font=font('Black', 46), fill=(*TEXT, 255))
    d.text((50, 92), f'Unitree Go2 (simulated)  ·  {subtitle}reach the goal, return home', font=font('Regular', 22),
           fill=(*TEXT, 220))
    fast = run.fast_at(t)
    chips = [('PRELIMINARY DEV RUN', (255, 196, 64), (30, 22, 0))]
    if run.textured:
        chips.append(('BRICK & PARQUET TEXTURES NEVER SEEN IN TRAINING', (110, 225, 205), (0, 30, 26)))
    chips.append((f'      {FAST:.0f}× FAST-FORWARD' if fast else f'{SPEED:.0f}× SPEED', (255, 110, 110) if fast else (225, 230, 238),
                  (40, 6, 6) if fast else (20, 24, 30)))
    x = 50
    for label, fill, ink in chips:
        width = int(ImageDraw.Draw(frame).textlength(label, font=font('Bold', 18)))+28
        frame.alpha_composite(rounded_panel((width, 34), radius=17, fill=(*fill, 232)), (x, 128))
        ImageDraw.Draw(frame).text((x+14, 133), label, font=font('Bold', 18), fill=(*ink, 255))
        if label.startswith('      '):  # the fast-forward symbol
            for dx in (0, 11):
                ImageDraw.Draw(frame).polygon([(x+14+dx, 137), (x+14+dx, 155), (x+26+dx, 146)], fill=(*ink, 255))
        x += width+10
    frame.alpha_composite(minimap(run, i, t), (W-430-40, 40))
    frame.alpha_composite(forecast_panel(run, t), (W-430-40, 40+430+16))
    ego = run.ego_frame(t).resize((512, 384), Image.LANCZOS)
    frame.alpha_composite(rounded_panel((540, 446)), (40, H-446-40))
    mask = Image.new('L', ego.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, ego.size[0]-1, ego.size[1]-1), radius=12, fill=255)
    frame.paste(ego, (54, H-446-40+48), mask)
    d = ImageDraw.Draw(frame)
    d.text((56, H-446-40+12), f"ROBOT CAMERA  ·  the {'JEPA' if run.jepa else 'controller'}'s actual input", font=font('Bold', 18),
           fill=(*TEXT, 235))
    phase = run.phase_at(t)
    status = {'OUTBOUND': ('Heading to goal', GOAL), 'RETURN': ('Returning home', HOME)}.get(phase, (phase or '', ACCENT))
    if t >= run.t1-.05 and run.success:
        status = ('Home · round trip complete', CHOSEN)
    bx = 540+40+40
    frame.alpha_composite(rounded_panel((W-540-40-40-40, 112)), (bx, H-112-40))
    d = ImageDraw.Draw(frame)
    d.ellipse((bx+26, H-112-40+44, bx+46, H-112-40+64), fill=(*status[1], 255))
    d.text((bx+58, H-112-40+32), status[0], font=font('Bold', 34), fill=(*TEXT, 255))
    a, b = max(i-250, 0), min(i+250, len(run.pose)-1)
    speed = float(np.linalg.norm(run.pose[b, :2]-run.pose[a, :2])/(run.t[b]-run.t[a]+1e-9))
    sx = bx+500
    for label, value in (('MISSION TIME', f'{max(t-run.t0, 0.):.1f} s'), ('DISTANCE', f'{distance:.1f} m'),
                         ('SPEED', f'{speed:.2f} m/s'), ('JEPA DECISIONS' if run.jepa else 'DECISIONS', f'{decisions_so_far}')):
        d.text((sx, H-112-40+22), label, font=font('Bold', 14), fill=(*TEXT, 160))
        d.text((sx, H-112-40+44), value, font=font('Bold', 30), fill=(*TEXT, 255))
        sx += 185
    return frame.convert('RGB')


def end_card(frame, run):
    image = frame.convert('RGBA')
    image.alpha_composite(rounded_panel((900, 210), radius=26, fill=(10, 14, 20, 215)), ((W-900)//2, (H-210)//2-60))
    d = ImageDraw.Draw(image)
    title = 'Round trip complete' if run.success else 'Run ended'
    d.text((W//2, H//2-120), title, font=font('Black', 54), fill=(*CHOSEN, 255) if run.success else (*TEXT, 255), anchor='mm')
    d.text((W//2, H//2-52), f'{run.t1-run.t0:.0f} s  ·  {run.contacts} wall contacts  ·  closest approach {run.clearance*100:.0f} cm',
           font=font('Regular', 28), fill=(*TEXT, 240), anchor='mm')
    note = f'preliminary maze {run.maze}  ·  development run, not a sealed result'
    if run.textured:
        note += '  ·  textures unseen in training'
    d.text((W//2, H//2-12), note, font=font('Italic', 20), fill=(*TEXT, 180), anchor='mm')
    return image.convert('RGB')


def make(run_dir, out, *, ego=None, still=None):
    run_dir, out = Path(run_dir), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if ego is None:  # the verified replay's egocentric frames
        import genesis as gs
        from go2nav.data import rerender
        ego = out.parent/f'{out.stem}_replay'
        rerender.replay(run_dir, ego)
        gs.destroy()
        ego = ego/'ego_frames'
    run = Run(run_dir, Path(ego))
    scene, robot, camera, dofs = build_scene(run)
    follow = Follow()
    if still is not None:
        seconds, png = still
        for t in np.arange(run.t0, seconds, SPEED/FPS):
            follow.update(run.pose[run.index(t)])
        view, i = render_view(run, scene, robot, camera, dofs, follow, seconds)
        distance = float(np.sum(np.linalg.norm(np.diff(run.pose[:i+1:25, :2], axis=0), axis=1)))
        compose(run, view, i, seconds, sum(1 for d in run.decisions if d[0] <= seconds), distance).save(png)
        return dict(still=str(png))
    encoder = subprocess.Popen(['ffmpeg', '-y', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                                '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
                                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
    times = run.playback()
    distance, last_i, last = 0., 0, None
    for t in times:
        view, i = render_view(run, scene, robot, camera, dofs, follow, t)
        if i > last_i:
            distance += float(np.sum(np.linalg.norm(np.diff(run.pose[last_i:i+1:25, :2], axis=0), axis=1)))
        last_i = i
        last = compose(run, view, i, t, sum(1 for d in run.decisions if d[0] <= t), distance)
        encoder.stdin.write(last.tobytes())
    card = end_card(last, run)
    for _ in range(4*FPS):
        encoder.stdin.write(card.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('ffmpeg failed')
    return dict(out=str(out), frames=len(times)+4*FPS, seconds=(len(times)+4*FPS)/FPS, run=str(run_dir),
                success=run.success, contacts=run.contacts, clearance_m=run.clearance, fast_forward_s=float(run.fast.sum()*.1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--ego', type=Path, help='egocentric frame folder (NNNN.png) from a verified replay')
    p.add_argument('--still', nargs=2, metavar=('SECONDS', 'PNG'), help='write one composed frame instead of the video')
    a = p.parse_args()
    print(json.dumps(make(a.run, a.out, ego=a.ego, still=None if a.still is None else (float(a.still[0]), a.still[1]))))


if __name__ == '__main__':
    main()
