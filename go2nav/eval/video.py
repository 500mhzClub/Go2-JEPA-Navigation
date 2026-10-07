"""Mission videos from a verified replay: the controller's egocentric RGB beside a chase camera, with a map of the
route and a status panel.

    python -m go2nav.eval.video RUN_DIR OUT_DIR [--note TEXT]

RUN_DIR is a mission run (`go2nav.eval.mission`, or an original run that `go2nav.data.rerender` accepts); a run
without episode_evaluation.json is evaluated first (`go2nav.eval.episode`).
- Pass 1 is the verified replay (`go2nav.data.rerender.replay`): the run's commands are re-executed from its seed,
  every physics sample is checked against the recorded trace and every frame against its recorded pixel hash, and
  the primary RGB frames (what the controller saw, 10 Hz) are kept for the left panel.
- Pass 2 replays the commands again in a fresh session, with the physics checked again, and renders a chase camera
  (0.6 m behind and 3 m above the body, smoothed) at every 1/30 s of simulated time.
The panel's phase, selected action, dispatch substitutions and arrivals come from the run's mission, planning,
request and evaluation records; its footer gives the outcome, disallowed contacts and minimum clearance. Every
video is labelled PRELIMINARY (development mode).

Output: OUT_DIR/<arm>_<set>_<maze>_ep<k>[_recovery-on].mp4 (30 fps, 1920x1080), a 4x-speed cut for missions over
120 s, a 12-frame contact sheet, and metadata.json (the run, the replay checks, ffprobe).
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

from go2nav.data import rerender
from go2nav.sim.session import rotation_from_pose

NAMES = dict(C0='Oracle (diagnostic)', C1='Command history', C1A='Command history, adaptive', C1R='Command history, refit',
             C2='Reactive', C3='JEPA', C4='Supervised predictor')
ORDINARY_DISPATCH = 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED'
FPS = 30
SIZE = (1920, 1080)
CHASE_OFFSET_BODY = np.array([-.6, 0., 3.])  # metres: behind (body x) and above (world z)
CHASE_SMOOTHING = .8                        # weight of the previous camera position
ACCELERATE_AFTER_S = 120.
MAP_SIZE = (640, 360)
COLOURS = dict(home=(50, 170, 70), beacon=(255, 170, 30), OUTBOUND=(80, 160, 250), RETURN=(230, 70, 150),
               robot=(220, 30, 30), wall=(65, 65, 65))


class Panel:
    """Composes one output frame: egocentric RGB, chase view, route map and status text."""

    def __init__(self, run: Path, ego_dir: Path, note: str | None):
        load = lambda name: json.loads((run/name).read_text())  # noqa: E731
        self.config, self.episode, self.spec = load('config.json'), load('episode.json'), load('specification.json')
        self.evaluation, self.requests = load('episode_evaluation.json'), load('requests.json')
        self.mission = {row['frame']: row for row in load('mission.json')}
        decisions = [row for row in load('planning.json') if 'selection' in row]
        self.decision_times = [row['measured_ns'] for row in decisions]
        self.actions = [row['selection']['action'] for row in decisions]
        self.arrivals = {row['phase']: row for row in self.evaluation['arrivals']}
        self.ego_dir, self.note = ego_dir, note
        self.ego_frame, self.ego, self.path = -1, None, []
        self.scale, self.origin = self._map_transform()
        self.map_base = self._draw_map()

    def _map_transform(self):
        boxes = self.spec['geometry']['wall_boxes']
        low = np.min([np.subtract(b['centre_xyz'][:2], np.divide(b['size_xyz'][:2], 2)) for b in boxes], axis=0)
        high = np.max([np.add(b['centre_xyz'][:2], np.divide(b['size_xyz'][:2], 2)) for b in boxes], axis=0)
        width, height, margin = MAP_SIZE[0]-110, MAP_SIZE[1], 12
        scale = min((width-2*margin)/(high[0]-low[0]), (height-2*margin)/(high[1]-low[1]))
        origin = np.array([margin, height-margin])+np.array([(width-2*margin-(high[0]-low[0])*scale)/2,
                                                             -(height-2*margin-(high[1]-low[1])*scale)/2])
        return scale, (origin, low)

    def point(self, xy):
        (x0, y0), low = self.origin
        return int(x0+(xy[0]-low[0])*self.scale), int(y0-(xy[1]-low[1])*self.scale)

    def _draw_map(self):
        image = np.full((MAP_SIZE[1], MAP_SIZE[0], 3), 235, np.uint8)
        for box in self.spec['geometry']['wall_boxes']:
            centre, half = np.array(box['centre_xyz'][:2]), np.array(box['size_xyz'][:2])/2
            cv2.rectangle(image, self.point(centre-half), self.point(centre+half), COLOURS['wall'], -1)
        cv2.circle(image, self.point(self.episode['home_se2_world'][:2]), 7, COLOURS['home'], -1)
        cv2.circle(image, self.point(self.episode['beacon_xy_world']), 7, COLOURS['beacon'], -1)
        for row, (text, key) in enumerate((('home', 'home'), ('beacon', 'beacon'), ('outbound', 'OUTBOUND'),
                                           ('return', 'RETURN'))):
            cv2.circle(image, (MAP_SIZE[0]-90, 20+row*22), 6, COLOURS[key], -1)
            cv2.putText(image, text, (MAP_SIZE[0]-78, 25+row*22), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 40, 40), 1,
                        cv2.LINE_AA)
        return image

    def arrival(self, phase, frame):
        row = self.arrivals.get(phase)
        if row is None or frame < row['frame']:
            return 'not yet'
        return 'reached (physically verified)' if row['passed'] else 'observed; rejected by the physical check'

    def outcome(self):
        e = self.evaluation
        hard = e['safety']['hard']
        result = ('round trip' if e['round_trip_success'] else 'beacon only' if e['beacon_success']
                  else f'failed ({e["source_error"] or "no arrival"})')
        return (f'{result} | disallowed contact samples: {e["disallowed_contact_samples"]} | '
                f'minimum clearance {hard["minimum_separation_lower_m"]*1000:.0f} mm')

    def draw(self, chase, pose, elapsed_s):
        frame = min(int(elapsed_s*10+1e-8), max(self.mission))
        if frame != self.ego_frame:
            self.ego = np.asarray(Image.open(self.ego_dir/f'{frame:04d}.png').convert('RGB'))
            self.ego_frame = frame
        canvas = np.full((SIZE[1], SIZE[0], 3), 20, np.uint8)
        canvas[:720, :960] = cv2.resize(self.ego, (960, 720))
        canvas[:720, 960:] = cv2.resize(chase, (960, 720))
        for text, x in (('EGOCENTRIC: the RGB the controller consumed (10 Hz)', 12),
                        ('EXOCENTRIC: chase camera (separate replay pass)', 972)):
            cv2.rectangle(canvas, (x-6, 8), (x+len(text)*11, 38), (0, 0, 0), -1)
            cv2.putText(canvas, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
        state = self.mission[frame]
        image = self.map_base.copy()
        self.path.append((self.point(pose[:2]), COLOURS.get(state['phase'], COLOURS['RETURN'])))
        for (a, _), (b, colour) in zip(self.path, self.path[1:]):
            cv2.line(image, a, b, colour, 2)
        heading = rotation_from_pose(pose)[:2, 0]
        cv2.circle(image, self.point(pose[:2]), 5, COLOURS['robot'], -1)
        cv2.arrowedLine(image, self.point(pose[:2]), self.point(pose[:2]+heading*.35), COLOURS['robot'], 2,
                        tipLength=.35)
        canvas[720:, :640] = image
        request = self.requests[min(int(elapsed_s/.02), len(self.requests)-1)]
        index = bisect_right(self.decision_times, request['simulator_ns'])-1
        action = self.actions[index] if index >= 0 else 'awaiting the first decision'
        dispatch = '' if request['reason'] == ORDINARY_DISPATCH else f' | dispatch: {request["reason"].lower()}'
        c = self.config
        lines = [f'{c["arm"]} {NAMES.get(c["arm"], "")} | reserve-exit-v2 harness | recovery {"on" if c["recovery"] else "off"}',
                 f'{c["set"]} maze {c["maze"]}, episode {c["episode"]} | mission time {elapsed_s:6.2f} s',
                 f'Phase: {state["phase"]} | action: {action}{dispatch}',
                 f'Beacon: {self.arrival("OUTBOUND", frame)} | home: {self.arrival("RETURN", frame)}',
                 f'Run result: {self.outcome()}',
                 'PRELIMINARY (development mode)' + (f' | {self.note}' if self.note else '')]
        for row, line in enumerate(lines):
            cv2.putText(canvas, line, (670, 770+row*48), cv2.FONT_HERSHEY_SIMPLEX, .68, (240, 240, 240), 1, cv2.LINE_AA)
        return canvas


def make(run, out, *, note=None):
    from go2nav import artifacts
    from go2nav.sim import scene
    from go2nav.sim.session import MissionSession, PhysicalStop
    import genesis as gs
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (run/'episode_evaluation.json').exists():
        from go2nav.eval.episode import evaluate
        evaluate(run)
    config, spec = json.loads((run/'config.json').read_text()), json.loads((run/'specification.json').read_text())
    stem = f'{config["arm"]}_{config["set"]}_{config["maze"]}_ep{config["episode"]}'+('_recovery-on' if config['recovery'] else '')
    frames = rerender.replay(run, out/'replay')  # pass 1: verified replay, egocentric frames
    gs.destroy()
    panel = Panel(run, out/'replay'/'ego_frames', note)
    requests = panel.requests
    with np.load(run/'native/physics_trace.npz') as archive:
        trace = {key: archive[key].copy() for key in rerender.TRACE_FIELDS}
    scene.init_genesis(spec['procedural_seed'])  # pass 2: chase camera
    session = MissionSession(spec, out/'chase_meshes', policy_checkpoint=artifacts.path('go2_locomotion_policy'),
                             dynamics=rerender.run_dynamics(run), demo_textures=rerender.run_demo_textures(run))
    session.settle()
    checked = rerender.check_samples(session, trace, 0)
    epoch_s = requests[0]['simulator_ns']/1e9
    duration_s = float(trace['timestamp_s'][-1])-epoch_s
    total = math.ceil(duration_s*FPS-1e-8)
    writers = [_encoder(out/f'{stem}.mp4')]
    if duration_s > ACCELERATE_AFTER_S:
        writers.append(_encoder(out/f'{stem}_4x.mp4'))
    sheet_frames, sheet = set(np.linspace(0, total-1, 12).astype(int).tolist()), []
    count, camera_position = 0, None

    def render(pose):
        nonlocal count, camera_position
        R = rotation_from_pose(pose)
        desired = pose[:3]+CHASE_OFFSET_BODY[0]*R[:, 0]+np.array([0., 0., CHASE_OFFSET_BODY[2]])
        camera_position = desired if camera_position is None else (1-CHASE_SMOOTHING)*desired+CHASE_SMOOTHING*camera_position
        camera = session.world.camera
        camera.set_pose(pos=camera_position, lookat=pose[:3], up=[0., 0., 1.])
        rgb = np.asarray(camera.render(rgb=True, depth=False, segmentation=False, normal=False)[0]).reshape(480, 640, 3)
        image = panel.draw(rgb, pose, count/FPS)
        writers[0].stdin.write(image.tobytes())
        if len(writers) > 1 and count % 4 == 0:
            fast = image.copy()
            cv2.rectangle(fast, (720, 640), (1200, 700), (0, 0, 160), -1)
            cv2.putText(fast, '4x SIMULATED TIME', (748, 682), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            writers[1].stdin.write(fast.tobytes())
        if count in sheet_frames:
            sheet.append(cv2.resize(image, (480, 270)))
        count += 1

    sample = session._sample

    def sample_and_render(requested, applied, timestamp_s):
        try:
            sample(requested, applied, timestamp_s)
        finally:  # a sample that stops the run is recorded before the stop is raised
            if count < total and timestamp_s-epoch_s+1e-9 >= count/FPS:
                render(session.samples[-1]['base_pose_world'])
    session._sample = sample_and_render
    try:
        render(session.samples[-1]['base_pose_world'])
        for row in requests:
            session.phase = 2
            try:
                session.step(row['requested_command'])
            except PhysicalStop:  # the run's own ending (its last request was never applied)
                if 'applied_command' in row:
                    raise
                break
            checked = rerender.check_samples(session, trace, checked)
    finally:
        for writer in writers:
            writer.stdin.close()
            writer.wait()
        session.world.scene.destroy()
        gs.destroy()
    if any(writer.returncode for writer in writers):
        raise RuntimeError('ffmpeg failed')
    if count != total:
        raise ValueError(f'rendered {count} chase frames, expected {total}')
    contact_sheet = np.zeros((1080, 1440, 3), np.uint8)
    for i, image in enumerate(sheet[:12]):
        contact_sheet[i//3*270:(i//3+1)*270, i % 3*480:(i % 3+1)*480] = image
    Image.fromarray(contact_sheet).save(out/f'{stem}_contact_sheet.png')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames', '-of', 'json',
                            str(out/f'{stem}.mp4')], capture_output=True, text=True, check=True)
    metadata = dict(status='PRELIMINARY_VIDEO_REPLAY_VERIFIED', run=str(run), config=config,
                    replay=dict(egocentric_frames=frames, physics_samples_checked=checked),
                    chase=dict(frames=count, offset_m=CHASE_OFFSET_BODY.tolist(), smoothing=CHASE_SMOOTHING,
                               render_time_quantisation_ms=2),
                    outcome=panel.outcome(), note=note,
                    outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob(f'{stem}*'))},
                    ffprobe=json.loads(probe.stdout))
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def _encoder(path: Path):
    return subprocess.Popen(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s',
                             f'{SIZE[0]}x{SIZE[1]}', '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                             '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                             str(path)], stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--note', help='an extra line for the panel footer')
    a = p.parse_args()
    metadata = make(a.run, a.out, note=a.note)
    print(json.dumps(dict(outputs=list(metadata['outputs']), outcome=metadata['outcome'])))


if __name__ == '__main__':
    main()
