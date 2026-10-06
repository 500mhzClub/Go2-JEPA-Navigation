"""The robot's collision geometry and its clearance from the maze walls, for the evaluator.

`RobotGeometry` reads the 27 collision primitives (boxes, cylinders, spheres) of Genesis' Go2 URDF and places them by
forward kinematics from the measured joint angles. `wall_clearance` bounds, at every physics step, each primitive's
distance to the nearest wall box (the walls are the true maze geometry; controllers never see them):
- **lower**: the separating-axis bound along the walls' face normals (can only under-estimate the distance);
- **upper**: the distance from the primitive's support point toward the nearest wall point (an achieved distance);
- **robust**: per interval, the smaller lower bound of its two ends minus how far the primitive can have moved
  (centre displacement plus rotation angle times its bounding radius).
"""
from pathlib import Path
import importlib.util
import math
import xml.etree.ElementTree as ET

import numpy as np

from go2nav.perception.rotations import rotation_increment
from go2nav.sim.sensors import JOINT_NAMES, rotation_xyzw

PRIMITIVES = 27


def go2_urdf():
    spec = importlib.util.find_spec('genesis')
    return Path(next(iter(spec.submodule_search_locations)))/'assets'/'urdf'/'go2'/'urdf'/'go2.urdf'


def origin_transform(element):
    result = np.eye(4)
    if element is None:
        return result
    xyz = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    result[:3, :3] = rotation_increment([0, 0, rpy[2]])@rotation_increment([0, rpy[1], 0])@rotation_increment([rpy[0], 0, 0])
    result[:3, 3] = xyz
    return result


def support_radius(kind, dimensions, directions):
    """Support of a primitive about its centre along each direction (rows)."""
    size, normals = np.asarray(dimensions, dtype=float), np.asarray(directions, dtype=float)
    if kind == 'box':
        return np.abs(normals)@(size/2)
    if kind == 'sphere':
        return size[0]*np.linalg.norm(normals, axis=1)
    radius, length = size
    return radius*np.linalg.norm(normals[:, :2], axis=1)+length/2*np.abs(normals[:, 2])


class RobotGeometry:
    def __init__(self, urdf_path=None):
        root = ET.parse(urdf_path or go2_urdf()).getroot()
        self.joints, self.shapes = [], []
        for node in root.findall('joint'):
            revolute = node.get('type') == 'revolute'
            self.joints.append(dict(name=node.get('name'), revolute=revolute, parent=node.find('parent').get('link'),
                                    child=node.find('child').get('link'),
                                    axis=np.fromstring(node.find('axis').get('xyz'), sep=' ') if revolute else np.zeros(3),
                                    origin=origin_transform(node.find('origin'))))
        for link in root.findall('link'):
            for index, node in enumerate(link.findall('collision')):
                primitive = list(node.find('geometry'))[0]
                kind = primitive.tag
                if kind == 'box':
                    size = np.fromstring(primitive.get('size'), sep=' ')
                elif kind == 'sphere':
                    size = np.array([float(primitive.get('radius'))])
                elif kind == 'cylinder':
                    size = np.array([float(primitive.get('radius')), float(primitive.get('length'))])
                else:
                    raise ValueError(f'unsupported collision geometry {kind}')
                self.shapes.append(dict(shape_id=f"{link.get('name')}:{index}", link=link.get('name'), kind=kind,
                                        dimensions=size, origin=origin_transform(node.find('origin'))))
        if len(self.shapes) != PRIMITIVES:
            raise ValueError('the Go2 URDF has 27 collision primitives')

    def transforms(self, joint_position):
        """Body-from-link transforms for the joint angles (rollout joint order)."""
        angles = dict(zip(JOINT_NAMES, np.asarray(joint_position, dtype=float), strict=True))
        transforms, pending = {'base': np.eye(4)}, list(self.joints)
        while pending:
            for joint in [j for j in pending if j['parent'] in transforms]:
                rotation = np.eye(4)
                if joint['revolute']:
                    rotation[:3, :3] = rotation_increment(joint['axis']*angles[joint['name']])
                transforms[joint['child']] = transforms[joint['parent']]@joint['origin']@rotation
                pending.remove(joint)
        return transforms

    def supports(self, joint_position, directions_body):
        """Per primitive: (lower, upper) extents along each body direction."""
        normals = np.asarray(directions_body, dtype=float)
        transforms = self.transforms(joint_position)
        rows = []
        for shape in self.shapes:
            transform = transforms[shape['link']]@shape['origin']
            radius = support_radius(shape['kind'], shape['dimensions'], normals@transform[:3, :3])
            centre = normals@transform[:3, 3]
            rows.append(((centre-radius).tolist(), (centre+radius).tolist()))
        return rows, transforms


class WallClearance:
    """Clearance of the robot's primitives from a maze's wall boxes (see the module docstring)."""

    def __init__(self, wall_boxes, geometry=None):
        self.model = geometry or RobotGeometry()
        self.centres = np.array([w['centre_xyz'] for w in wall_boxes])
        self.halves = np.array([w['size_xyz'] for w in wall_boxes])/2
        self.axes = np.array([[[math.cos(w['yaw_rad']), math.sin(w['yaw_rad']), 0], [-math.sin(w['yaw_rad']), math.cos(w['yaw_rad']), 0],
                               [0, 0, 1]] for w in wall_boxes])
        self.normals = self.axes.reshape(-1, 3)
        self.wall_projection = np.einsum('wij,wj->wi', self.axes, self.centres)
        self.radii = np.array([np.linalg.norm(s['dimensions']/2) if s['kind'] == 'box' else s['dimensions'][0]
                               if s['kind'] == 'sphere' else math.hypot(s['dimensions'][0], s['dimensions'][1]/2)
                               for s in self.model.shapes])

    def evaluate(self, base_pose_world, joint_position):
        """(lower (steps, 27), upper (steps, 27), robust (steps-1, 27)) for consecutive physics steps."""
        lower, upper, centres, rotations = [], [], [], []
        for pose, q in zip(base_pose_world, joint_position, strict=True):
            Q = rotation_xyzw(pose[3:])
            supports, transforms = self.model.supports(q, self.normals@Q)
            translation = self.normals@pose[:3]
            lo = (np.array([s[0] for s in supports])+translation).reshape(PRIMITIVES, -1, 3)
            hi = (np.array([s[1] for s in supports])+translation).reshape(PRIMITIVES, -1, 3)
            lower.append(np.maximum(lo-(self.wall_projection+self.halves), self.wall_projection-self.halves-hi).max(axis=2).min(axis=1))
            placed = [transforms[s['link']]@s['origin'] for s in self.model.shapes]
            world_centres = np.array([pose[:3]+Q@T[:3, 3] for T in placed])
            world_rotations = np.array([Q@T[:3, :3] for T in placed])
            centres.append(world_centres)
            rotations.append(world_rotations)
            upper.append([self._witness_distance(shape, centre, orientation)
                          for shape, centre, orientation in zip(self.model.shapes, world_centres, world_rotations, strict=True)])
        lower, upper, centres, rotations = np.asarray(lower), np.asarray(upper), np.asarray(centres), np.asarray(rotations)
        relative = np.einsum('spji,spjk->spik', rotations[:-1], rotations[1:])
        angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1))
        displacement = np.linalg.norm(np.diff(centres, axis=0), axis=-1)+angle*self.radii
        return lower, upper, np.minimum(lower[:-1], lower[1:])-displacement

    def _witness_distance(self, shape, centre, orientation):
        """Distance to the nearest wall of the primitive's support point toward each wall's nearest point."""
        delta = centre-self.centres
        local = np.einsum('wij,wj->wi', self.axes, delta)
        nearest = np.clip(local, -self.halves, self.halves)
        direction = np.einsum('wji,wj->wi', self.axes, nearest-local)
        local_direction = direction@orientation
        point, dims = np.zeros_like(local_direction), shape['dimensions']
        if shape['kind'] == 'box':
            point = np.sign(local_direction)*dims/2
        elif shape['kind'] == 'sphere':
            norms = np.linalg.norm(local_direction, axis=1, keepdims=True)
            point = np.divide(local_direction*dims[0], norms, out=point, where=norms > 0)
        else:
            norms = np.linalg.norm(local_direction[:, :2], axis=1, keepdims=True)
            point[:, :2] = np.divide(local_direction[:, :2]*dims[0], norms, out=point[:, :2], where=norms > 0)
            point[:, 2] = np.sign(local_direction[:, 2])*dims[1]/2
        witness = centre+point@orientation.T
        box = np.einsum('wij,wj->wi', self.axes, witness-self.centres)
        return float(np.linalg.norm(np.maximum(np.abs(box)-self.halves, 0), axis=1).min())
