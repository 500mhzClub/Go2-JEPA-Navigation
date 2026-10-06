"""Robot contacts with the world, as the evaluator sees them (never a controller input).

Every physics step the simulator's contact list (`robot.get_contacts(exclude_self_contact=False)`) is recorded.
- A contact is **disallowed** if the robot touches anything with a force above 1 mN, except a support link (a calf or
  foot) touching the ground. A disallowed contact stops the mission (`DISALLOWED_CONTACT`).
- The **guard** (armed after settling) also stops the mission on any loaded ground contact of a robot geometry other
  than the four foot spheres, a base speed above 0.3 m/s, or leaving the 16 x 16 m domain.
"""
import numpy as np

FIELDS = ('geom_a', 'geom_b', 'link_a', 'link_b', 'force_a', 'force_b', 'position', 'valid_mask')
FORCE_FLOOR_N = 1e-3
GUARD_SPEED_M_S, GUARD_HALF_EXTENT_M = .3, 8.


def native_contacts(robot):
    """The simulator's contact arrays for environment 0 (shape (1, K, ...))."""
    native = robot.get_contacts(exclude_self_contact=False)

    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    return {key: np.array(array(native[key]), copy=True) for key in FIELDS}


def attribute(packet, links, names, objects):
    """External robot contacts of environment 0: robot link, the other link and object, force on the robot, position,
    and whether the contact is disallowed (see the module docstring). Self-contacts are skipped."""
    rows = []
    a_links, b_links, valid = packet['link_a'][0], packet['link_b'][0], packet['valid_mask'][0]
    force_a, force_b = (np.asarray(packet[k], dtype=float)[0] for k in ('force_a', 'force_b'))
    position = np.asarray(packet['position'], dtype=float)[0]
    for index, (a, b, active) in enumerate(zip(a_links, b_links, valid)):
        if not active:
            continue
        a, b = int(a), int(b)
        a_robot, b_robot = a in links['robot'], b in links['robot']
        if a_robot == b_robot:
            continue
        robot_id, other_id = (a, b) if a_robot else (b, a)
        force = force_a[index] if a_robot else force_b[index]
        magnitude = float(np.linalg.norm(force))
        disallowed = magnitude > FORCE_FLOOR_N and not (robot_id in links['support'] and other_id in links['ground'])
        rows.append(dict(contact_index=index, robot_link_id=robot_id, robot_link_name=names.get(robot_id),
                         environment_link_id=other_id, environment_link_name=names.get(other_id),
                         environment_object_id=objects.get(other_id), force_on_robot_world_n=force.tolist(),
                         force_magnitude_n=magnitude, position_world_m=position[index].tolist(), disallowed=disallowed))
    return rows


def nonfoot_ground_contacts(packet, geoms):
    """Indices of loaded ground contacts of robot geometries that are not foot spheres."""
    robot, feet, ground = set(geoms['robot']), set(geoms['foot']), set(geoms['ground'])
    a, b, valid = (np.asarray(packet[k])[0] for k in ('geom_a', 'geom_b', 'valid_mask'))
    fa, fb = (np.asarray(packet[k], dtype=float)[0] for k in ('force_a', 'force_b'))
    violations = []
    for i in np.flatnonzero(valid):
        if a[i] in robot and b[i] in ground:
            geom, force = int(a[i]), fa[i]
        elif b[i] in robot and a[i] in ground:
            geom, force = int(b[i]), fb[i]
        else:
            continue
        if geom not in feet and float(np.linalg.norm(force)) > 0:
            violations.append(int(i))
    return violations
