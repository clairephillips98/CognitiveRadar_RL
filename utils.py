import cmath
import math


def cartesian_to_polar(cartesian_p):
    x, y = cartesian_p
    z = complex(x, y)
    r, theta = cmath.polar(z)
    return r, theta


def polar_to_cartesian(angle_radians, radius):
    # Calculate x, y coordinates
    x = radius * math.cos(angle_radians)
    y = radius * math.sin(angle_radians)

    return x, y


# Function to convert radians to degrees
def radians_to_degrees(radians):
    degrees = math.degrees(radians)
    return degrees


def degrees_to_radians(degrees):
    degrees = math.degrees(degrees)
    return degrees


def relative_location(p1, p2):
    rel = (p1[0] - p2[0], p1[1] - p2[1])
    return rel


def is_angle_between(angle, start_angle, end_angle):
    # Normalize angles to be in the range [0, 2*pi)
    angle = angle % (2 * math.pi)
    start_angle = start_angle % (2 * math.pi)
    end_angle = end_angle % (2 * math.pi)

    # Check if angle is between start_angle and end_angle
    if start_angle <= end_angle:
        return start_angle <= angle <= end_angle
    else:
        # Handle the case where the range spans across 0
        return start_angle <= angle or angle <= end_angle


def in_wedge_cartesian(target, radar, radius, angle_start, angle_end):
    # take in target location and radar location ad the wedge to look in
    rel_location_polar = cartesian_to_polar(relative_location(target, radar))
    if (rel_location_polar[0] < radius) & (is_angle_between(rel_location_polar[1], angle_start, angle_end)):
        return True
    else:
        return False

def min_max_radar_breadth(radar):
    x_lower = radar.cartesian_coordinates[0] - radar.max_distance
    x_upper = radar.cartesian_coordinates[0] + radar.max_distance
    y_lower = radar.cartesian_coordinates[1] - radar.max_distance
    y_upper = radar.cartesian_coordinates[1] + radar.max_distance
    return x_lower, y_lower, x_upper, y_upper

def in_wedge_polar(rel_location_polar, radius, angle_start, angle_end):
    # take in relative location and the wedge to look in
    if (rel_location_polar[0] < radius) & (is_angle_between(rel_location_polar[1], angle_start, angle_end)):
        return True
    else:
        return False

def merge_dictionaries(*dicts):
  merged = {}
  for d in dicts:
    merged.update(d)
  return merged