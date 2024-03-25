import cmath
import math


def cartesian_to_polar(cartesian_p):
    x, y = cartesian_p
    z = complex(x, y)
    r, theta = cmath.polar(z)
    return r, 360*theta/(2*math.pi)


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
    angle = angle % 360
    start_angle = start_angle % 360
    end_angle = end_angle % 360

    # Check if angle is between start_angle and end_angle
    if start_angle <= end_angle:
        return start_angle <= angle <= end_angle
    else:
        # Handle the case where the range spans across 0
        return start_angle <= angle or angle <= end_angle

def in_circle_cartesian(target, radar, radius):
    # take in target location and radar location ad the wedge to look in
    tar_radius,_ = cartesian_to_polar(relative_location(target, radar))
    if (tar_radius < radius):
        return True
    else:
        return False
def in_wedge_cartesian(target, radar, radius, angle_start, angle_end):
    # take in target location and radar location ad the wedge to look in
    tar_radius,tar_angle = cartesian_to_polar(relative_location(target, radar))
    if (tar_radius < radius) & (is_angle_between(tar_angle, angle_start, angle_end)):
        return True, tar_radius,tar_angle
    else:
        return False, tar_radius,tar_angle

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


def draw_shape(x, y, center, start_angle, end_angle, radius):
    # Compute distances from the center
    scaled_center = [(center[0] - self.overall_bounds['x_lower']) / self.scale + self.blur_radius,
                     (center[1] - self.overall_bounds['y_lower']) / self.scale + self.blur_radius]
    distances = torch.sqrt((x - scaled_center[0]) ** 2 + (y - scaled_center[1]) ** 2)
    # Compute angles from the center
    angles = torch.atan2(y - scaled_center[1], x - scaled_center[0])
    angles = angles = (angles * 180 / torch.tensor(pi)).int() % 360  # Convert angles to degrees
    # Create a binary mask for the pie slice
    scaled_radius = radius / self.scale
    if start_angle <= end_angle:
        mask = (distances <= scaled_radius) & (angles >= start_angle) & (angles <= end_angle)
    else:
        mask = (distances <= scaled_radius) & ((angles >= start_angle) | (angles <= end_angle))
    return mask