"""""
Claire Phillips
ECE2500
Jan 18 2024

Defining a moving target
"""""
import math
from utils import relative_location,cartesian_to_polar,is_angle_between
import radar
import target


class RadarTargetRelationShip:

    def __init__(self, Radar, Target):
        self.Radar = Radar
        self.Target = Target
        self.rcs = self.radar_cross_section()
        self.relative_location = relative_location(self.Radar.cartesian_coordinates, self.Target.cartesian_coordinates)
        self.distance, self.angle = cartesian_to_polar(self.relative_location)
        self.target_viewed = self.target_in_view()

    def radar_cross_section(self):
        # no idea if this equaiton is correct but
        # https://www.rfcafe.com/references/electrical/ew-radar-handbook/radar-cross-section.htm#:~:text=Here%2C%20the%20RCS%20of%20a,where%20%CE%BB%2D2%CF%80r.
        return math.pi * self.Target.radius ** 2

    def update_t(self, t):
        self.Target.update_t(t)
        self.Radar.update_t(t)
        self.relative_location = relative_location(self.Radar.cartesian_coordinates, self.Target.cartesian_coordinates)
        self.distance, self.angle = cartesian_to_polar(self.relative_location)
        self.target_viewed = self.target_in_view()

    def target_in_view(self):
        return (self.distance < self.Radar.max_distance) & is_angle_between(self.angle,self.Radar.viewing_angle,self.Radar.viewing_angle+self.Radar.radians_of_view)

class SeenObject:

    def __init__(self, cartesian_coordinates, time, radar):
        # create a record of an object being seen at given coordinates
        self.cartesian_coordinates = cartesian_coordinates
        self.time = [time]
        self.radars = [radar]
        self.object_there = True
        self.relative_position = relative_location(cartesian_coordinates,radar.cartesian_coordinates)
        #maybe a metric of motion?

    def rechecked(self, time, seen):
        # update if an object is still there upon looking again
        self.time.append(time)
        self.object_there = seen

