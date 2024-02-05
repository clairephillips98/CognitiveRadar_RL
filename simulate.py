"""
Claire Phillips
Jan. 26, 2024
"""

from radar import Radar, look_new_direction
from target import Target
from utils import min_max_radar_breadth, radians_to_degrees
from math import pi
from PIL import Image, ImageDraw, ImageFilter
from create_plots import make_gif
from math import ceil, pi,floor
import numpy as np
import torch
import cv2



def create_radars():
    radar_1 = Radar(peak_power=400, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(0, 0), wavelength=3,
                    radians_of_view=pi / 4)
    radar_2 = Radar(peak_power=200, duty_cycle=3,
                    pulsewidth=4, bandwidth=1, frequency=3,
                    pulse_repetition_rate=3, antenna_size=4, cartesian_coordinates=(1, 1), wavelength=3,
                    radians_of_view=pi / 6)
    return [radar_1]  # , radar_2 just 1 radar for now


def bounds(radar):
    # create bounds around radars so we have image size
    x_lower, y_lower, x_upper, y_upper = min_max_radar_breadth(radar)  # update to include both radars
    return {'x_lower': x_lower, 'x_upper': x_upper, 'y_lower': y_lower, 'y_upper': y_upper, }


def create_targets(n_ts, bounds):
    targets = [Target(radius=1, bounds=bounds, path_eqn=None, name=n) for n in range(n_ts)]
    return targets


class Simulation:

    def __init__(self, blur_radius: int = 3, scale : int = 50):
        self.t = 0
        self.radars = create_radars()
        self.overall_bounds = bounds(self.radars[0])  # these are overall bounds for when there are multiple radars
        self.targets = create_targets(10, self.overall_bounds)
        self.scale = scale
        self.blur_radius = blur_radius
        self.base_image = Image.new("RGB", (
        self.blur_radius*3+ceil((self.overall_bounds['x_upper'] - self.overall_bounds['x_lower']) / self.scale),
        self.blur_radius*3+ceil((self.overall_bounds['y_upper'] - self.overall_bounds['y_lower']) / self.scale)), (150, 150, 150))
        self.last_image = self.base_image.copy()
        self.mask_image = self.create_mask()
        self.images = []
        self.polar_images = []
        self.initial_scan()

    def create_mask(self, angle_start=0, angle_stop=360):
        mask_im = Image.new("L", self.base_image.size, 0)
        draw = ImageDraw.Draw(mask_im)
        for radar in self.radars:
            draw.ellipse((140, 50, 260, 170), fill=255)
            x_min, y_min, x_max, y_max = min_max_radar_breadth(radar)
            draw.pieslice((self.blur_radius, self.blur_radius, (x_max - x_min) / self.scale+self.blur_radius, (y_max - y_min) / self.scale+self.blur_radius),
            angle_start,angle_stop, fill=255)
        return mask_im

    def initial_scan(self):
        # initial scan - just look in every direction for the max number of looks required

        steps = min([radar.num_states for radar in self.radars])
        integer_array = np.arange(0, steps)
        # np.random.shuffle(integer_array) we look in an order actually
        for step in integer_array:
            [rad.update_viewing_angle(step * rad.radians_of_view) for rad in self.radars]
            self.update_t(False)

    def update_t(self, random_dir=True):
        self.t = self.t + 1
        [rad.update_t(self.t, random_dir) for rad in self.radars]  # i think this is acutally pointless
        [tar.update_t(self.t) for tar in self.targets]
        [rad.update_seen_list(self.targets) for rad in self.radars]
        np_image, polar_image = self.create_image()
        self.images.append(np_image)
        self.polar_images.append(polar_image)

    def create_image(self):
        image = self.last_image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        new_image = ImageDraw.Draw(image)
        for radar in self.radars:
            x_min, y_min, x_max, y_max = min_max_radar_breadth(radar)
            new_image.pieslice((self.blur_radius, self.blur_radius, (x_max - x_min) / self.scale+self.blur_radius, (y_max - y_min) / self.scale+self.blur_radius),
                               radians_to_degrees(radar.viewing_angle),
                               radians_to_degrees(radar.viewing_angle + radar.radians_of_view), fill='white')
            visible_targets = radar.visible_targets(self.targets)
            for target in visible_targets:
                new_image.pieslice((floor((target.cartesian_coordinates[0] - target.radius - x_min) / self.scale)+self.blur_radius,
                                    floor((target.cartesian_coordinates[1] - target.radius - y_min) / self.scale)+self.blur_radius,
                                    ceil((target.cartesian_coordinates[0] + target.radius - x_min) / self.scale)+self.blur_radius,
                                    ceil((target.cartesian_coordinates[1] + target.radius - y_min) / self.scale)+self.blur_radius
                                    ), 0,
                                   360, fill='red')

        self.last_image=self.base_image.copy()
        self.last_image.paste(image, (0, 0), self.mask_image)
        #self.base_image.save("./here.png")
        np_image = np.array(self.last_image)
        polar_image= self.polar(np_image)
        return np_image,polar_image

    def reward_slice_cross_entropy(self):
        # figure out what section of the polar plot reflects the slice were observing
        # get past slice with guassian blur, converted into polar coodinates slice
        # get updated slice converted into polar coordiantes slice
        # prop_factor =
        # scale all values to be between 0 and 1
        # cross_entropy = -sum(floor(actual_value)*rescale_factor*log(last_slice))
        # flooring the actual value so that any value which is grey - the exterior is set to 0
        return None

    def reward_slice_abs_diff(self, new_image, last_image):
        # figure out what section of the polar plot reflects the slice were observing
        # get past slice with guassian blur, converted into polar coodinates slice
        # get updated slice converted into polar coordiantes slice
        # scale all values to be between 0 and 1
        
        return None

    def polar(self,image, display = False):

        # Get image dimensions
        height, width = image.shape[:2]
        x_min, y_min, x_max, y_max = min_max_radar_breadth(self.radars[0])
        # Define the center of the image (assumes the center is at the middle of the image)
        center = ((x_max - x_min) /self.scale/2+self.blur_radius, (y_max - y_min) / self.scale/2+self.blur_radius)
        # Convert Cartesian to polar coordinates
        polar_image = cv2.linearPolar(image, center, max(center), cv2.WARP_FILL_OUTLIERS)

        if display == True:
            # Convert the result back to a PIL Image
            polar_pil_image = Image.fromarray(polar_image)

            # Display the original and polar images
            polar_pil_image.save('./here.png')

        print(polar_image[:,:,0].shape)
        return polar_image[:,:,0]

def main():
    test = Simulation(2,25)
    for t in range(20):
        test.update_t()
    images = [Image.fromarray(im) for im in test.images]
    images[0].save("./images/cartesian.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    images = [Image.fromarray(im) for im in test.polar_images]
    images[0].save("./images/polar.gif", save_all=True, append_images=images, duration=test.t, loop=0)
    print('done')


if __name__ == "__main__":
    main()
