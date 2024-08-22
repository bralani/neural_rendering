import pygame, time
import numpy as np
import torch
from pygame.locals import *

camera_pos = np.array([10, 0, 0], dtype=float)
camera_dir = np.array([-0.74240387650610384, -0.51983679072568478, -0.42261826174069972])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_camera(model):
    pygame.init()
    height, width = 256, 256
    screen_height, screen_width = 512, 512
    screen = pygame.display.set_mode((screen_width, screen_height))


    class Sphere:
        def __init__(self, center, radius, color):
            self.center = np.array(center)
            self.radius = radius
            self.color = np.array(color)  # Assicurati che il colore sia un array NumPy

        def convert_cart_to_sph(self, points):
            x = points[:, 0] - self.center[0]
            y = points[:, 1] - self.center[1]
            z = points[:, 2] - self.center[2]


            theta = torch.arccos(z / self.radius)
            phi = torch.arctan2(y, x)

            return torch.stack([theta, phi], axis=-1).to(device)

        def intersect(self, ray_origins, ray_direction):
            oc = ray_origins - self.center
            
            a = np.dot(ray_direction, ray_direction)
            b = 2.0 * np.dot(oc, ray_direction)
            c = np.einsum('ij,ij->i', oc, oc) - self.radius ** 2

            discriminant = b * b - 4 * a * c
            
            valid = discriminant > 0
            if np.sum(valid) == 0:
                return np.zeros(ray_origins.shape[0])

            sqrt_disc = np.sqrt(discriminant[valid])
            
            t0 = (-b[valid] - sqrt_disc) / (2.0 * a)
            t1 = (-b[valid] + sqrt_disc) / (2.0 * a)
            
            t0, t1 = np.minimum(t0, t1), np.maximum(t0, t1)

            valid[valid] = t0 > 0
            if np.sum(valid) == 0:
                return np.zeros(ray_origins.shape[0])
            
            sqrt_disc = np.sqrt(discriminant[valid])

            t0 = (-b[valid] - sqrt_disc) / (2.0 * a)
            t1 = (-b[valid] + sqrt_disc) / (2.0 * a)
            t0, t1 = np.minimum(t0, t1), np.maximum(t0, t1)
            
            intersection_points1 = np.full((ray_origins.shape[0], 3), np.nan)
            intersection_points2 = np.full((ray_origins.shape[0], 3), np.nan)
            
            intersection_points1[valid] = ray_origins[valid] + ray_direction * t0[:, np.newaxis]
            intersection_points2[valid] = ray_origins[valid] + ray_direction * t1[:, np.newaxis]

            point1_sph = self.convert_cart_to_sph(torch.Tensor(intersection_points1[valid]))
            point2_sph = self.convert_cart_to_sph(torch.Tensor(intersection_points2[valid]))

            
            rgb_values = np.zeros((ray_origins.shape[0], 3))
            input = torch.cat([point1_sph, point2_sph], axis=-1)
            if len(input) > 0:
                output_label, output_rgb = model(input)
                output_label[output_label > 0.5] = 1
                output_label[output_label <= 0.5] = 0
                output = (output_label * output_rgb) * 255
                output = output.cpu().detach().numpy()
                output = output.astype(int)
                rgb_values[valid] = output

            
            return rgb_values


    def trace_ray(ray_origins, ray_direction, scene, height, width):
        colors = np.zeros((height, width, 3), dtype=float)
        
        ray_origins = ray_origins.reshape(-1, 3)
        
        for obj in scene:
            hit_colors = obj.intersect(ray_origins, ray_direction)
        
        hit_colors = np.array(hit_colors)
        colors = hit_colors.reshape(height, width, 3)
        colors = np.rot90(colors, -1)
        
        return colors

    def handle_input():
        global camera_pos, camera_dir
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()

        if keys[K_w]:
            camera_pos += camera_dir * .3
        if keys[K_s]:
            camera_pos -= camera_dir * .3
        if keys[K_a]:
            camera_pos -= np.cross(camera_dir, [0, 1, 0]) * .1
        if keys[K_d]:
            camera_pos += np.cross(camera_dir, [0, 1, 0]) * .1
        
        if mouse_buttons[0]:
            mouse_rel = pygame.mouse.get_rel()
            sensitivity = 0.002
            horizontal_angle = -mouse_rel[0] * sensitivity
            rotation_matrix_y = np.array([[np.cos(horizontal_angle), 0, np.sin(horizontal_angle)],
                                        [0, 1, 0],
                                        [-np.sin(horizontal_angle), 0, np.cos(horizontal_angle)]])
            camera_dir[:] = np.dot(rotation_matrix_y, camera_dir)

            vertical_angle = mouse_rel[1] * sensitivity
            right_vector = np.cross([0, 1, 0], camera_dir)
            right_vector /= np.linalg.norm(right_vector)
            rotation_matrix_x = np.array([[1, 0, 0],
                                        [0, np.cos(vertical_angle), -np.sin(vertical_angle)],
                                        [0, np.sin(vertical_angle), np.cos(vertical_angle)]])
            camera_dir[:] = np.dot(rotation_matrix_x, camera_dir)
            camera_dir /= np.linalg.norm(camera_dir)





    def generate_rays(camera_position, view_direction, image_size, resolution):
        num_pixels_x, num_pixels_y = resolution
        width, height = resolution

        viewbase_model = np.array([232.98318264408528, 17.129209479118988, -13.872954520341018])
        dx_model = np.array([-0.48454082907333751, 0.69199601922625631, -4.2929945385577007E-16])
        dy_model = np.array([-0.29245015477688463, -0.20477580292537115, 0.76562267676141127])

        # Create a grid of pixel indices
        a_x, a_y = np.meshgrid(np.arange(num_pixels_x), np.arange(num_pixels_y))

        # Use broadcasting to calculate all pixel positions at once
        pixel_positions = viewbase_model + a_x[..., np.newaxis] * dx_model + a_y[..., np.newaxis] * dy_model

        return pixel_positions

    def render_scene(scene):

        now = time.time()
        ray_origins = generate_rays(camera_pos, camera_dir, (width, height), (width, height))
        print("Time to generate rays: ", time.time() - now)

        now = time.time()
        # Traccia tutti i raggi (assumendo che trace_ray supporti array di input)
        colors = trace_ray(ray_origins, camera_dir, scene, height, width)
        print("Time to trace rays: ", time.time() - now)

        now = time.time()
        # resize colors to screen resolution
        colors = np.repeat(np.repeat(colors, screen_height // height, axis=0), screen_width // width, axis=1)
        print("Time to resize colors: ", time.time() - now)

        # Clipping e conversione a interi
        colors = np.clip(colors, 0, 255).astype(int)

        now = time.time()
        pygame.surfarray.blit_array(screen, colors)
        print("Time to render image: ", time.time() - now)

        pygame.display.flip()

    # Loop principale
    running = True
    start = time.time()
    num_frames = 0
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        handle_input()
        render_scene([Sphere([20, 0, 19.5], 108.13070794182381, [255, 255, 255])])
        num_frames += 1
        delta = time.time() - start
        fps = num_frames / delta
        print("FPS: ", fps)

        if num_frames > 30:
            start = time.time()
            num_frames = 0

    pygame.quit()