import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numba import jit, prange
import random
import os
import glob
from PIL import Image
from datetime import datetime

# --- The Core Fractal Math Generator (Untouched, as it's our foundation) ---
class FractalMathEngine:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height

    @staticmethod
    @jit(nopython=True, parallel=True)
    def hybrid_mandelbrot_formula(grid, max_iter, power, warp_freq, warp_phase):
        iterations=np.zeros(grid.shape,dtype=np.float64)
        for i in prange(grid.shape[0]):
            for j in prange(grid.shape[1]):
                c = grid[i, j]
                z = 0.0j
                if warp_freq > 0:
                    c += 0.1*np.sin(c.real*warp_freq + warp_phase) + 0.1j*np.cos(c.imag*warp_freq + warp_phase)
                for n in range(max_iter):
                    if np.abs(z)>4.0:
                        log_zn=np.log(np.abs(z))/2.; nu=np.log(log_zn/np.log(2.))/np.log(2.)
                        iterations[i,j]=n-nu+1; break
                    z=(z.real - 1j*np.abs(z.imag))**power + c
                else: iterations[i,j]=max_iter
        return iterations

    @staticmethod
    @jit(nopython=True, parallel=True)
    def julia_set_formula(grid, max_iter, c, warp_freq, warp_phase):
        iterations = np.zeros(grid.shape, dtype=np.float64)
        field_lines = np.zeros(grid.shape, dtype=np.complex128)
        for i in prange(grid.shape[0]):
            for j in prange(grid.shape[1]):
                z = grid[i, j]
                if warp_freq > 0:
                    z += 0.1*np.sin(z.real*warp_freq+warp_phase)+0.1j*np.cos(z.imag*warp_freq+warp_phase)
                for n in range(max_iter):
                    if np.abs(z) > 4.0:
                        log_zn=np.log(np.abs(z))/2.; nu=np.log(log_zn/np.log(2.))/np.log(2.)
                        iterations[i, j]=n-nu+1; break
                    z = z**2 + c
                    field_lines[i,j] = z
                else: iterations[i, j] = max_iter
        return iterations, field_lines

# --- The Thematic Animation Engine ---
class ThematicAnimationEngine:
    def __init__(self, math_engine, output_dir):
        self.math_engine = math_engine
        self.output_dir = output_dir
        self.w = self.math_engine.width
        self.h = self.math_engine.height

    def _get_grid(self, zoom=2.2, center_x=0.0, center_y=0.0, rotation_angle=0.0):
        x = np.linspace(center_x - zoom, center_x + zoom, self.w)
        y = np.linspace(center_y - zoom, center_y + zoom, self.h)
        X, Y = np.meshgrid(x, y)
        if rotation_angle != 0.0:
            angle = np.deg2rad(rotation_angle)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            X_rot = X * cos_a - Y * sin_a
            Y_rot = X * sin_a + Y * cos_a
            X, Y = X_rot, Y_rot
        return X + 1j * Y

    def _compile_gif(self, frames_dir, anim_id, theme):
        print(f"\nCompiling GIF for {theme}...")
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        frames = [Image.open(f) for f in frame_files]
        output_path = os.path.join(self.output_dir, f"animation_{anim_id:02d}_{theme}.gif")
        frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)
        # Clean up frames
        for f in frame_files: os.remove(f)
        os.rmdir(frames_dir)
        print(f"Saved: {output_path}")

    # --- THEME SIMULATORS ---

    def create_fire_animation(self, anim_id, num_frames=100):
        theme = "fire"
        frames_dir = os.path.join(self.output_dir, f"frames_{theme}_{anim_id}")
        os.makedirs(frames_dir, exist_ok=True)
        colormap = LinearSegmentedColormap.from_list("fire_cmap", ['#000000', '#4D1A00', '#993500', '#E55100', '#FFC399'])

        power = random.uniform(2.5, 4.0)
        y_speed = 3.0 / num_frames # Upward drift speed

        print(f"\n--- Generating Animation #{anim_id}: {theme.title()} ---")
        for i in range(num_frames):
            t = i / num_frames
            grid = self._get_grid(zoom=1.5, center_y=-1.0 + i * y_speed)
            warp_phase = t * 2 * np.pi * 3 # Flickering speed

            iterations = self.math_engine.hybrid_mandelbrot_formula(grid, 80, power, 10.0, warp_phase)

            # Normalize and color
            data = np.log1p(iterations)
            data = (data - data.min()) / (data.max() - data.min())

            fig, ax = plt.subplots(figsize=(6,6), dpi=100)
            ax.imshow(data, cmap=colormap, origin='lower')
            ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            plt.close(fig)
            print(f"\rSaved frame {i+1}/{num_frames}", end="")
        self._compile_gif(frames_dir, anim_id, theme)

    def create_rain_animation(self, anim_id, num_frames=100):
        theme = "rain"
        frames_dir = os.path.join(self.output_dir, f"frames_{theme}_{anim_id}")
        os.makedirs(frames_dir, exist_ok=True)
        colormap = LinearSegmentedColormap.from_list("rain_cmap", ['#020111', '#3a3a52', '#8b8589', '#d9d9d9'])

        # Create a static "stormy sky" background
        sky_grid = self._get_grid(zoom=1.8)
        c = complex(random.uniform(-1, 1), random.uniform(-1, 1))
        sky_iterations, _ = self.math_engine.julia_set_formula(sky_grid, 50, c, 15.0, 0.0)
        sky_data = np.log1p(sky_iterations)
        sky_data = (sky_data - sky_data.min()) / (sky_data.max() - sky_data.min())

        print(f"\n--- Generating Animation #{anim_id}: {theme.title()} ---")
        for i in range(num_frames):
            t = i / num_frames
            # Create the rain "shader"
            y_coords = np.linspace(0, 1, self.h)[:, np.newaxis]
            rain_streaks = (y_coords * 15.0 + t * 20.0) % 1.0 # Moving vertical lines
            rain_shader = np.power(rain_streaks, 8) # Make streaks thin

            # <<< FIX: Use standard multiplication instead of in-place to allow broadcasting >>>
            rain_shader = rain_shader * (1.0 - sky_data) # Rain is heavier in dark clouds

            final_data = 0.8 * sky_data + 0.2 * rain_shader

            fig, ax = plt.subplots(figsize=(6,6), dpi=100)
            ax.imshow(final_data, cmap=colormap, origin='lower')
            ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            plt.close(fig)
            print(f"\rSaved frame {i+1}/{num_frames}", end="")
        self._compile_gif(frames_dir, anim_id, theme)

    def create_particle_flow_animation(self, anim_id, num_frames=120):
        theme = "particle_flow"
        frames_dir = os.path.join(self.output_dir, f"frames_{theme}_{anim_id}")
        os.makedirs(frames_dir, exist_ok=True)
        colormap = LinearSegmentedColormap.from_list("flow_cmap", ['#000000', '#100C2A', '#3C1053'])

        # 1. Generate the static vector field from a fractal
        c = complex(random.uniform(-1, 1), random.uniform(-1, 1))
        grid = self._get_grid()
        _, field_lines = self.math_engine.julia_set_formula(grid, 50, c, 5.0, 0.0)

        # Normalize the vector field
        angles = np.angle(field_lines)
        vx = np.cos(angles) * 0.01 # x-component of flow
        vy = np.sin(angles) * 0.01 # y-component of flow

        # Background image
        bg_data = np.log1p(np.abs(field_lines))
        bg_data = (bg_data - bg_data.min()) / (bg_data.max() - bg_data.min())

        # 2. Initialize particles
        num_particles = 2000
        particles = np.random.rand(num_particles, 2) # Positions from 0.0 to 1.0

        print(f"\n--- Generating Animation #{anim_id}: {theme.title()} ---")
        for i in range(num_frames):
            # 3. Update particle positions
            px_idx = (particles[:, 0] * (self.w - 1)).astype(int)
            py_idx = (particles[:, 1] * (self.h - 1)).astype(int)

            particles[:, 0] += vx[py_idx, px_idx]
            particles[:, 1] += vy[py_idx, px_idx]

            # Re-spawn particles that go off-screen
            off_screen = (particles < 0) | (particles > 1)
            particles[off_screen] = np.random.rand(np.sum(off_screen))

            # 4. Render
            fig, ax = plt.subplots(figsize=(6,6), dpi=100)
            ax.imshow(bg_data, cmap=colormap, origin='lower')
            # Convert particle positions from [0,1] to pixel coordinates for scatter plot
            ax.scatter(particles[:, 0]*self.w, particles[:, 1]*self.h, s=1, c='white', alpha=0.8)
            ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            plt.close(fig)
            print(f"\rSaved frame {i+1}/{num_frames}", end="")
        self._compile_gif(frames_dir, anim_id, theme)

    def create_cosmic_nebula_animation(self, anim_id, num_frames=120):
        theme = "cosmic_nebula"
        frames_dir = os.path.join(self.output_dir, f"frames_{theme}_{anim_id}")
        os.makedirs(frames_dir, exist_ok=True)
        colormap = LinearSegmentedColormap.from_list("cosmic_cmap", ['#000000', '#2c1810', '#8b4513', '#daa520', '#ffd700', '#ffffff'])

        c = complex(random.uniform(-0.8, 0.8), random.uniform(-0.2, 0.2))
        rotation_speed = 360.0 / num_frames

        # Static starfield for parallax
        num_stars = 500
        star_pos = np.random.rand(num_stars, 2) * self.w # Positions in pixel coords
        star_brightness = np.random.rand(num_stars) * 0.5 + 0.5

        print(f"\n--- Generating Animation #{anim_id}: {theme.title()} ---")
        for i in range(num_frames):
            t = i / num_frames
            grid = self._get_grid(rotation_angle = i * rotation_speed)
            warp_phase = t * 2 * np.pi # Internal churning

            iterations, _ = self.math_engine.julia_set_formula(grid, 100, c, 8.0, warp_phase)

            data = np.log1p(iterations)
            data = np.power((data - data.min()) / (data.max() - data.min()), 0.5)

            fig, ax = plt.subplots(figsize=(6,6), dpi=100)
            ax.imshow(data, cmap=colormap, origin='lower')

            # Twinkling stars
            twinkle = np.random.rand(num_stars) * 0.5 + 0.5
            ax.scatter(star_pos[:,0], star_pos[:,1], s=0.5, c='white', alpha=star_brightness*twinkle)

            ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            plt.close(fig)
            print(f"\rSaved frame {i+1}/{num_frames}", end="")
        self._compile_gif(frames_dir, anim_id, theme)

# --- Main Execution ---
if __name__ == "__main__":
    # The math_engine holds the core fractal formulas
    math_engine = FractalMathEngine(width=512, height=512)

    # The main output directory for all animations
    output_dir = f"thematic_animations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # The animation_engine uses the math to build themed animations
    animation_engine = ThematicAnimationEngine(math_engine, output_dir)

    # --- Generate one of each animation ---
    # Each call will produce a completely unique, randomized version of the theme.
    animation_engine.create_fire_animation(anim_id=1, num_frames=100)
    animation_engine.create_rain_animation(anim_id=2, num_frames=120)
    animation_engine.create_particle_flow_animation(anim_id=3, num_frames=120)
    animation_engine.create_cosmic_nebula_animation(anim_id=4, num_frames=100)

    print(f"\nFinished! All animations saved to the '{output_dir}' directory.")
