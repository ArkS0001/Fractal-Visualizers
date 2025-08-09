import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numba import jit, prange
import random
import os
import glob
from PIL import Image
from datetime import datetime

# --- Main Generator Class (Handles the math) ---
class EndlessFractalGenerator:
    def __init__(self, width=512, height=512): # Smaller resolution for faster animation frames
        self.width = width
        self.height = height
        self.formulas = [self.julia_set_formula, self.hybrid_mandelbrot_formula, self.phoenix_julia_formula]

    # --- Numba-Accelerated Fractal Formulas ---
    @staticmethod
    @jit(nopython=True, parallel=True)
    def julia_set_formula(grid, max_iter, power, c, warp_freq, color_offset):
        iterations = np.zeros(grid.shape, dtype=np.float64)
        orbit_trap = np.full(grid.shape, 1e6, dtype=np.float64)
        field_lines = np.zeros(grid.shape, dtype=np.complex128)
        for i in prange(grid.shape[0]):
            for j in prange(grid.shape[1]):
                z = grid[i, j]
                if warp_freq > 0:
                    z += 0.1 * np.sin(z.real*warp_freq) + 0.1j*np.cos(z.imag*warp_freq)
                for n in range(max_iter):
                    if np.abs(z) > 4.0:
                        log_zn=np.log(np.abs(z))/2.; nu=np.log(log_zn/np.log(2.))/np.log(2.)
                        iterations[i,j]=n-nu+1; break
                    z=z**power+c
                    orbit_trap[i,j]=min(orbit_trap[i,j],np.abs(z)); field_lines[i,j]=z
                else: iterations[i,j]=max_iter
        return iterations, orbit_trap, field_lines, color_offset

    @staticmethod
    @jit(nopython=True, parallel=True)
    def hybrid_mandelbrot_formula(grid, max_iter, power, warp_freq, color_offset):
        # Note: Mandelbrot animations are typically just zooms, as 'c' is the grid itself.
        iterations = np.zeros(grid.shape, dtype=np.float64)
        orbit_trap = np.full(grid.shape, 1e6, dtype=np.float64)
        field_lines = np.zeros(grid.shape, dtype=np.complex128)
        for i in prange(grid.shape[0]):
            for j in prange(grid.shape[1]):
                c = grid[i, j]
                z = 0.0j
                if warp_freq > 0:
                    c += 0.1*np.sin(c.real*warp_freq) + 0.1j*np.cos(c.imag*warp_freq)
                for n in range(max_iter):
                    if np.abs(z)>4.0:
                        log_zn=np.log(np.abs(z))/2.; nu=np.log(log_zn/np.log(2.))/np.log(2.)
                        iterations[i,j]=n-nu+1; break
                    z=(z.real - 1j*np.abs(z.imag))**power + c
                    orbit_trap[i,j]=min(orbit_trap[i,j],np.abs(z)); field_lines[i,j]=z
                else: iterations[i,j]=max_iter
        return iterations, orbit_trap, field_lines, color_offset
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def phoenix_julia_formula(grid, max_iter, power, c, p, warp_freq, color_offset):
        iterations=np.zeros(grid.shape,dtype=np.float64)
        orbit_trap=np.full(grid.shape,1e6,dtype=np.float64)
        field_lines=np.zeros(grid.shape,dtype=np.complex128)
        for i in prange(grid.shape[0]):
            for j in prange(grid.shape[1]):
                z=grid[i,j]; z_prev=0.0j
                if warp_freq>0: z+=0.1*np.sin(z.real*warp_freq)+0.1j*np.cos(z.imag*warp_freq)
                for n in range(max_iter):
                    if np.abs(z)>4.0:
                        log_zn=np.log(np.abs(z))/2.; nu=np.log(log_zn/np.log(2.))/np.log(2.)
                        iterations[i,j]=n-nu+1; break
                    z_new=z**power+c+p*z_prev; z_prev=z; z=z_new
                    orbit_trap[i,j]=min(orbit_trap[i,j],np.abs(z)); field_lines[i,j]=z
                else: iterations[i,j]=max_iter
        return iterations, orbit_trap, field_lines, color_offset


# --- Animation Engine (Handles dynamics and rendering) ---
class FractalAnimationEngine:
    def __init__(self, generator, output_dir, animation_id, num_frames=100):
        self.generator = generator
        self.animation_id = animation_id
        self.num_frames = num_frames
        self.frames_dir = os.path.join(output_dir, f"anim_{animation_id:02d}_frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        # --- Define the "Physics" for this unique animation ---
        self.formula = random.choice(self.generator.formulas)
        self.coloring = random.choice(['smooth', 'orbit_trap', 'field_lines'])
        self.colormap = self.create_random_palette()

        # Initial fractal parameters
        self.power = random.uniform(1.8, 3.2)
        
        # Motion paths for parameters over time (t from 0 to 1)
        self.c_start = complex(random.uniform(-1,1), random.uniform(-1,1))
        self.c_radius = random.uniform(0.05, 0.2)
        self.c_speed_x = random.uniform(0.5, 2.0)
        self.c_speed_y = random.uniform(0.5, 2.0)

        self.p_start = complex(random.uniform(-0.6, -0.4), random.uniform(0.4, 0.6))
        self.p_radius = random.uniform(0.01, 0.05)
        
        # Zoom dynamics
        self.zoom_start = 2.2
        self.zoom_end = random.uniform(0.1, 1.5)
        
        # Warp dynamics
        self.warp_start = random.uniform(2.0, 15.0) if random.random() > 0.3 else 0.0
        self.warp_pulse_amp = random.uniform(0.5, 2.0)
        self.warp_pulse_speed = random.uniform(1.0, 4.0)
        
        print(f"\n--- Animation #{animation_id} Initialized ---")
        print(f"Formula: {self.formula.__name__}, Coloring: {self.coloring}")
        
    def get_params_at_time(self, t):
        """Calculates all fractal parameters for a given time t (0 to 1)."""
        params = {'power': self.power}
        
        # Evolving 'c' value (Julia and Phoenix)
        angle_x = 2 * np.pi * self.c_speed_x * t
        angle_y = 2 * np.pi * self.c_speed_y * t
        params['c'] = self.c_start + self.c_radius * (np.cos(angle_x) + 1j * np.sin(angle_y))

        # Evolving 'p' value (Phoenix only)
        if self.formula.__name__ == 'phoenix_julia_formula':
            params['p'] = self.p_start + self.p_radius * (np.cos(angle_y) + 1j * np.sin(angle_x))
            
        # Evolving Domain Warp frequency
        params['warp_freq'] = self.warp_start + self.warp_pulse_amp * np.sin(2 * np.pi * self.warp_pulse_speed * t)
        
        # Evolving color offset
        params['color_offset'] = t
        
        # Evolving zoom
        zoom = self.zoom_start * (self.zoom_end / self.zoom_start)**t
        
        return params, zoom

    def create_animation(self, max_iter=150):
        print(f"Generating {self.num_frames} frames...")
        for i in range(self.num_frames):
            t = i / self.num_frames
            params, zoom = self.get_params_at_time(t)
            
            # Create coordinate grid based on zoom
            x_range = np.linspace(-zoom, zoom, self.generator.width)
            y_range = np.linspace(-zoom, zoom, self.generator.height)
            grid = np.meshgrid(x_range, y_range)
            grid = grid[0] + 1j*grid[1]

            # Generate fractal data
            iterations, orbit_trap, field_lines, color_offset = self.formula(grid, max_iter, **params)
            
            # --- Coloring ---
            if self.coloring == 'orbit_trap': raw_colors = np.power(orbit_trap, 0.25)
            elif self.coloring == 'field_lines':
                angle=np.angle(field_lines)/(2*np.pi); magnitude=np.log1p(np.abs(field_lines))
                raw_colors = angle + 0.1*magnitude
            else: raw_colors = np.log1p(iterations)
            
            # Apply color cycling
            raw_colors = (raw_colors + color_offset) % 1.0

            # Normalize and save frame
            min_v, max_v = raw_colors.min(), raw_colors.max()
            if max_v - min_v > 1e-9: final_colors = (raw_colors - min_v)/(max_v-min_v)
            else: final_colors = np.zeros_like(raw_colors)

            fig, ax = plt.subplots(figsize=(6,6), dpi=100) # Smaller fig size for speed
            ax.imshow(final_colors, cmap=self.colormap, origin='lower')
            ax.axis('off')
            plt.tight_layout(pad=0)
            frame_path = os.path.join(self.frames_dir, f"frame_{i:04d}.png")
            plt.savefig(frame_path)
            plt.close(fig)

            print(f"\rSaved frame {i+1}/{self.num_frames}", end="")
        
        print("\nFrames generated. Compiling GIF...")
        self.compile_gif()

    def compile_gif(self):
        """Finds all generated frames and compiles them into a GIF."""
        frame_files = sorted(glob.glob(os.path.join(self.frames_dir, "*.png")))
        frames = [Image.open(f) for f in frame_files]
        
        output_path = os.path.join(os.path.dirname(self.frames_dir), f"animation_{self.animation_id:02d}.gif")

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=40,  # Milliseconds per frame (e.g., 40ms = 25 FPS)
            loop=0  # 0 means loop forever
        )
        print(f"GIF saved to: {output_path}")

    def create_random_palette(self):
        colors = [[random.random() for _ in range(3)] for _ in range(random.randint(3,5))]
        if random.random() > 0.5: colors.insert(0, [0,0,0])
        else: colors.append([1,1,1])
        return LinearSegmentedColormap.from_list("random_cmap", colors)

# --- Main Execution ---
if __name__ == "__main__":
    NUMBER_OF_ANIMATIONS = 2 # How many different GIFs to create
    FRAMES_PER_ANIMATION = 120 # Higher number = smoother but slower animation
    
    # The generator holds the math
    fractal_generator = EndlessFractalGenerator(width=512, height=512)
    
    # The main output directory for all animations
    output_directory = f"animated_fractals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Starting generation of {NUMBER_OF_ANIMATIONS} unique fractal animations...")
    
    for i in range(NUMBER_OF_ANIMATIONS):
        # The engine defines the "physics" for one animation
        engine = FractalAnimationEngine(fractal_generator, output_directory, i + 1, num_frames=FRAMES_PER_ANIMATION)
        engine.create_animation(max_iter=150)
        
    print(f"\nFinished! All animations saved to the '{output_directory}' directory.")
