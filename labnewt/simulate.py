"""Class that manages simulation."""

import os
import time

import numpy as np

from labnewt.diagnostics import relative_error


class Simulation:
    def __init__(self, model, stop_time):
        self.model = model
        self.stop_time = stop_time
        self.clock = 0.0
        self.timestep = 0

    def run(self, print_progress=True, save_frames=False):
        if save_frames:
            os.makedirs("./frames", exist_ok=True)
        # Initialise model, and time it.
        start0 = time.perf_counter()
        if not self.model.initialised:
            self.model._initialise()
        if save_frames:
            self.model.plot_fields(f"./frames/frame_{self.timestep:04d}")
        end0 = time.perf_counter()
        if print_progress:
            print(f"Model initialisation complete in {end0 - start0:.6f} seconds")

        # Run first time step of model, and time it.
        start1 = time.perf_counter()
        self.model._step()
        self.clock += self.model.dt
        self.timestep += 1
        if save_frames:
            self.model.plot_fields(f"./frames/frame_{self.timestep:04d}")
        end1 = time.perf_counter()
        time_taken = end1 - start1
        if print_progress:
            print(f"Model first time step complete in {time_taken:.6f} seconds")

        # Estimate model run time and print it.
        if print_progress:
            total_time = (self.stop_time / self.model.dt) * time_taken
            print(f"Estimated time to completion: {total_time:.2f} seconds")

        n_timesteps = int(self.stop_time / self.model.dt)
        n_timesteps_10percent = int(n_timesteps / 10)

        # Run model to completion.
        start2 = time.perf_counter()
        start = time.perf_counter()
        while self.clock < self.stop_time:
            self.model._step()
            self.clock += self.model.dt
            self.timestep += 1
            if save_frames and self.timestep % 10 == 0:
                self.model.plot_fields(f"./frames/frame_{self.timestep:04d}")
            if not print_progress:
                continue
            if self.timestep % n_timesteps_10percent == 0:
                end = time.perf_counter()
                percent_complete = self.clock / self.stop_time * 100
                string1 = f"Model completion: {percent_complete:.1f}%."
                string2 = f"Time since last checkpoint: {end - start:.2f} seconds."
                print(string1 + "  " + string2)
                start = time.perf_counter()

        end2 = time.perf_counter()
        bulk_time_taken = end2 - start2
        total_time_taken = end2 - start0
        if print_progress:
            print("--------------------------------------------------")
            print("Model completed")
            print(f"Model bulk time to complete:  {bulk_time_taken:.2f} seconds")
            print(f"Model total time to complete: {total_time_taken:.2f} seconds")
            print("--------------------------------------------------")

    def run_to_steady_state(self, max_timesteps, rtol=1.0e-05, print_progress=True):
        # Initialise model and time it
        start0 = time.perf_counter()
        if not self.model.initialised:
            self.model._initialise()
        end0 = time.perf_counter()
        if print_progress:
            print(f"Model initialisation complete in {end0 - start0:.6f} seconds")

        # Run first time step of model, and time it.
        start1 = time.perf_counter()
        self.model._step()
        self.clock += self.model.dt
        self.timestep += 1
        end1 = time.perf_counter()
        time_taken = end1 - start1
        if print_progress:
            print(f"Model first time step complete in {time_taken:.6f} seconds")

        # Run model to completion.
        converged = False
        start2 = time.perf_counter()
        while self.timestep < max_timesteps:
            fi0 = np.copy(self.model.fi)
            self.model._step()
            self.clock += self.model.dt
            self.timestep += 1
            err = relative_error(self.model.fi, fi0)
            if err < rtol:
                converged = True
                break
            else:
                fi0[:] = self.model.fi

        end2 = time.perf_counter()
        bulk_time_taken = end2 - start2
        total_time_taken = end2 - start0
        if print_progress:
            print("--------------------------------------------------")
            print("Model completed")
            if converged:
                print(f"Model converged after {self.timestep} timesteps.")
            else:
                msg = (
                    f"WARNING: model did NOT converge after {self.timestep} timesteps!"
                )
                print(msg)
            print(f"Model bulk time to complete:  {bulk_time_taken:.2f} seconds")
            print(f"Model total time to complete: {total_time_taken:.2f} seconds")
            print("--------------------------------------------------")
