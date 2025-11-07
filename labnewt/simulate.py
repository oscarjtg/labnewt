"""Class that manages simulation."""

import time


class Simulation:
    def __init__(self, model, stop_time):
        self.model = model
        self.stop_time = stop_time
        self.clock = 0.0

    def run(self, print_progress=True):
        # Initialise model, and time it.
        start0 = time.perf_counter()
        self.model._initialise()
        end0 = time.perf_counter()
        if print_progress:
            print(f"Model initialisation complete in {end0 - start0:.6f} seconds")

        # Run first time step of model, and time it.
        start1 = time.perf_counter()
        self.model._step()
        self.clock += self.model.dt
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
        timestep = 0

        # Run model to completion.
        start2 = time.perf_counter()
        start = time.perf_counter()
        while self.clock < self.stop_time:
            self.model._step()
            self.clock += self.model.dt
            timestep += 1
            if not print_progress:
                continue
            if timestep % n_timesteps_10percent == 0:
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
