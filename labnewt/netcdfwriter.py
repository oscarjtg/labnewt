import os
import time
from typing import Optional, Sequence

import netCDF4 as nc
import numpy as np


class NetCDFWriter:
    def __init__(
        self,
        fields: Sequence[str],
        path: str,
        interval: int,
        on_init: Optional[bool] = True,
        zlib: Optional[bool] = True,
        complevel: Optional[int] = 4,
        shuffle: Optional[bool] = True,
    ):
        """
        Pass into a `Simulation` object to save model array to a NetCDF file.

        Parameters
        ----------
        fields : Sequence[str]
            List of strings giving labels for the fields to be saved.
        path : str
            String giving the full or relative path to the NetCDF file.
        interval : int
            Integer number of timesteps between each save.
        on_init : bool, optional
            If `True`, saves the fields after initialisation.
        zlib : bool, optional
            If `True`, enables compression.
        complevel : int, optional
            Integer between 0-9 giving the compression level.
        shuffle : bool, optional
            If `True`, enables bit shuffling for more compression.

        Returns
        -------
        None
        """
        self.fields = fields
        self.path = path
        self.interval = int(interval)
        self.on_init = on_init
        self.zlib = zlib
        assert (
            0 <= complevel & complevel <= 9
        ), "Invalid complevel (should be between 0-9, inclusive)."
        self.complevel = int(complevel)
        self.shuffle = shuffle
        self._file_open = False
        self._file = None
        self._arrays = None

    def __call__(self, model):
        if not self._file_open:
            self._open_file()
            self._create_file_dimensions(model)
            self._create_file_variables(model)
            self._add_file_info()
            self._add_xy_data_to_file(model)
        self._add_data_to_file(model)

    def close(self):
        self._close_file()

    def _open_file(self):
        dir = os.path.dirname(self.path)
        if dir:
            os.makedirs(dir, exist_ok=True)
        self._file = nc.Dataset(self.path, "w", format="NETCDF4")
        self._file_open = True

    def _close_file(self):
        if not self._file_open:
            return
        self._file.close()
        self._file_open = False
        self._file = None

    def _parse_field_label(self, label, model):
        parts = label.split(".")
        field = model
        for part in parts:
            field = getattr(field, part)
        return field

    def _create_file_dimensions(self, model):
        self._file.createDimension("time", None)
        self._file.createDimension("q", model.stencil.nq)
        self._file.createDimension("y", model.ny)
        self._file.createDimension("x", model.nx)

    def _create_file_variables(self, model):
        self._file.createVariable("time", "f8", ("time",))
        self._file.createVariable("q", "i8", ("q",))
        self._file.createVariable("y", "f8", ("y",))
        self._file.createVariable("x", "f8", ("x",))
        for label in self.fields:
            data = self._parse_field_label(label, model)
            if data.ndim == 2:
                self._file.createVariable(
                    label,
                    "f8",
                    (
                        "time",
                        "y",
                        "x",
                    ),
                    zlib=self.zlib,
                    complevel=self.complevel,
                    shuffle=self.shuffle,
                )
            elif data.ndim == 3:
                self._file.createVariable(
                    label,
                    "f8",
                    (
                        "time",
                        "q",
                        "y",
                        "x",
                    ),
                    zlib=self.zlib,
                    complevel=self.complevel,
                    shuffle=self.shuffle,
                )

    def _add_file_info(self):
        self._file.description = "Lattice Boltzmann model run"
        self._file.history = "Created " + time.ctime(time.time())
        self._file.source = "LaBNeWT: Lattice Boltzmann Numerical Wave Tank"

    def _add_xy_data_to_file(self, model):
        self._file.variables["q"][:] = np.arange(model.stencil.nq)
        self._file.variables["y"][:] = model.y
        self._file.variables["x"][:] = model.x

    def _add_data_to_file(self, model):
        times = self._file.variables["time"]
        next_time_index = len(times)
        times[next_time_index] = model.clock
        for label in self.fields:
            data = self._parse_field_label(label, model)
            var = self._file.variables[label]
            if data.ndim == 2:
                var[next_time_index, :, :] = data
            elif data.ndim == 3:
                var[next_time_index, :, :, :] = data
