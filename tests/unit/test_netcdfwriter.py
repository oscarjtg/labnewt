import pytest

from labnewt import Model, NetCDFWriter, Simulation


def test_initialise_netcdfwriter_defaults():
    netcdfwriter = NetCDFWriter(["u", "v"], "test.nc", 20)
    assert netcdfwriter.fields == ["u", "v"]
    assert netcdfwriter.path == "test.nc"
    assert netcdfwriter.interval == 20
    assert netcdfwriter.on_init
    assert netcdfwriter.zlib
    assert netcdfwriter.complevel == 4
    assert netcdfwriter.shuffle


def test_initialise_netcdfwriter():
    fields = ["u"]
    netcdfwriter = NetCDFWriter(
        fields,
        "test2.nc",
        10,
        on_init=False,
        zlib=False,
        complevel=0,
        shuffle=False,
    )
    assert netcdfwriter.fields == fields
    assert netcdfwriter.path == "test2.nc"
    assert netcdfwriter.interval == 10
    assert not netcdfwriter.on_init
    assert not netcdfwriter.zlib
    assert netcdfwriter.complevel == 0
    assert not netcdfwriter.shuffle


def test_initialise_netcdfwriter_invalid_complevel():
    fields = ["u"]
    with pytest.raises(AssertionError) as exc_info:
        NetCDFWriter(fields, "test2.nc", 10, complevel=10)

    # Optional: check the exact error message
    assert (
        str(exc_info.value) == "Invalid complevel (should be between 0-9, inclusive)."
    )


def test_add_netcdfwriter_to_simulation():
    model = Model(10, 10, 1.0, 1.0, 0.1)
    fields = ["u", "v", "r"]
    sim = Simulation(model, 10.0)
    sim.callbacks["netcdfwriter"] = NetCDFWriter(fields, "test.nc", 5)
    assert set(sim.callbacks.keys()) == {"netcdfwriter"}


def test_file_closed_after_simulation(tmp_path):
    model = Model(2, 3, 1.0, 1.0, 0.1)
    fields = ["u", "v", "r"]
    path = tmp_path / "test.nc"
    sim = Simulation(model, 10.0)
    sim.callbacks["netcdfwriter"] = NetCDFWriter(fields, path, 5)
    sim.run(print_progress=False)
    assert not sim.callbacks["netcdfwriter"]._file_open
    assert sim.callbacks["netcdfwriter"]._file is None
