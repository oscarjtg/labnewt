from labnewt.callback import Callback


class DummyModel:
    pass


def myfunc(model):
    return 2


def test_callback_call():
    cb = Callback(myfunc, 10, True)
    model = DummyModel

    assert cb.on_init
    assert cb.interval == 10
    assert cb(model) == 2
