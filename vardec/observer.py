
class CoveringObserver:

    def __init__(self):
        pass

    def on_cover_init_and_ret_pi_simple(self):
        raise NotImplementedError()

    def on_cover_init_pi_complex(self):
        raise NotImplementedError()


class DummyObserver(CoveringObserver):

    def __init__(self):
        CoveringObserver.__init__(self)

    def on_cover_init_and_ret_pi_simple(self):
        pass

    def on_cover_init_pi_complex(self):
        pass
