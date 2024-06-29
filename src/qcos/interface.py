import qcos_ext

class QCOS:
    def __init__(self, *args, **kwargs):
        self.m = None
        self.n = None
        self.p = None
        self.P = None
        self.c = None
        self.A = None
        self.b = None
        self.G = None
        self.h = None
        self.l = None
        self.nsoc = None
        self.q = None

        # self.ext = importlib.import_module(qcos_ext)