import importlib
import numpy as np
from types import SimpleNamespace

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

        self.ext = importlib.import_module('qcos_ext')
        self._solver = None

    def update_settings(self, **kwargs):
        assert self.settings is not None

        settings_changed = False

        for k in self.ext.QCOSSettings.__dict__:
            if not k.startswith('__'):
                if k in kwargs:
                    setattr(self.settings, k, kwargs[k])
                    settings_changed = True

        if settings_changed and self._solver is not None:
            self._solver.update_settings(self.settings)

    def setup(self, n, m, p, P, c, A, b, G, h, l, nsoc, q, **settings):
        self.m = m
        self.n = n
        self.p = p
        self.P = self.ext.CSC(P.astype(np.float64))
        self.c = c.astype(np.float64)
        self.A = self.ext.CSC(A.astype(np.float64))
        self.b = b.astype(np.float64)
        self.G = self.ext.CSC(G.astype(np.float64))
        self.h = h.astype(np.float64)
        self.l = l
        self.nsoc = nsoc
        self.q = q.astype(np.int32)
        self.settings = self.ext.QCOSSettings()
        self.ext.set_default_settings(self.settings)
        self.update_settings(**settings)

        self._solver = self.ext.QCOSSolver(
            self.n,
            self.m,
            self.p,
            self.P,
            self.c,
            self.A,
            self.b,
            self.G,
            self.h,
            self.l,
            self.nsoc,
            self.q,
            self.settings
        )

    def solve(self):
        self._solver.solve()

        results = SimpleNamespace(x=self._solver.solution.x, s=self._solver.solution.s, y=self._solver.solution.y, z=self._solver.solution.z)
        return results

        # Handle nonconvergence.