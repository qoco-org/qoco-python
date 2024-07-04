import importlib
import numpy as np
from scipy import sparse
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

        self.solvecodes = ["QCOS_UNSOLVED", "QCOS_SOLVED", "QCOS_SOLVED_INACCURATE", "QCOS_MAX_ITER"]

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

        if P is not None:
            self.P = self.ext.CSC(sparse.triu(P, format="csc").astype(np.float64))
        else:
            self.P = self.ext.CSC(None)
        
        if c is not None:
            self.c = c.astype(np.float64)
        else:
            raise ValueError("c cannot be None")
        
        if A is not None:
            self.A = self.ext.CSC(A.astype(np.float64))
        else:
            self.A = self.ext.CSC(None)

        if b is not None:
            self.b = b.astype(np.float64)
        else:
            self.b = np.zeros((0), np.float64)

        if G is not None:
            self.G = self.ext.CSC(G.astype(np.float64))
        else:
            self.G = self.ext.CSC(None)

        if h is not None:
            self.h = h.astype(np.float64)
        else:
            self.h = np.zeros((0), np.float64)

        self.l = l
        self.nsoc = nsoc
        if q is not None:
            if not isinstance(q,np.ndarray):
                q = np.array(q)
            self.q = q.astype(np.int32)
        else:
            self.q = np.zeros((0), np.int32)
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

        results = SimpleNamespace(x=self._solver.solution.x, s=self._solver.solution.s, y=self._solver.solution.y, z=self._solver.solution.z, iters=self._solver.solution.iters, solve_time_sec=self._solver.solution.solve_time_sec, obj=self._solver.solution.obj, pres=self._solver.solution.pres, dres=self._solver.solution.dres, gap=self._solver.solution.gap, status=self.solvecodes[self._solver.solution.status])
        return results

        # Handle nonconvergence.