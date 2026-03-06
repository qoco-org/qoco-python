# Copyright (c) 2024, Govind M. Chari <govindchari1@gmail.com>
# This source code is licensed under the BSD 3-Clause License

import importlib
import numpy as np
from scipy import sparse
from types import SimpleNamespace

ALGEBRAS = (
    "cuda",
    "builtin",
)

ALGEBRA_MODULES = {
    "cuda": "qoco_cuda",
    "builtin": "qoco.qoco_ext",
}


def algebra_available(algebra):
    assert algebra in ALGEBRAS, f"Unknown algebra {algebra}"
    module = ALGEBRA_MODULES[algebra]

    try:
        importlib.import_module(module)
    except ImportError:
        return False
    else:
        return True


def algebras_available():
    return [algebra for algebra in ALGEBRAS if algebra_available(algebra)]


class QOCO:
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

        self.Psp = None
        self.c = None
        self.Asp = None
        self.b = None
        self.Gsp = None
        self.h = None
        self.l = None
        self.nsoc = None
        self.q = None

        self.solvecodes = [
            "QOCO_UNSOLVED",
            "QOCO_SOLVED",
            "QOCO_SOLVED_INACCURATE",
            "QOCO_NUMERICAL_ERROR",
            "QOCO_MAX_ITER",
        ]

        self.algebra = kwargs.pop("algebra") if "algebra" in kwargs else "builtin"
        if not algebra_available(self.algebra):
            raise RuntimeError(f"Algebra {self.algebra} not available")
        self.ext = importlib.import_module(ALGEBRA_MODULES[self.algebra])
        self._solver = None

    def update_settings(self, **kwargs):
        assert self.settings is not None

        settings_changed = False

        for k in self.ext.QOCOSettings.__dict__:
            if not k.startswith("__"):
                if k in kwargs:
                    setattr(self.settings, k, kwargs[k])
                    settings_changed = True

        if settings_changed and self._solver is not None:
            self._solver.update_settings(self.settings)

    def update_vector_data(self, c=None, b=None, h=None):
        """
        Update data vectors.
        
        Parameters
        ----------
        c : np.ndarray, optional
            New c vector of size n. If None, c is not updated. Default is None.
        b : np.ndarray, optional
            New b vector of size p. If None, b is not updated. Default is None.
        h : np.ndarray, optional
            New h vector of size m. If None, h is not updated. Default is None.
        """
        cnew_ptr = None
        bnew_ptr = None
        hnew_ptr = None
        
        if c is not None:
            if not isinstance(c, np.ndarray):
                c = np.array(c)
            c = c.astype(np.float64)
            if c.shape[0] != self.n:
                raise ValueError(f"c size must be n = {self.n}")
            cnew_ptr = c
        
        if b is not None:
            if not isinstance(b, np.ndarray):
                b = np.array(b)
            b = b.astype(np.float64)
            if b.shape[0] != self.p:
                raise ValueError(f"b size must be p = {self.p}")
            bnew_ptr = b
        
        if h is not None:
            if not isinstance(h, np.ndarray):
                h = np.array(h)
            h = h.astype(np.float64)
            if h.shape[0] != self.m:
                raise ValueError(f"h size must be m = {self.m}")
            hnew_ptr = h
        
        return self._solver.update_vector_data(cnew_ptr, bnew_ptr, hnew_ptr)

    def update_matrix_data(self, P=None, A=None, G=None):
        """
        Update sparse matrix data.
        
        The new matrices must have the same sparsity structure as the original ones.
        
        Parameters
        ----------
        P : np.ndarray, optional
            New data for P matrix (only the nonzero values). If None, P is not updated. 
            Default is None.
        A : np.ndarray, optional
            New data for A matrix (only the nonzero values). If None, A is not updated.
            Default is None.
        G : np.ndarray, optional
            New data for G matrix (only the nonzero values). If None, G is not updated.
            Default is None.
        """
        Pxnew_ptr = None
        Axnew_ptr = None
        Gxnew_ptr = None
        
        if P is not None:
            if not isinstance(P, np.ndarray):
                P = np.array(P)
            P = P.astype(np.float64)
            Pxnew_ptr = P
        
        if A is not None:
            if not isinstance(A, np.ndarray):
                A = np.array(A)
            A = A.astype(np.float64)
            Axnew_ptr = A
        
        if G is not None:
            if not isinstance(G, np.ndarray):
                G = np.array(G)
            G = G.astype(np.float64)
            Gxnew_ptr = G
        
        return self._solver.update_matrix_data(Pxnew_ptr, Axnew_ptr, Gxnew_ptr)

    def setup(self, n, m, p, P, c, A, b, G, h, l, nsoc, q, **settings):
        self.m = m
        self.n = n
        self.p = p
        self.Psp = P.astype(np.float64) if P is not None else None
        self.Asp = A.astype(np.float64) if A is not None else None
        self.Gsp = G.astype(np.float64) if G is not None else None

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
            if not isinstance(q, np.ndarray):
                q = np.array(q)
            self.q = q.astype(np.int32)
        else:
            self.q = np.zeros((0), np.int32)
        self.settings = self.ext.QOCOSettings()
        self.ext.set_default_settings(self.settings)
        self.update_settings(**settings)
        self._solver = self.ext.QOCOSolver(
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
            self.settings,
        )

    def solve(self):
        self._solver.solve()

        results = SimpleNamespace(
            x=self._solver.solution.x,
            s=self._solver.solution.s,
            y=self._solver.solution.y,
            z=self._solver.solution.z,
            iters=self._solver.solution.iters,
            setup_time_sec=self._solver.solution.setup_time_sec,
            solve_time_sec=self._solver.solution.solve_time_sec,
            obj=self._solver.solution.obj,
            pres=self._solver.solution.pres,
            dres=self._solver.solution.dres,
            gap=self._solver.solution.gap,
            status=self.solvecodes[self._solver.solution.status],
        )
        return results
