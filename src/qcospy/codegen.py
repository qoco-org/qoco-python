# Copyright (c) 2024, Govind M. Chari <govindchari1@gmail.com>
# This source code is licensed under the BSD 2-Clause License

import os
import shutil
import qdldl
import numpy as np
from scipy import sparse
from qcospy.codegen_utils import *


def _generate_solver(n, m, p, P, c, A, b, G, h, l, nsoc, q, output_dir, name):
    solver_dir = output_dir + "/" + name

    print("\n")
    if os.path.exists(solver_dir):
        print("Removing existing solver.")
        shutil.rmtree(solver_dir)

    print("Generating solver.")
    os.mkdir(solver_dir)
    W = sparse.identity(l)
    W = sparse.csc_matrix(W)
    for qi in q:
        Wsoc = np.ones((qi, qi), dtype=np.float64)
        Wsoc = sparse.csc_matrix(Wsoc)
        W = sparse.bmat([[W, None], [None, Wsoc]])
    W = sparse.csc_matrix(W)

    Wnnz = int((W.nnz - m) / 2 + m)
    Wnnz_cnt = 0

    # Maps sparse 1D index (1,...,m^2) of W to its sparse index (1,...,Wnnz). Note that accessing an lower triangular element of W returns -1.
    Wsparse2dense = -np.ones((m * m))
    for j in range(m):
        for i in range(m):
            if W[i, j] != 0.0 and i <= j:
                Wsparse2dense[i + j * m] = Wnnz_cnt
                Wnnz_cnt += 1

    # Get sparsity pattern of the regularized KKT matrix.
    Preg = P + sparse.identity(n) if P is not None else sparse.identity(n)
    A = A if A is not None else None
    G = G if G is not None else None
    At = A.T if A is not None else None
    Gt = G.T if G is not None else None

    K = sparse.bmat(
        [
            [Preg, At, Gt],
            [A, -sparse.identity(p), None],
            [G, None, -W - 1e3 * sparse.identity(m)],
        ]
    )
    solver = qdldl.Solver(K)
    L, D, perm = solver.factors()

    N = n + m + p
    Lidx = [False for _ in range(N**2)]
    for i in range(N):
        for j in range(N):
            if L[i, j] != 0.0:
                Lidx[j * N + i] = True

    generate_cmakelists(solver_dir)
    generate_workspace(solver_dir, n, m, p, P, c, A, b, G, h, q, L.nnz, Wnnz)
    Lsparse2dense = generate_ldl(
        solver_dir, n, m, p, P, A, G, perm, Lidx, Wsparse2dense
    )
    generate_cone(solver_dir, m, Wnnz, Wsparse2dense)
    generate_kkt(solver_dir, n, m, p, P, c, A, b, G, h, perm, Wsparse2dense)
    generate_utils(solver_dir, n, m, p, P, c, A, b, G, h, l, nsoc, q, Wsparse2dense)
    generate_solver(solver_dir, m, Wsparse2dense)
    generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q)


def generate_cmakelists(solver_dir):
    f = open(solver_dir + "/CMakeLists.txt", "a")
    f.write("cmake_minimum_required(VERSION 3.18)\n")
    f.write("project(qcos_custom)\n\n")
    f.write("if(ENABLE_PRINTING)\n")
    f.write("   add_compile_definitions(ENABLE_PRINTING)\n")
    f.write("endif()\n\n")
    f.write("if(QCOS_CUSTOM_BUILD_TYPE STREQUAL Debug)\n")
    f.write("   set(QCOS_CUSTOM_BUILD_TYPE Debug)\n")
    f.write(
        '   set(CMAKE_C_FLAGS "-g -march=native -Werror -Wall -Wextra -fsanitize=address,undefined")\n'
    )
    f.write("else()\n")
    f.write("   set(QCOS_CUSTOM_BUILD_TYPE Release)\n")
    f.write('   set(CMAKE_C_FLAGS "-O3 -march=native -Wall -Wextra")\n')
    f.write("endif()\n")
    f.write('message(STATUS "Build Type: " ${QCOS_CUSTOM_BUILD_TYPE})\n')
    f.write('message(STATUS "Build Flags: " ${CMAKE_C_FLAGS})\n')

    f.write('set(CMAKE_C_FLAGS "-O3 -march=native -Wall -Wextra")\n')
    f.write("# Build qcos_custom shared library.\n")
    f.write("add_library(qcos_custom SHARED)\n")
    f.write(
        "target_sources(qcos_custom PRIVATE qcos_custom.c cone.c utils.c ldl.c kkt.c)\n\n"
    )
    f.write("target_link_libraries(qcos_custom m)\n")
    f.write("# Build qcos demo.\n")
    f.write("add_executable(runtest runtest.c)\n")
    f.write("target_link_libraries(runtest qcos_custom)\n")
    f.close()


def generate_workspace(solver_dir, n, m, p, P, c, A, b, G, h, q, Lnnz, Wnnz):
    f = open(solver_dir + "/workspace.h", "a")
    write_license(f)
    f.write("#ifndef WORKSPACE_H\n")
    f.write("#define WORKSPACE_H\n\n")

    f.write("typedef struct {\n")
    f.write("   int max_iters;\n")
    f.write("   int bisection_iters;\n")
    f.write("   double kkt_reg;\n")
    f.write("   double eabs;\n")
    f.write("   double erel;\n")
    f.write("   unsigned char solved;\n")
    f.write("   unsigned char verbose;\n")
    f.write("} Settings;\n\n")

    f.write("typedef struct {\n")
    f.write("   double x[%i];\n" % (n))
    f.write("   double s[%i];\n" % (m))
    f.write("   double y[%i];\n" % (p))
    f.write("   double z[%i];\n" % (m))
    f.write("   int iters;\n")
    f.write("   double obj;\n")
    f.write("   double pres;\n")
    f.write("   double dres;\n")
    f.write("   double gap;\n")
    f.write("   unsigned char solved;\n")
    f.write("} Solution;\n\n")

    Pnnz = len(P.data) if P is not None else 0
    Annz = len(A.data) if A is not None else 0
    Gnnz = len(G.data) if G is not None else 0
    qmax = max(q) if len(q) > 0 else 0

    f.write("typedef struct {\n")
    f.write("   int n;\n")
    f.write("   int m;\n")
    f.write("   int p;\n")
    f.write("   double P[%i];\n" % (Pnnz))
    f.write("   double c[%i];\n" % (n))
    f.write("   double A[%i];\n" % (Annz))
    f.write("   double b[%i];\n" % (p))
    f.write("   double G[%i];\n" % (Gnnz))
    f.write("   double h[%i];\n" % (m))
    f.write("   int l;\n")
    f.write("   int nsoc;\n")
    f.write("   int q[%i];\n" % (len(q)))
    f.write("   double x[%i];\n" % (n))
    f.write("   double s[%i];\n" % (m))
    f.write("   double y[%i];\n" % (p))
    f.write("   double z[%i];\n" % (m))
    f.write("   double L[%i];\n" % (Lnnz))
    f.write("   double D[%i];\n" % (n + m + p))
    f.write("   double W[%i];\n" % Wnnz)
    f.write("   double lambda[%i];\n" % m)
    f.write("   double xbuff[%i];\n" % n)
    f.write("   double ubuff1[%i];\n" % m)
    f.write("   double ubuff2[%i];\n" % m)
    f.write("   double ubuff3[%i];\n" % m)
    f.write("   double Ds[%i];\n" % m)
    f.write("   double Winv[%i];\n" % Wnnz)
    f.write("   double WtW[%i];\n" % Wnnz)
    f.write("   double kkt_rhs[%i];\n" % (n + m + p))
    f.write("   double kkt_res[%i];\n" % (n + m + p))
    f.write("   double xyz[%i];\n" % (n + m + p))
    f.write("   double xyzbuff[%i];\n" % (n + m + p))
    f.write("   double sbar[%i];\n" % qmax)
    f.write("   double zbar[%i];\n" % qmax)
    f.write("   double mu;\n")
    f.write("   double sigma;\n")
    f.write("   double a;\n\n")
    f.write("   Settings settings;\n")
    f.write("   Solution sol;\n")
    f.write("} Workspace;\n\n")

    f.write("#endif")
    f.close()


def generate_ldl(solver_dir, n, m, p, P, A, G, perm, Lidx, Wsparse2dense):
    f = open(solver_dir + "/ldl.h", "a")
    write_license(f)
    f.write("#ifndef LDL_H\n")
    f.write("#define LDL_H\n\n")
    f.write('#include "workspace.h"\n\n')

    f.write("void ldl(Workspace* work);\n")
    f.write("void tri_solve(Workspace* work);\n")
    f.write("#endif")
    f.close()

    f = open(solver_dir + "/ldl.c", "a")
    write_license(f)
    f.write('#include "workspace.h"\n\n')
    f.write("void ldl(Workspace* work){\n")
    N = n + m + p

    # Maps sparse 1D index (1,...,N^2) of L to its sparse index (1,...,Lnnz).
    Lsparse2dense = -np.ones(N**2)

    # Number of nonzeros of L added (Used to get sparse index of the current element under consideration).
    Lnnz = 0

    # The factorization will only access strictly lower triangular elements of L.
    for j in range(N):
        # D update.
        f.write("   work->D[%d] = " % j)
        write_Kelem(f, j, j, n, m, p, P, A, G, perm, Wsparse2dense, True)
        for k in range(j):
            if Lidx[k * N + j]:
                f.write(" - work->D[%i] * " % k)
                f.write(
                    "work->L[%i] * work->L[%i]"
                    % (Lsparse2dense[k * N + j], Lsparse2dense[k * N + j])
                )
        f.write(";\n")

        # L update.
        for i in range(j + 1, N):
            if Lidx[j * N + i]:
                Lsparse2dense[j * N + i] = Lnnz
                f.write("   work->L[%i] = " % (Lnnz))
                write_Kelem(f, j, i, n, m, p, P, A, G, perm, Wsparse2dense, True)
                for k in range(j):
                    if Lidx[k * N + i] and Lidx[k * N + j]:
                        f.write(
                            " - work->L[%i] * work->L[%i] * work->D[%i]"
                            % (Lsparse2dense[k * N + i], Lsparse2dense[k * N + j], k)
                        )
                f.write(";\n")
                f.write("   work->L[%i] /= work->D[%i];\n" % (Lnnz, j))
                Lnnz += 1
    f.write("}\n\n")

    f.write("void tri_solve(Workspace* work){\n")
    for i in range(N):
        f.write("   work->xyzbuff[%i] = work->kkt_rhs[%i]" % (i, perm[i]))
        for j in range(i):
            if Lidx[j * N + i]:
                f.write(
                    " - work->L[%i] * work->xyzbuff[%i]" % (Lsparse2dense[j * N + i], j)
                )
        f.write(";\n")

    for i in range(N):
        f.write("   work->xyzbuff[%i] /= work->D[%i];\n" % (i, i))

    for i in range(N - 1, -1, -1):
        f.write("   work->xyz[%i] = work->xyzbuff[%i]" % (perm[i], i))
        for j in range(i + 1, N):
            if Lidx[i * N + j]:
                f.write(
                    " - work->L[%i] * work->xyz[%i]"
                    % (Lsparse2dense[i * N + j], perm[j])
                )
        f.write(";\n")
    f.write("}")

    f.close()
    return Lsparse2dense


def generate_cone(solver_dir, m, Wnnz, Wsparse2dense):
    # Write header.
    f = open(solver_dir + "/cone.h", "a")
    write_license(f)
    f.write("#ifndef CONE_H\n")
    f.write("#define CONE_H\n\n")
    f.write('#include "utils.h"\n\n')

    f.write("void soc_product(double* u, double* v, double* p, int n);\n")
    f.write("void soc_division(double* lam, double* v, double* d, int n);\n")
    f.write("double soc_residual(double* u, int n);\n")
    f.write("double soc_residual2(double* u, int n);\n")
    f.write("double cone_residual(double* u, int l, int nsoc, int* q);\n")
    f.write("void bring2cone(double* u, int l, int nsoc, int* q);\n")
    f.write(
        "void cone_product(double* u, double* v, double* p, int l, int nsoc, int* q);\n"
    )
    f.write(
        "void cone_division(double* lambda, double* v, double* d, int l, int nsoc, int* q);\n"
    )
    f.write("void compute_mu(Workspace* work);\n")
    f.write("void compute_nt_scaling(Workspace* work);\n")
    f.write("void compute_lambda(Workspace* work);\n")
    f.write("void compute_WtW(Workspace* work);\n")
    f.write(
        "// Computes z = W * x, where W has the same sparsity structure as the Nesterov-Todd scaling matrices.\n"
    )
    f.write("void nt_multiply(double* W, double* x, double* z);\n")
    f.write("double linesearch(double* u, double* Du, double f, Workspace* work);\n")
    f.write("void compute_centering(Workspace* work);\n")
    f.write("#endif")
    f.close()

    # Write source.
    f = open(solver_dir + "/cone.c", "a")
    write_license(f)
    f.write('#include "cone.h"\n\n')

    f.write("void soc_product(double* u, double* v, double* p, int n){\n")
    f.write("   p[0] = dot(u, v, n);\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write("       p[i] = u[0] * v[i] + v[0] * u[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void soc_division(double* lam, double* v, double* d, int n){\n")
    f.write("   double f = lam[0] * lam[0] - dot(&lam[1], &lam[1], n - 1);\n")
    f.write("   double finv = safe_div(1.0, f);\n")
    f.write("   double lam0inv = safe_div(1.0, lam[0]);\n")
    f.write("   double lam1dv1 = dot(&lam[1], &v[1], n - 1);\n")
    f.write("   d[0] = finv * (lam[0] * v[0] - dot(&lam[1], &v[1], n - 1));\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write(
        "       d[i] = finv * (-lam[i] * v[0] + lam0inv * f * v[i] + lam0inv * lam1dv1 * lam[i]);\n"
    )
    f.write("   }\n")
    f.write("}\n")

    f.write("double soc_residual(double* u, int n){\n")
    f.write("   double res = 0;\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write("      res += u[i] * u[i];\n")
    f.write("   }\n")
    f.write("   res = qcos_sqrt(res) - u[0];\n")
    f.write("   return res;\n")
    f.write("}\n\n")
    f.write("double soc_residual2(double* u, int n){\n")
    f.write("   double res = u[0] * u[0];\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write("      res -= u[i] * u[i];\n")
    f.write("   }\n")
    f.write("   return res;\n")
    f.write("}\n\n")

    f.write("double cone_residual(double* u, int l, int nsoc, int* q){\n")
    f.write("   double res = -1e7;\n")
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < l; ++idx) {\n")
    f.write("      res = qcos_max(-u[idx], res);\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < nsoc; ++i) {\n")
    f.write("      res = qcos_max(soc_residual(&u[idx], q[i]), res);\n")
    f.write("      idx += q[i];\n")
    f.write("   }\n")
    f.write("   return res;\n")
    f.write("}\n\n")

    f.write("void bring2cone(double* u, int l, int nsoc, int* q){\n")
    f.write("   if (cone_residual(u, l, nsoc, q) >= 0) {\n")
    f.write("      double a = 0.0;\n\n")
    f.write("      int idx;\n")
    f.write("      for (idx = 0; idx < l; ++idx) {\n")
    f.write("         a = qcos_max(a, -u[idx]);\n")
    f.write("      }\n")
    f.write("      a = qcos_max(a, 0.0);\n\n")
    f.write("      for (int i = 0; i < nsoc; ++i) {\n")
    f.write("         double soc_res = soc_residual(&u[idx], q[i]);\n")
    f.write("         if (soc_res > 0 && soc_res > a) {\n")
    f.write("            a = soc_res;\n")
    f.write("         }\n")
    f.write("         idx += q[i];\n")
    f.write("      }\n")
    f.write("      for (idx = 0; idx < l; ++idx){\n")
    f.write("         u[idx] += (1 + a);\n")
    f.write("      }\n")
    f.write("      for (int i = 0; i < nsoc; ++i) {\n")
    f.write("         u[idx] += (1 + a);\n")
    f.write("         idx += q[i];\n")
    f.write("      }\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write(
        "void cone_product(double* u, double* v, double* p, int l, int nsoc, int* q) {\n"
    )
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < l; ++idx) {\n")
    f.write("       p[idx] = u[idx] * v[idx];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < nsoc; ++i) {\n")
    f.write("       soc_product(&u[idx], &v[idx], &p[idx], q[i]);\n")
    f.write("       idx += q[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write(
        "void cone_division(double* lambda, double* v, double* d, int l, int nsoc, int* q) {\n"
    )
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < l; ++idx) {\n")
    f.write("       d[idx] = safe_div(v[idx], lambda[idx]);\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < nsoc; ++i) {\n")
    f.write("       soc_division(&lambda[idx], &v[idx], &d[idx], q[i]);\n")
    f.write("       idx += q[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void compute_mu(Workspace* work){\n")
    if m == 0:
        f.write("   work->mu = 0.0;\n")
    else:
        f.write("   work->mu = (dot(work->s, work->z, work->m) / work->m);\n")
    f.write("}\n\n")

    f.write("void compute_nt_scaling(Workspace* work){\n")
    for i in range(Wnnz):
        f.write("   work->W[%i] = 0.0;\n" % i)
        f.write("   work->Winv[%i] = 0.0;\n" % i)
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < work->l; ++idx){\n")
    f.write("       work->W[idx] = qcos_sqrt(safe_div(work->s[idx], work->z[idx]));\n")
    f.write("       work->Winv[idx] = safe_div(1.0, work->W[idx]);\n")
    f.write("   }\n\n")

    f.write("   int nt_idx = idx;\n")
    f.write("   for (int i = 0; i < work->nsoc; ++i) {\n")
    f.write("       // Compute normalized vectors.\n")
    f.write("       double s_scal = soc_residual2(&work->s[idx], work->q[i]);\n")
    f.write("       s_scal = qcos_sqrt(s_scal);\n")
    f.write("       double f = safe_div(1.0, s_scal);\n")
    f.write("       scale_arrayf(&work->s[idx], work->sbar, f, work->q[i]);\n\n")

    f.write("       double z_scal = soc_residual2(&work->z[idx], work->q[i]);\n")
    f.write("       z_scal = qcos_sqrt(z_scal);\n")
    f.write("       f = safe_div(1.0, z_scal);\n")
    f.write("       scale_arrayf(&work->z[idx], work->zbar, f, work->q[i]);\n\n")

    f.write(
        "       double gamma = qcos_sqrt(0.5 * (1 + dot(work->sbar, work->zbar, work->q[i])));\n"
    )
    f.write("       f = safe_div(1.0, (2 * gamma));\n\n")
    f.write("       // Overwrite sbar with wbar.\n")
    f.write("       work->sbar[0] = f * (work->sbar[0] + work->zbar[0]);\n")
    f.write("       for (int j = 1; j < work->q[i]; ++j){\n")
    f.write("           work->sbar[j] = f * (work->sbar[j] - work->zbar[j]);\n")
    f.write("       }\n\n")
    f.write("       // Overwrite zbar with v.\n")
    f.write("       f = safe_div(1.0, qcos_sqrt(2 * (work->sbar[0] + 1)));\n")
    f.write("       work->zbar[0] = f * (work->sbar[0] + 1.0);\n")
    f.write("       for (int j = 1; j < work->q[i]; ++j) {\n")
    f.write("           work->zbar[j] = f * work->sbar[j];\n")
    f.write("       }\n\n")
    f.write("       // Compute W for second-order cones.\n")
    f.write("       int shift = 0;\n")
    f.write("       f = qcos_sqrt(safe_div(s_scal, z_scal));\n")
    f.write("       double finv = safe_div(1.0, f);\n")
    f.write("       for (int j = 0; j < work->q[i]; ++j) {\n")
    f.write("           for (int k = 0; k <= j; ++k) {\n")
    f.write(
        "               work->W[nt_idx + shift] = 2 * (work->zbar[k] * work->zbar[j]);\n"
    )
    f.write("               if (j != 0 && k == 0) {\n")
    f.write(
        "                   work->Winv[nt_idx + shift] = -work->W[nt_idx + shift];\n"
    )
    f.write("               }\n")
    f.write("               else {\n")
    f.write(
        "                   work->Winv[nt_idx + shift] = work->W[nt_idx + shift];\n"
    )
    f.write("               }\n")
    f.write("               if (j == k && j == 0) {\n")
    f.write("                   work->W[nt_idx + shift] -= 1;\n")
    f.write("                   work->Winv[nt_idx + shift] -= 1;\n")
    f.write("               }\n")
    f.write("               else if (j == k) {\n")
    f.write("                   work->W[nt_idx + shift] += 1;\n")
    f.write("                   work->Winv[nt_idx + shift] += 1;\n")
    f.write("               }\n")
    f.write("               work->W[nt_idx + shift] *= f;\n")
    f.write("               work->Winv[nt_idx + shift] *= finv;\n")
    f.write("               shift += 1;\n")
    f.write("           }\n")
    f.write("       }\n")
    f.write("       idx += work->q[i];\n")
    f.write("       nt_idx += (work->q[i] * work->q[i] + work->q[i]) / 2;\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void compute_WtW(Workspace* work){\n")
    for i in range(m):
        for j in range(i, m):
            if Wsparse2dense[j * m + i] != -1:
                f.write("   work->WtW[%i] = " % Wsparse2dense[j * m + i])
                for k in range(m):
                    row1 = k
                    col1 = j
                    row2 = k
                    col2 = i
                    if col1 < row1:
                        row1, col1 = col1, row1
                    if col2 < row2:
                        row2, col2 = col2, row2
                    if (
                        Wsparse2dense[col1 * m + row1] != -1
                        and Wsparse2dense[col2 * m + row2] != -1
                    ):
                        f.write(
                            " + work->W[%i] * work->W[%i]"
                            % (
                                Wsparse2dense[col1 * m + row1],
                                Wsparse2dense[col2 * m + row2],
                            )
                        )
                f.write(";\n")

    f.write("}\n\n")

    f.write("void compute_lambda(Workspace* work) {\n")
    f.write("   nt_multiply(work->W, work->z, work->lambda);\n")
    f.write("}\n\n")

    f.write("void nt_multiply(double* W, double* x, double* z) {\n")
    for i in range(m):
        f.write("   z[%i] = " % i)
        for j in range(m):
            row = i
            col = j
            if col < row:
                row, col = col, row
            if Wsparse2dense[col * m + row] != -1:
                f.write(" + W[%i] * x[%i]" % (Wsparse2dense[col * m + row], j))
        f.write(";\n")
    f.write("}\n\n")

    f.write("double linesearch(double* u, double* Du, double f, Workspace* work) {\n")
    f.write("   double al = 0.0;\n")
    f.write("   double au = 1.0;\n")
    f.write("   double a = 0.0;\n")
    f.write("   for (int i = 0; i < work->settings.bisection_iters; ++i) {\n")
    f.write("       a = 0.5 * (al + au);\n")
    f.write("       axpy(Du, u, work->ubuff1, safe_div(a, f), work->m);\n")
    f.write(
        "       if (cone_residual(work->ubuff1, work->l, work->nsoc, work->q) >= 0) {\n"
    )
    f.write("           au = a;\n")
    f.write("       }\n")
    f.write("       else {\n")
    f.write("           al = a;\n")
    f.write("       }\n")
    f.write("   }\n")
    f.write("   return al;\n")
    f.write("}\n")

    f.write("void compute_centering(Workspace* work) {\n")
    f.write(
        "   double a = qcos_min(linesearch(work->z, &work->xyz[work->n + work->p], 1.0, work), linesearch(work->s, work->Ds, 1.0, work));\n"
    )
    f.write(
        "   axpy(&work->xyz[work->n + work->p], work->z, work->ubuff1, a, work->m);\n"
    )
    f.write("   axpy(work->Ds, work->s, work->ubuff2, a, work->m);\n")
    f.write(
        "   double rho = safe_div(dot(work->ubuff1, work->ubuff2, work->m), dot(work->z, work->s, work->m));\n"
    )
    f.write("   double sigma = qcos_min(1.0, rho);\n")
    f.write("   sigma = qcos_max(0.0, sigma);\n")
    f.write("   sigma = sigma * sigma * sigma;\n")
    f.write("   work->sigma = sigma;\n")
    f.write("}\n")
    f.close()


def generate_kkt(solver_dir, n, m, p, P, c, A, b, G, h, perm, Wsparse2dense):
    # Write header.
    f = open(solver_dir + "/kkt.h", "a")
    write_license(f)
    f.write("#ifndef KKT_H\n")
    f.write("#define KKT_H\n\n")
    f.write('#include "cone.h"\n')
    f.write('#include "ldl.h"\n')
    f.write('#include "workspace.h"\n\n')
    f.write("void compute_kkt_residual(Workspace* work);\n")
    f.write("void construct_kkt_aff_rhs(Workspace* work);\n")
    f.write("void construct_kkt_comb_rhs(Workspace* work);\n")
    f.write("void predictor_corrector(Workspace* work);\n")
    f.write("#endif")
    f.close()

    # Write source.
    N = n + m + p
    f = open(solver_dir + "/kkt.c", "a")
    write_license(f)
    f.write('#include "kkt.h"\n\n')
    f.write("void compute_kkt_residual(Workspace* work){\n")

    f.write("   // Zero out NT Block.\n")
    for i in range(m**2):
        if Wsparse2dense[i] != -1:
            f.write("   work->WtW[%i] = 0.0;\n" % Wsparse2dense[i])

    for i in range(N):
        f.write("   work->kkt_res[%i] = " % i)
        for j in range(N):
            if write_Kelem(
                f,
                i,
                j,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
            ):
                if j < n:
                    f.write(" * work->x[%i]" % j)
                elif j >= n and j < n + p:
                    f.write(" * work->y[%i]" % (j - n))
                elif j >= n + p and j < n + m + p:
                    f.write(" * work->z[%i]" % (j - n - p))
                f.write(" + ")

        # Add [c;-b;s-h]
        if i < n:
            f.write(" work->c[%i]" % i)
        elif i >= n and i < n + p:
            f.write(" - work->b[%i]" % (i - n))
        elif i >= n + p and i < n + m + p:
            f.write("work->s[%i] - work->h[%i]" % (i - n - p, i - n - p))
        else:
            raise ValueError("Should not happen.")
        f.write(";\n")
    f.write("}\n\n")

    f.write("void construct_kkt_aff_rhs(Workspace* work) {\n")
    f.write(
        "   copy_and_negate_arrayf(work->kkt_res, work->kkt_rhs, work->n + work->p + work->m);\n"
    )
    f.write("   nt_multiply(work->W, work->lambda, work->ubuff1);\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->kkt_rhs[work->n + work->p + i] += work->ubuff1[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void construct_kkt_comb_rhs(Workspace* work) {\n")
    f.write(
        "   copy_and_negate_arrayf(work->kkt_res, work->kkt_rhs, work->n + work->p + work->m);\n"
    )
    f.write("   nt_multiply(work->Winv, work->Ds, work->ubuff1);\n")
    f.write("   nt_multiply(work->W, &work->xyz[work->n + work->p], work->ubuff2);\n")
    f.write(
        "   cone_product(work->ubuff1, work->ubuff2, work->ubuff3, work->l, work->nsoc, work->q);\n"
    )
    f.write("   double sm = work->sigma * work->mu;\n")
    f.write("   int idx = 0;\n")
    f.write("   for (idx = 0; idx < work->l; ++idx){\n")
    f.write("       work->ubuff3[idx] -= sm;\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->nsoc; ++i) {\n")
    f.write("       work->ubuff3[idx] -= sm;\n")
    f.write("       idx += work->q[i];\n")
    f.write("   }\n")
    f.write(
        "   cone_product(work->lambda, work->lambda, work->ubuff1, work->l, work->nsoc, work->q);\n"
    )
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->Ds[i] = -work->ubuff1[i] - work->ubuff3[i];\n")
    f.write("   }\n")
    f.write(
        "   cone_division(work->lambda, work->Ds, work->ubuff2, work->l, work->nsoc, work->q);\n"
    )
    f.write("   nt_multiply(work->W, work->ubuff2, work->ubuff1);\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->kkt_rhs[work->n + work->p + i] -= work->ubuff1[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void predictor_corrector(Workspace* work) {\n")
    f.write("   ldl(work);\n\n")
    f.write("   // Construct rhs for affine scaling direction.\n")
    f.write("   construct_kkt_aff_rhs(work);\n\n")
    f.write("   // Solve KKT system to get affine scaling direction.\n")
    f.write("   tri_solve(work);\n\n")
    f.write("   // Compute Dsaff.\n")
    f.write("   nt_multiply(work->W, &work->xyz[work->n + work->p], work->ubuff1);\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->ubuff1[i] = -work->lambda[i] - work->ubuff1[i];\n")
    f.write("   }\n")
    f.write("   nt_multiply(work->W, work->ubuff1, work->Ds);\n\n")
    f.write("   compute_centering(work);\n")
    f.write("   construct_kkt_comb_rhs(work);\n")
    f.write("   tri_solve(work);\n\n")
    f.write("   // Compute Dz.\n")
    f.write(
        "   cone_division(work->lambda, work->Ds, work->ubuff1, work->l, work->nsoc, work->q);\n"
    )
    f.write("   nt_multiply(work->W, &work->xyz[work->n + work->p], work->ubuff2);\n")
    f.write("   for (int i = 0; i < work->m; ++i) {\n")
    f.write("       work->ubuff3[i] = work->ubuff1[i] - work->ubuff2[i];\n")
    f.write("   }\n")
    f.write("   nt_multiply(work->W, work->ubuff3, work->Ds);\n\n")
    f.write("   // Compute step-size.\n")
    f.write("   nt_multiply(work->Winv, work->Ds, work->ubuff3);\n")
    f.write("   nt_multiply(work->W, &work->xyz[work->n + work->p], work->ubuff2);\n")
    f.write(
        "   double a = qcos_min(linesearch(work->lambda, work->ubuff3, 0.99, work), linesearch(work->lambda, work->ubuff2, 0.99, work));\n"
    )
    f.write("   work->a = a;\n\n")
    f.write("   // Update iterate.\n")
    f.write("   for (int i = 0; i < work->n; ++i){\n")
    f.write("       work->x[i] += a * work->xyz[i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->m; ++i){\n")
    f.write("       work->s[i] += a * work->Ds[i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->p; ++i){\n")
    f.write("       work->y[i] += a * work->xyz[work->n + i];\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < work->m; ++i){\n")
    f.write("       work->z[i] += a * work->xyz[work->n + work->p + i];\n")
    f.write("   }\n")
    f.write("}\n\n")
    f.close()


def generate_utils(solver_dir, n, m, p, P, c, A, b, G, h, l, nsoc, q, Wsparse2dense):
    # Write header.
    f = open(solver_dir + "/utils.h", "a")
    write_license(f)
    f.write("#ifndef UTILS_H\n")
    f.write("#define UTILS_H\n\n")
    f.write('#include "workspace.h"\n\n')
    f.write("#define qcos_abs(x) ((x)<0 ? -(x) : (x))\n")
    f.write("#define safe_div(a, b) (qcos_abs(a) > 1e-15) ? (a / b) : 1e16\n")
    f.write("#include <math.h>\n")
    f.write("#define qcos_sqrt(a) sqrt(a)\n")
    f.write("#define qcos_min(a, b) (((a) < (b)) ? (a) : (b))\n")
    f.write("#define qcos_max(a, b) (((a) > (b)) ? (a) : (b))\n\n")
    f.write("void load_data(Workspace* work);\n")
    f.write("void set_default_settings(Workspace* work);\n")
    f.write("void copy_arrayf(double* x, double* y, int n);\n")
    f.write("void copy_and_negate_arrayf(double* x, double* y, int n);\n")
    f.write("double dot(double* x, double* y, int n);\n")
    f.write("double inf_norm(double*x, int n);\n")
    f.write("void Px(double* x, double* y, Workspace* work);\n")
    f.write("void Ax(double* x, double* y, Workspace* work);\n")
    f.write("void Gx(double* x, double* y, Workspace* work);\n")
    f.write("void Atx(double* x, double* y, Workspace* work);\n")
    f.write("void Gtx(double* x, double* y, Workspace* work);\n")
    f.write("void scale_arrayf(double* x, double* y, double s, int n);\n")
    f.write("void axpy(double* x, double* y, double* z, double a, int n);\n")
    f.write("unsigned char check_stopping(Workspace* work);\n")
    f.write("#ifdef ENABLE_PRINTING\n")
    f.write("#include <stdio.h>\n")
    f.write("void print_header(Workspace* work);\n")
    f.write("void print_footer(Workspace* work);\n")
    f.write("void log_iter(Workspace* work);\n")
    f.write("#endif\n")
    f.write("#endif")
    f.close()

    # Write source.
    f = open(solver_dir + "/utils.c", "a")
    write_license(f)
    f.write('#include "utils.h"\n\n')

    f.write("void load_data(Workspace* work){\n")
    f.write("   work->n = %d;\n" % n)
    f.write("   work->m = %d;\n" % m)
    f.write("   work->p = %d;\n" % p)

    Pnnz = len(P.data) if P is not None else 0
    Annz = len(A.data) if A is not None else 0
    Gnnz = len(G.data) if G is not None else 0

    for i in range(Pnnz):
        f.write("   work->P[%i] = %.17g;\n" % (i, P.data[i]))
    f.write("\n")

    for i in range(len(c)):
        f.write("   work->c[%i] = %.17g;\n" % (i, c[i]))
    f.write("\n")

    for i in range(Annz):
        f.write("   work->A[%i] = %.17g;\n" % (i, A.data[i]))
    f.write("\n")

    for i in range(len(b)):
        f.write("   work->b[%i] = %.17g;\n" % (i, b[i]))
    f.write("\n")

    for i in range(Gnnz):
        f.write("   work->G[%i] = %.17g;\n" % (i, G.data[i]))
    f.write("\n")

    for i in range(len(h)):
        f.write("   work->h[%i] = %.17g;\n" % (i, h[i]))
    f.write("\n")
    f.write("   work->l = %d;\n" % l)
    f.write("   work->nsoc = %d;\n" % nsoc)

    for i in range(len(q)):
        f.write("   work->q[%i] = %d;\n" % (i, q[i]))
    f.write("\n")

    f.write("   work->mu = 0.0;\n")
    f.write("   work->sigma = 0.0;\n")
    f.write("   work->a = 0.0;\n")
    f.write("   work->sol.iters = 0;\n")
    f.write("   work->sol.pres = 0;\n")
    f.write("   work->sol.dres = 0;\n")
    f.write("   work->sol.gap = 0;\n")
    f.write("   work->sol.obj = 0;\n")
    f.write("   work->sol.solved = 0;\n")
    f.write("}\n\n")

    f.write("void set_default_settings(Workspace* work){\n")
    f.write("   work->settings.max_iters = 50;\n")
    f.write("   work->settings.bisection_iters = 5;\n")
    f.write("   work->settings.kkt_reg = 1e-7;\n")
    f.write("   work->settings.eabs = 1e-7;\n")
    f.write("   work->settings.erel = 1e-7;\n")
    f.write("   work->settings.solved = 0;\n")
    f.write("   work->settings.verbose = 1;\n")
    f.write("}\n\n")

    f.write("void copy_arrayf(double* x, double* y, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("      y[i] = x[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void copy_and_negate_arrayf(double* x, double* y, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("      y[i] = -x[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("double dot(double* x, double* y, int n){\n")
    f.write("   double ans = 0;\n")
    f.write("      for (int i = 0; i < n; ++i){\n")
    f.write("         ans += x[i] * y[i];\n")
    f.write("      }\n")
    f.write("   return ans;\n")
    f.write("}\n\n")

    f.write("double inf_norm(double* x, int n) {\n")
    f.write("   double norm = 0.0;\n")
    f.write("   double xi;\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("       xi = qcos_abs(x[i]);\n")
    f.write("       norm = qcos_max(norm , xi);\n")
    f.write("   }\n")
    f.write("   return norm;\n")
    f.write("}\n")

    f.write("void Px(double* x, double* y, Workspace* work){\n")
    N = n + m + p
    for i in range(n):
        f.write("   y[%i] = " % i)
        for j in range(n):
            if write_Kelem(
                f,
                i,
                j,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void Ax(double* x, double* y, Workspace* work){\n")
    N = n + m + p
    for i in range(p):
        f.write("   y[%i] = " % i)
        for j in range(n):
            if write_Kelem(
                f,
                i + n,
                j,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void Gx(double* x, double* y, Workspace* work){\n")
    N = n + m + p
    for i in range(m):
        f.write("   y[%i] = " % i)
        for j in range(n):
            if write_Kelem(
                f,
                i + n + p,
                j,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void Atx(double* x, double* y, Workspace* work){\n")
    N = n + m + p
    for i in range(n):
        f.write("   y[%i] = " % i)
        for j in range(p):
            if write_Kelem(
                f,
                i,
                j + n,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void Gtx(double* x, double* y, Workspace* work){\n")
    N = n + m + p
    for i in range(n):
        f.write("   y[%i] = " % i)
        for j in range(m):
            if write_Kelem(
                f,
                i,
                j + n + p,
                n,
                m,
                p,
                P,
                A,
                G,
                np.linspace(0, N - 1, N, dtype=np.int32),
                Wsparse2dense,
                False,
            ):
                f.write(" * x[%i]" % j)
                f.write(" + ")
        f.write("0;\n")
    f.write("}\n\n")

    f.write("void scale_arrayf(double* x, double* y, double s, int n){\n")
    f.write("      for (int i = 0; i < n; ++i){\n")
    f.write("         y[i] = s * x[i];\n")
    f.write("      }\n")
    f.write("}\n\n")

    f.write("void axpy(double* x, double* y, double* z, double a, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("       z[i] = a * x[i] + y[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("unsigned char check_stopping(Workspace* work){\n")
    f.write("   // Compute objective.\n")
    f.write("   double obj = dot(work->c, work->x, work->n);\n")
    f.write("   Px(work->x, work->xbuff, work);\n")
    f.write("   double Pxinf = inf_norm(work->xbuff, work->n);\n")
    f.write("   obj += 0.5 * dot(work->x, work->xbuff, work->n);\n")
    f.write("   work->sol.obj = obj;\n\n")
    f.write("   // Compute primal residual, dual residual, and duality gap.\n")
    f.write("   double pres = inf_norm(&work->kkt_res[work->n], work->p + work->m);\n")
    f.write("   double dres = inf_norm(work->kkt_res, work->n);\n")
    f.write("   work->sol.pres = pres;\n")
    f.write("   work->sol.dres = dres;\n")
    f.write("   work->sol.gap = work->mu * work->m;\n\n")

    f.write("   double binf = work->p > 0 ? inf_norm(work->b, work->p) : 0;\n")
    f.write("   double sinf = work->m > 0 ? inf_norm(work->s, work->m) : 0;\n")
    f.write("   double zinf = work->p > 0 ? inf_norm(work->z, work->m) : 0;\n")
    f.write("   double cinf = work->n > 0 ? inf_norm(work->c, work->n) : 0;\n")
    f.write("   double hinf = work->m > 0 ? inf_norm(work->h, work->m) : 0;\n")
    f.write("   Gtx(work->z, work->xbuff, work);\n")
    f.write("   double Gtzinf = inf_norm(work->xbuff, work->n);\n")
    f.write("   Atx(work->y, work->xbuff, work);\n")
    f.write("   double Atyinf = inf_norm(work->xbuff, work->n);\n")
    f.write("   Gx(work->x, work->ubuff1, work);\n")
    f.write("   double Gxinf = inf_norm(work->ubuff1, work->m);\n")
    # Using xbuff instead of adding a ybuff, since n >= p
    f.write("   Ax(work->x, work->xbuff, work);\n")
    f.write("   double Axinf = inf_norm(work->xbuff, work->p);\n\n")

    f.write("   // Compute max{Axinf, binf, Gxinf, hinf, sinf}.\n")
    f.write("   double pres_rel = qcos_max(Axinf, binf);\n")
    f.write("   pres_rel = qcos_max(pres_rel, Gxinf);\n")
    f.write("   pres_rel = qcos_max(pres_rel, hinf);\n")
    f.write("   pres_rel = qcos_max(pres_rel, sinf);\n\n")
    f.write("   // Compute max{Pxinf, Atyinf, Gtzinf, cinf}.\n")
    f.write("   double dres_rel = qcos_max(Pxinf, Atyinf);\n")
    f.write("   dres_rel = qcos_max(dres_rel, Gtzinf);\n")
    f.write("   dres_rel = qcos_max(dres_rel, cinf);\n\n")

    f.write("   // Compute max{sinf, zinf}.\n")
    f.write("   double gap_rel = qcos_max(sinf, zinf);\n\n")

    f.write(
        "   if (pres < work->settings.eabs + work->settings.erel * pres_rel && dres < work->settings.eabs + work->settings.erel * dres_rel && work->sol.gap < work->settings.eabs + work->settings.erel * gap_rel) {\n"
    )
    f.write("      return 1;\n")
    f.write("   }\n")
    f.write("   return 0;\n")
    f.write("}\n\n")

    Pnnz = len(P.data) if P is not None else 0
    Annz = len(A.data) if A is not None else 0
    Gnnz = len(G.data) if G is not None else 0

    f.write("#ifdef ENABLE_PRINTING\n")
    f.write("void print_header(Workspace* work) {\n")
    f.write('   printf("\\n");\n')
    f.write(
        '   printf("+-------------------------------------------------------+\\n");\n'
    )
    f.write(
        '   printf("|              QCOS Custom Generated Solver             |\\n");\n'
    )
    f.write(
        '   printf("|             (c) Govind M. Chari, 2024                 |\\n");\n'
    )
    f.write(
        '   printf("|    University of Washington Autonomous Controls Lab   |\\n");\n'
    )
    f.write(
        '   printf("+-------------------------------------------------------+\\n");\n'
    )
    f.write(
        '   printf("| Problem Data:                                         |\\n");\n'
    )
    f.write(
        '   printf("|     variables:        %-9d                       |\\n");\n' % n
    )
    f.write(
        '   printf("|     constraints:      %-9d                       |\\n");\n'
        % (l + p + nsoc)
    )
    f.write(
        '   printf("|     eq constraints:   %-9d                       |\\n");\n' % p
    )
    f.write(
        '   printf("|     ineq constraints: %-9d                       |\\n");\n' % l
    )
    f.write(
        '   printf("|     soc constraints:  %-9d                       |\\n");\n' % nsoc
    )
    f.write(
        '   printf("|     nnz(P):           %-9d                       |\\n");\n' % Pnnz
    )
    f.write(
        '   printf("|     nnz(A):           %-9d                       |\\n");\n' % Annz
    )
    f.write(
        '   printf("|     nnz(G):           %-9d                       |\\n");\n' % Gnnz
    )
    f.write(
        '   printf("| Solver Settings:                                      |\\n");\n'
    )
    f.write(
        '   printf("|     max_iter: %-3d eabs: %3.2e erel: %3.2e      |\\n", work->settings.max_iters, work->settings.eabs, work->settings.erel);\n'
    )
    f.write(
        '   printf("|     bisection_iters: %-2d static_regularization: %3.2e     |\\n", work->settings.bisection_iters, work->settings.kkt_reg);\n'
    )
    f.write(
        '   printf("+-------------------------------------------------------+\\n");\n'
    )
    f.write('   printf("\\n");\n')
    f.write(
        '   printf("+--------+-----------+------------+------------+------------+-----------+-----------+\\n");\n'
    )
    f.write(
        '   printf("|  Iter  |   Pcost   |    Pres    |    Dres    |     Gap    |     Mu    |    Step   |\\n");\n'
    )
    f.write(
        '   printf("+--------+-----------+------------+------------+------------+-----------+-----------+\\n");\n'
    )
    f.write("}\n\n")

    f.write("void print_footer(Workspace* work){\n")
    f.write('       printf("\\nsolved: %d ", work->sol.solved);\n')
    f.write('       printf("\\nnumber of iterations: %d ", work->sol.iters);\n')
    f.write('       printf("\\nobjective: %f ", work->sol.obj);\n')
    f.write("}\n\n")

    f.write("void log_iter(Workspace* work) {\n")
    f.write(
        'printf("|   %2d   | %+.2e | %+.3e | %+.3e | %+.3e | %+.2e |   %.3f   |\\n",work->sol.iters, work->sol.obj, work->sol.pres, work->sol.dres, work->sol.gap, work->mu, work->a);'
    )
    f.write(
        'printf("+--------+-----------+------------+------------+------------+-----------+-----------+\\n");'
    )
    f.write("}\n")
    f.write("#endif\n")
    f.close()


def generate_solver(solver_dir, m, Wsparse2dense):
    f = open(solver_dir + "/qcos_custom.h", "a")
    write_license(f)
    f.write("#ifndef QCOS_CUSTOM_H\n")
    f.write("#define QCOS_CUSTOM_H\n\n")
    f.write('#include "cone.h"\n')
    f.write('#include "kkt.h"\n')
    f.write('#include "ldl.h"\n')
    f.write('#include "utils.h"\n')
    f.write('#include "workspace.h"\n\n')
    f.write("void qcos_custom_solve(Workspace* work);\n")
    f.write("#endif")
    f.close()

    f = open(solver_dir + "/qcos_custom.c", "a")
    write_license(f)
    f.write('#include "qcos_custom.h"\n\n')
    f.write("void initialize_ipm(Workspace* work){\n")
    f.write("   // Set NT block to I.\n")
    for i in range(m**2):
        if Wsparse2dense[i] != -1:
            f.write("   work->WtW[%i] = 0.0;\n" % Wsparse2dense[i])
    for i in range(m):
        f.write("   work->WtW[%i] = 1.0;\n" % Wsparse2dense[i * m + i])

    f.write("\n   // kkt_rhs = [-c;b;h].\n")
    f.write("   for(int i = 0; i < work->n; ++i){\n")
    f.write("       work->kkt_rhs[i] = -work->c[i];\n")
    f.write("   }\n")
    f.write("   for(int i = 0; i < work->p; ++i){\n")
    f.write("       work->kkt_rhs[work->n + i] = work->b[i];\n")
    f.write("   }\n")
    f.write("   for(int i = 0; i < work->m; ++i){\n")
    f.write("       work->kkt_rhs[work->n + work->p + i] = work->h[i];\n")
    f.write("   }\n\n")
    f.write("   ldl(work);\n")
    f.write("   tri_solve(work);\n")
    f.write("   copy_arrayf(work->xyz, work->x, work->n);\n")
    f.write("   copy_arrayf(&work->xyz[work->n], work->y, work->p);\n")
    f.write("   copy_arrayf(&work->xyz[work->n + work->p], work->z, work->m);\n")
    f.write(
        "   copy_and_negate_arrayf(&work->xyz[work->n + work->p], work->s, work->m);\n"
    )
    f.write("   bring2cone(work->s, work->l, work->nsoc, work->q);\n")
    f.write("   bring2cone(work->z, work->l, work->nsoc, work->q);\n")
    f.write("}\n\n")

    f.write("void qcos_custom_solve(Workspace* work){\n")
    f.write("#ifdef ENABLE_PRINTING\n")
    f.write("   if (work->settings.verbose) {\n")
    f.write("       print_header(work);\n")
    f.write("   }\n")
    f.write("#endif\n")
    f.write("   initialize_ipm(work);\n")
    f.write("   for (int i = 1; i < work->settings.max_iters; ++i) {\n")
    f.write("      compute_kkt_residual(work);\n")
    f.write("      compute_mu(work);\n")
    f.write("      if (check_stopping(work)) {\n")
    f.write("         work->sol.solved = 1;\n")
    f.write("         copy_arrayf(work->x, work->sol.x, work->n);\n")
    f.write("         copy_arrayf(work->s, work->sol.s, work->m);\n")
    f.write("         copy_arrayf(work->y, work->sol.y, work->p);\n")
    f.write("         copy_arrayf(work->z, work->sol.z, work->m);\n")
    f.write("         break;\n")
    f.write("      }\n")
    f.write("      compute_nt_scaling(work);\n")
    f.write("      compute_lambda(work);\n")
    f.write("      compute_WtW(work);\n")
    f.write("      predictor_corrector(work);\n")
    f.write("      work->sol.iters = i;\n")
    f.write("   #ifdef ENABLE_PRINTING\n")
    f.write("       if (work->settings.verbose) {\n")
    f.write("           log_iter(work);\n")
    f.write("       }\n")
    f.write("   #endif\n")
    f.write("   }\n")
    f.write("#ifdef ENABLE_PRINTING\n")
    f.write("   if (work->settings.verbose) {\n")
    f.write("       print_footer(work);\n")
    f.write("   }\n")
    f.write("#endif\n")
    f.write("}\n\n")
    f.close()


def generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q):
    f = open(solver_dir + "/runtest.c", "a")
    write_license(f)
    f.write("#include <stdio.h>\n")
    f.write("#include <time.h>\n")
    f.write('#include "qcos_custom.h"\n\n')
    f.write("int main(){\n")
    f.write("   Workspace work;\n")
    f.write("   load_data(&work);\n")
    f.write("   set_default_settings(&work);\n")
    f.write("   work.settings.verbose = 0;\n")
    f.write("   double N = 1000;\n")
    f.write("   double total_time = 0;\n")
    f.write("   for (int i = 0; i < N; ++i) {\n")
    f.write("       struct timespec start, end;\n")
    f.write("       clock_gettime(CLOCK_MONOTONIC, &start);\n")
    f.write("       qcos_custom_solve(&work);\n")
    f.write("       clock_gettime(CLOCK_MONOTONIC, &end);\n")
    f.write(
        "       double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;\n"
    )
    f.write("       total_time += elapsed_time;\n")
    f.write("   }\n")
    f.write("   double average_solvetime_ms = 1e3 * total_time / N;\n")
    f.write('   printf("\\nTotal Time: %.9f ms", 1e3 * total_time);\n')
    f.write('   printf("\\nAverage Solvetime: %.9f ms", 1e3 * total_time / N);\n')
    f.write('   FILE *file = fopen("result.bin", "wb");\n')
    f.write("   fwrite(&work.sol.solved, sizeof(unsigned char), 1, file);\n")
    f.write("   fwrite(&work.sol.obj, sizeof(double), 1, file);\n")
    f.write("   fwrite(&average_solvetime_ms, sizeof(double), 1, file);\n")
    f.write("   fclose(file);\n")

    # f.write("   printf(\"xyz: {\");")
    # f.write("   for(int i = 0; i < work.n + work.m + work.p; ++i){\n")
    # f.write("   printf(\"%f, \", work.xyz[i]);\n")
    # f.write("   }\n")

    # f.write("   printf(\"\\nkkt_res: {\");")
    # f.write("   for(int i = 0; i < work.n + work.m + work.p; ++i){\n")
    # f.write("   printf(\"%f, \", work.kkt_res[i]);\n")
    # f.write("   }\n")

    f.write('   printf("\\nobj: %f", work.sol.obj);\n')

    f.write('   printf("\\n\\nx: {");')
    f.write("   for(int i = 0; i < work.n; ++i){\n")
    f.write('   printf("%f, ", work.sol.x[i]);\n')
    f.write("   }\n")

    f.write('   printf("\\ns: {");')
    f.write("   for(int i = 0; i < work.m; ++i){\n")
    f.write('   printf("%f, ", work.sol.s[i]);\n')
    f.write("   }\n")

    f.write('   printf("\\ny: {");')
    f.write("   for(int i = 0; i < work.p; ++i){\n")
    f.write('   printf("%f, ", work.sol.y[i]);\n')
    f.write("   }\n")

    f.write('   printf("\\nz: {");')
    f.write("   for(int i = 0; i < work.m; ++i){\n")
    f.write('   printf("%f, ", work.sol.z[i]);\n')
    f.write("   }\n")
    f.write('   printf("}\\n");\n')

    # f.write("   printf(\"\\nW: {\");")
    # f.write("   for(int i = 0; i < 9; ++i){\n")
    # f.write("   printf(\"%f, \", work.W[i]);\n")
    # f.write("   }\n")

    # f.write("   printf(\"\\nWinv: {\");")
    # f.write("   for(int i = 0; i < 9; ++i){\n")
    # f.write("   printf(\"%f, \", work.Winv[i]);\n")
    # f.write("   }\n")

    # f.write("   printf(\"\\nWtW: {\");")
    # f.write("   for(int i = 0; i < 9; ++i){\n")
    # f.write("   printf(\"%f, \", work.WtW[i]);\n")
    # f.write("   }\n")

    # f.write("   printf(\"\\nlambda: {\");")
    # f.write("   for(int i = 0; i < work.m; ++i){\n")
    # f.write("   printf(\"%f, \", work.lambda[i]);\n")
    # f.write("   }\n")

    # f.write("   printf(\"\\nkkt_rhs: {\");")
    # f.write("   for(int i = 0; i < work.m + work.p + work.n; ++i){\n")
    # f.write("   printf(\"%f, \", work.kkt_rhs[i]);\n")
    # f.write("   }\n")

    # f.write("   printf(\"\\nDs: {\");")
    # f.write("   for(int i = 0; i < work.m; ++i){\n")
    # f.write("   printf(\"%f, \", work.Ds[i]);\n")
    # f.write("   }\n")

    # f.write("   printf(\"\\nsigma: %f, \", work.sigma);\n")
    # f.write("   printf(\"\\na: %f, \", work.a);\n")
    f.write("}")

    f.close()
