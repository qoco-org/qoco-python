import os
import shutil

def _generate_solver(n, m, p, P, c, A, b, G, h, l, nsoc, q, output_dir, name="qcosgen"):
    solver_dir = output_dir + "/" + name

    print("\n")
    if os.path.exists(solver_dir):
        print("Removing existing solver.")
        shutil.rmtree(solver_dir)
    
    print("Generating solver.")
    os.mkdir(solver_dir)

    generate_cmakelists(solver_dir)
    generate_workspace(solver_dir, P, c, A, b, G, h, q)
    generate_ldl(A, solver_dir)
    generate_utils(solver_dir, P, c, A, b, G, h, l, nsoc, q)
    generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q)

def generate_workspace(solver_dir, P, c, A, b, G, h, q):
    f = open(solver_dir + "/workspace.h", "a")
    f.write("#ifndef WORKSPACE_H\n")
    f.write("#define WORKSPACE_H\n\n")

    f.write("typedef struct {\n")
    f.write("   double P[%i];\n" % (len(P.x)))
    f.write("   double c[%i];\n" % (len(c)))
    f.write("   double A[%i];\n" % (len(A.x)))
    f.write("   double b[%i];\n" % (len(b)))
    f.write("   double G[%i];\n" % (len(G.x)))
    f.write("   double h[%i];\n" % (len(h)))
    f.write("   int l;\n")
    f.write("   int nsoc;\n")
    f.write("   int q[%i];\n" % (len(q)))
    f.write("} Workspace;\n\n")
    f.write("#endif")
    f.close()

def generate_utils(solver_dir, P, c, A, b, G, h, l, nsoc, q):
    # Write header.
    f = open(solver_dir + "/utils.h", "a")
    f.write("#ifndef UTILS_H\n")
    f.write("#define UTILS_H\n\n")

    f.write("#include \"workspace.h\"\n\n")

    f.write("void load_data(Workspace* work);\n")
    f.write("#endif")
    f.close()

    # Write source.
    f = open(solver_dir + "/utils.c", "a")
    f.write("#include \"utils.h\"\n\n")

    f.write("void load_data(Workspace* work){\n")
    for i in range(len(P.x)):
        f.write("   work->P[%i] = %.17g;\n" % (i, P.x[i]))
    f.write("\n")

    for i in range(len(c)):
        f.write("   work->c[%i] = %.17g;\n" % (i, c[i]))
    f.write("\n")

    for i in range(len(A.x)):
        f.write("   work->A[%i] = %.17g;\n" % (i, A.x[i]))
    f.write("\n")

    for i in range(len(b)):
        f.write("   work->b[%i] = %.17g;\n" % (i, b[i]))
    f.write("\n")

    for i in range(len(G.x)):
        f.write("   work->G[%i] = %.17g;\n" % (i, G.x[i]))
    f.write("\n")

    for i in range(len(h)):
        f.write("   work->h[%i] = %.17g;\n" % (i, h[i]))
    f.write("\n")

    f.write("   work->l = %d;\n" % l)
    f.write("   work->nsoc = %d;\n" % nsoc)

    for i in range(len(q)):
        f.write("   work->q[%i] = %d;\n" % (i, q[i]))
    f.write("\n")
    f.write("}")
    f.close()

def generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q):
    f = open(solver_dir + "/runtest.c", "a")
    f.write("#include \"workspace.h\"\n\n")
    f.write("#include \"utils.h\"\n\n")
    f.write("int main(){\n")
    f.write("   Workspace work;\n")
    f.write("   load_data(&work);\n")
    f.write("}")
    f.close()

def generate_ldl(M, solver_dir):
    f = open(solver_dir + "/ldl.c", "a")
    f.write("#include \"workspace.h\"\n\n")
    f.close()

def generate_cmakelists(solver_dir):
    f = open(solver_dir + "/CMakeLists.txt", "a")
    f.write("cmake_minimum_required(VERSION 3.18)\n")
    f.write("set(CMAKE_C_FLAGS \"-O3 -march=native -Wall -Wextra\")\n")
    f.write("project(qcosgen)\n")
    f.write("# Build qcosgen shared library.\n")
    f.write("add_library(qcosgen SHARED)\n")
    f.write("target_sources(qcosgen PRIVATE utils.c ldl.c)\n\n")
    f.write("# Build qcos demo.\n")
    f.write("add_executable(runtest runtest.c)\n")
    f.write("target_link_libraries(runtest qcosgen)\n")
    f.close()
