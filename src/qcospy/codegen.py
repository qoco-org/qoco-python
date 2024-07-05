import os
import shutil
from qcospy.codegen_utils import *

def _generate_solver(n, m, p, P, c, A, b, G, h, l, nsoc, q, output_dir, name="qcosgen"):
    solver_dir = output_dir + "/" + name

    print("\n")
    if os.path.exists(solver_dir):
        print("Removing existing solver.")
        shutil.rmtree(solver_dir)
    
    print("Generating solver.")
    os.mkdir(solver_dir)

    generate_workspace(solver_dir, P, c, A, b, G, h, l, nsoc, q)
    generate_ldl(A, solver_dir)

def generate_workspace(solver_dir, P, c, A, b, G, h, l, nsoc, q):
    f = open(solver_dir + "/workspace.h", "a")
    f.write("#ifndef WORKSPACE_H\n")
    f.write("#define WORKSPACE_H\n\n")
    f.write("typedef struct {\n")
    f.write("   ")
    write_vectorf(f, P.x, "P")
    f.write("   ")
    write_vectorf(f, c, "c")
    f.write("   ")
    write_vectorf(f, A.x, "A")
    f.write("   ")
    write_vectorf(f, b, "b")
    f.write("   ")
    write_vectorf(f, G.x, "G")
    f.write("   ")
    write_vectorf(f, h, "h")
    f.write("   ")
    write_int(f, l, "l")
    f.write("   ")
    write_int(f, nsoc, "nsoc")
    f.write("   ")
    write_vectori(f, q, "q")


    f.write("} work;\n")
    f.write("#endif")


def generate_ldl(M, solver_dir):
    f = open(solver_dir + "/ldl.c", "a")
    f.write("#include \"workspace.h\"\n\n")
    f.close()
