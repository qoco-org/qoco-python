def write_vectori(f, x, name):
    if x is None or len(x) == 0:
        f.write("int* %s = nullptr;\n" % name)
    else:
        f.write("int %s[%i] = {" % (name, len(x)))
        for i in x:
            f.write("%i," % i)
        f.write("};\n")

def write_vectorf(f, x, name):
    if x is None or len(x) == 0:
        f.write("double* %s = nullptr;\n" % name)
    else:
        f.write("double %s[%i] = {" % (name, len(x)))
        for i in x:
            f.write("%.17g," % i)
        f.write("};\n")

def write_int(f, x, name):
    f.write("int %s = %i;\n" % (name, x))
    