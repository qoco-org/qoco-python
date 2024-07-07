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
    
# Writes the 1D array index of the (i,j) element of the KKT matrix [P+reg*I A' G';A -reg*I 0;G 0 -W'W]
def write_Kelem(f, i, j, n, m, p, P, A, G, reg):
    if (i > j):
        raise ValueError("Accessing lower triangular part of K.")
    
    # P block
    if (i < n and j < n):
        # Check if element is nonzero.
        if P[i,j] == 0.0:
            f.write("0")
        else:
            # need to get index of P[i,j] in the data array for P.
            dataidx = get_data_idx(P, i, j)
            f.write("work.P[%d]" % dataidx)
        if (i == j):
            f.write(" + %f" % reg)
    
    # A' block    
    if (i < n and j >= n and j < n + p):
        # Row and column of A
        col = i
        row = j - n

        # Check if element is nonzero.
        if A[row,col] == 0.0:
            f.write("0")
        else:
            # need to get index of A[row,col] in the data array for A.
            dataidx = get_data_idx(A, row, col)
            f.write("work.A[%d]" % dataidx)

    # G' block    
    if (i < n and j >= n + p and j < n + p + m):
        # Row and column of G
        col = i
        row = j - n - p

        # Check if element is nonzero.
        if G[row,col] == 0.0:
            f.write("0")
        else:
            # need to get index of A[row,col] in the data array for A.
            dataidx = get_data_idx(G, row, col)
            f.write("work.G[%d]" % dataidx)
    
    # -reg * I block.
    if (i >= n and i < n + p and j >= n and j < n + p):
        # Row and column of G
        row = i - n
        col = j - n

        # Check if element is nonzero.
        if row != col:
            f.write("0")
        else:
            f.write("-%f" % reg)

    # Nesterov-Todd block. TODO: assumes W is dense and doesnt use W'*W.
    if (i >= n + p and i < n + p + m and j >= n + p and j < n + m + p):
        # Row and column of G
        row = i - n - p
        col = j - n - p
        f.write("work.W[%d]" % (col * m + row))


def get_data_idx(M, i, j):
    for dataidx in range(M.indptr[j], M.indptr[j+1]):
        if (M.indices[dataidx] == i):
            return dataidx

