# Writes the 1D array index of the (i,j) element of the KKT matrix [P+reg*I A' G';A -reg*I 0;G 0 -W'W]
def write_Kelem(f, i, j, n, m, p, P, A, G, perm, Wsparse2dense, reg):

    # Row and column to access within KKT matrix.
    i = perm[i]
    j = perm[j]

    # If accessing a lower triangular element, index the corresponding upper triangular element (Otherwise would have to add logic for accessing lower triangular blocks).
    if (i > j):
        temp = i
        i = j
        j = temp
    
    # P block
    if (i < n and j < n):
        # Check if element is nonzero. TODO: There should be a better way to check if an element is nonzero.
        if P[i,j] == 0.0:
            # f.write("0")
            return False
        else:
            # need to get index of P[i,j] in the data array for P.
            dataidx = get_data_idx(P, i, j)
            f.write("work->P[%d]" % dataidx)
        if (i == j and reg):
            f.write(" + work->settings.kkt_reg")
    
    # A' block    
    elif (i < n and j >= n and j < n + p):
        # Row and column of A
        col = i
        row = j - n

        # Check if element is nonzero. TODO: There should be a better way to check if an element is nonzero.
        if A[row,col] == 0.0:
            # f.write("0")
            return False
        else:
            # need to get index of A[row,col] in the data array for A.
            dataidx = get_data_idx(A, row, col)
            f.write("work->A[%d]" % dataidx)

    # G' block    
    elif (i < n and j >= n + p and j < n + p + m):
        # Row and column of G
        col = i
        row = j - n - p

        # Check if element is nonzero.  TODO: There should be a better way to check if an element is nonzero.
        if G[row,col] == 0.0:
            # f.write("0")
            return False
        else:
            # need to get index of A[row,col] in the data array for A.
            dataidx = get_data_idx(G, row, col)
            f.write("work->G[%d]" % dataidx)
    
    # -reg * I block.
    elif (i >= n and i < n + p and j >= n and j < n + p):
        # Row and column of G
        row = i - n
        col = j - n

        # Check if element is nonzero.
        if row != col:
            # f.write("0")
            return False
        elif (reg):
            f.write("-work->settings.kkt_reg")
        else:
            return False

    # Accessing the (2,3) block of the KKT matrix.
    elif (i >= n and i < n + p and j >= n + p and j < n + m + p):
        # f.write("0")
        return False

    # Nesterov-Todd block.
    elif (i >= n + p and i < n + p + m and j >= n + p and j < n + m + p):
        # Row and column of G
        row = i - n - p
        col = j - n - p

        # Only access lower triangular elements of W.
        if (row > col):
            temp = row
            row = col
            col = temp
        if (Wsparse2dense[col * m + row] != -1):
            f.write(" - work->WtW[%d]" % Wsparse2dense[col * m + row])
            if (row == col):
                f.write(" - work->settings.kkt_reg")
        else:
            # f.write("0")
            return False
    return True

def get_data_idx(M, i, j):
    for dataidx in range(M.indptr[j], M.indptr[j+1]):
        if (M.indices[dataidx] == i):
            return dataidx

