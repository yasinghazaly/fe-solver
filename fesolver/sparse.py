import numpy as np 

class CSR :
    """Square sparse matrix in compressed sparse row (CSR) format.

    Built from coordinate triplets, which may be in any order and may
    repeat a position. The triplets are sorted by row and then column,
    entries at the same position are added together, and the result is
    stored by row.

    Attributes
    ----------
    indptr : ndarray of int, shape (n + 1,)
        Row i occupies positions indptr[i] up to (not including)
        indptr[i + 1] of indices and data. A row with no entries repeats
        the previous value. The last entry is the number of stored values.
    indices : ndarray of int
        Column of each stored value. Columns increase within each row.
    data : ndarray of float
        The stored values, row by row.
    n : int
        Number of rows and columns.

    Only positions that appear in the triplets are stored. A stored value
    that happens to be zero, for instance after two contributions cancel,
    is kept.
    """
    def __init__ (self,rows,cols,vals,n) :
       self.indptr, self.indices,self.data = self.coo_to_csr(rows,cols,vals,n)
       self.n = int(n)
       
    @staticmethod
    def coo_to_csr (rows,cols,vals, n) : 
        """Convert triplets to the three CSR arrays.

        Parameters
        ----------
        rows, cols : ndarray of int
            Row and column of each entry, in any order.
        vals : ndarray of float
            Value of each entry. Entries at the same (row, column) are added.
        n : int
            Matrix size. It must be given, because the last rows may be
            empty and the triplets alone cannot say how many rows exist.

        Returns
        -------
        indptr, indices, data : ndarray
            The CSR arrays as described in the class docstring.
        """
        order = np.lexsort((cols,rows))
        sorted_rows = rows[order]
        sorted_cols = cols[order]
        sorted_vals = vals[order]
        output = {}
        for index, value in enumerate(sorted_vals) : 
            if (sorted_rows[index], sorted_cols[index]) in output : 
                output[(sorted_rows[index], sorted_cols[index])] += value
            else : 
                output[(sorted_rows[index],sorted_cols[index])] = value
            
        output_rows,output_cols = zip(*output.keys())
        output_rows = list(output_rows)
        indices = list(output_cols)
        data = list(output.values())
        
        indptr = [0]
        indptr_dict = {}
        for index, value in enumerate(output_rows) : 
            if value not in indptr_dict.keys() : 
                indptr_dict[int(value)] = 1
            else : 
                indptr_dict[int(value)] += 1
                
        running_total = 0 
        for index in range(n) : 
            running_total += indptr_dict.get(index,0)
            indptr.append(running_total)
        
        return np.array(indptr,dtype=int) , np.array(indices,dtype=int) , np.array(data,dtype=float) 

    def to_dense(self) : 
        """Return the matrix as a dense (n, n) array.

        Used to check the sparse code against the dense reference. It
        stores every zero, so it is only suitable for small matrices.
        """
        matrix = np.zeros((self.n,self.n))
        for row in range(self.n) : 
            start = self.indptr[row]
            end = self.indptr[row+1]
            matrix[row,self.indices[start:end]] = self.data[start:end]
        return matrix
    
    def matvec(self , x) : 
        """Matrix-vector product K @ x.

        Parameters
        ----------
        x : ndarray, shape (n,)

        Returns
        -------
        ndarray, shape (n,)
            For each row, the sum of its stored values times x at their
            columns. Rows with no entries give zero.
        """
        y = np.zeros(self.n)
        for row in range(self.n) : 
            start = self.indptr[row]
            end = self.indptr[row+1]
            y[row] = self.data[start:end] @ x[self.indices[start:end]]
        return y 
    
    def diagonal(self) : 
        """Return the diagonal as a vector of length n.
 
        A row with no stored diagonal entry gives zero. Used by the Jacobi
        preconditioner.
 
        Returns
        -------
        ndarray, shape (n,)
        """
        M = []
        for row in range(self.n) : 
            start = self.indptr[row]
            end = self.indptr[row+1]
            cols = self.indices[start:end]
            if row in cols : 
                index = np.where(cols == row)[0][0]
                M.append(self.data[start+index])
            else : 
                M.append(0)
        return np.asarray(M)
         
    def lower_triangle(self) : 
        """Return the lower triangle, diagonal included, as a new CSR.
 
        Keeps every stored entry with column <= row. Because columns
        increase within each row, the diagonal is the last stored entry of
        each row, at indptr[i + 1] - 1. The original matrix is unchanged.
 
        Returns
        -------
        CSR
            Same size, holding only the lower triangle.
        """
        rows = []
        cols = []
        vals = []
        for row in range(self.n) :
            start = self.indptr[row]
            end = self.indptr[row+1]
            for position in range(start,end,1) :
                if self.indices[position] <= row : 
                    rows.append(row)
                    cols.append(self.indices[position])
                    vals.append(self.data[position])
                    
        return CSR(np.asarray(rows),np.asarray(cols),np.asarray(vals),self.n)
    
    def incomplete_cholesky(self) : 
        """Incomplete Cholesky factor with zero fill-in, IC(0).
 
        Finds a lower-triangular L with exactly the pattern of the
        lower triangle of K, so that L L^T equals K at every stored
        position. Exact Cholesky would create new non-zeros (fill-in);
        IC(0) drops them, which keeps L as sparse as K but makes
        L L^T only an approximation of K.

        L is built row by row. For an off-diagonal entry (i, j),

            L(i, j) = (K(i, j) - sum of L(i, k) L(j, k)) / L(j, j)

        summed over the columns k < j stored in both rows i and j. The
        diagonal is

            L(i, i) = sqrt(K(i, i) - sum of L(i, k)^2)

        over the entries already found in row i.

        The factorisation can break down for a symmetric positive
        definite matrix that is not diagonally dominant, when the value
        under the square root is negative. The result is then NaN
        (numpy warns); cg checks for this and raises.

        Returns
        -------
        CSR
            L, with the same indptr and indices as lower_triangle().
        """
        L = self.lower_triangle()
        for row in range(L.n) :
            memory = {}
            start = L.indptr[row]
            end = L.indptr[row+1]
            for position in range(start,end,1) :
                row_sum = 0
                if row == L.indices[position] : 
                    for value in memory.values(): 
                        row_sum += value**2
                    L.data[position] = np.sqrt(L.data[position] - row_sum)
                elif row != L.indices[position] : 
                    j = L.indices[position]
                    column_sum = 0 
                    for q in range(L.indptr[j], L.indptr[j+1] - 1) : 
                        k = L.indices[q]
                        if k in memory : 
                            column_sum += memory[k] * L.data[q]
                    L.data[position] = (1/L.data[L.indptr[L.indices[position]+1]-1]) * (L.data[position] - column_sum)
                    memory[L.indices[position]] = L.data[position]
        return L

    def forward_solve(self,L,r) : 
        """Solve L y = r for lower-triangular L, from the top row down.
 
        For each row, y(i) = (r(i) - sum of L(i, k) y(k)) / L(i, i), with
        the sum over the stored entries before the diagonal. The diagonal
        must be the last stored entry of each row, as it is for a factor
        from incomplete_cholesky.
 
        Parameters
        ----------
        L : CSR
            Lower-triangular matrix with a non-zero diagonal.
        r : ndarray, shape (n,)
 
        Returns
        -------
        list of float, length n
        """
        y = []
        for row in range(L.n) : 
            start = L.indptr[row]
            end = L.indptr[row+1]
            row_sum = 0 
            for position in range(start,end-1) : 
                row_sum += L.data[position] * y[L.indices[position]]
            y.append((r[row] - row_sum) / L.data[end-1])
        return y
    
    def transpose(self) : 
        """Return the transpose as a new CSR.
 
        CSR gives fast access to rows but not columns. The backward solve
        needs the columns of L, so L^T is built once and then read by rows.
 
        Returns
        -------
        CSR
            Entry (i, j) of the original becomes entry (j, i).
        """
        rows = []
        cols = []
        vals = []
        for row in range(self.n) :
            start = self.indptr[row]
            end = self.indptr[row+1]
            for position in range(start,end,1) :
                rows.append(row)
                cols.append(self.indices[position])
                vals.append(self.data[position])        
        return CSR(np.asarray(cols),np.asarray(rows),np.asarray(vals),self.n)
              
    def backward_solve(self,L_transposed,y) : 
        """Solve L^T z = y, from the bottom row up.
 
        Takes L^T itself, built once by the caller with transpose(), not
        L. L^T is upper-triangular, so the diagonal is the first stored
        entry of each row. For each row, z(i) = (y(i) - sum of L^T(i, k)
        z(k)) / L^T(i, i), with the sum over the entries after the diagonal.
 
        Parameters
        ----------
        L_transposed : CSR
            Upper-triangular matrix with a non-zero diagonal.
        y : array_like, shape (n,)
            Usually the output of forward_solve.
 
        Returns
        -------
        ndarray, shape (n,)
        """
        z = np.zeros(L_transposed.n)
        for row in range(L_transposed.n-1, -1, -1) : 
            start = L_transposed.indptr[row]
            end = L_transposed.indptr[row+1]
            row_sum = 0 
            for position in range(start+1,end) : 
                row_sum += L_transposed.data[position] * z[L_transposed.indices[position]]
            z[row] = ((y[row] - row_sum) / L_transposed.data[start])
        return z
                       
          