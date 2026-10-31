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
    
    





