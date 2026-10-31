import numpy as np 

class CSR :
    def __init__ (self,rows,cols,vals,n) :
       self.indptr, self.indices,self.data = self.coo_to_csr(rows,cols,vals,n)
       self.n = int(n)
       
    @staticmethod
    def coo_to_csr (rows,cols,vals, n) : 
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
        matrix = np.zeros((self.n,self.n))
        for index in range(self.n) : 
            start = self.indptr[index]
            end = self.indptr[index+1]
            matrix[index,self.indices[start:end]] = self.data[start:end]
        return matrix
    
    def matvec(self , x) : 
        y = np.zeros(self.n)
        for index in range(self.n) : 
            start = self.indptr[index]
            end = self.indptr[index+1]
            y[index] = self.data[start:end] @ x[self.indices[start:end]]
        return y 
    
    





