#Emanuel Casiano-Diaz
#Tridiagonalizes a real symmetric matrix through Lanczos iterations then finds its eigenvalues via 
#Shifted Inverse Power Iteration

#References:
#http://qubit-ulm.com/wp-content/uploads/2012/04/Lanczos_Algebra.pdf
#http://users.ece.utexas.edu/~sanghavi/courses/scribed_notes/Lecture_13_and_14_Scribe_Notes.pdf
#Numerical Analysis, T.Sauer, Chapter 12

import numpy as np
from scipy.linalg import norm, solve
from scipy.sparse.linalg import eigsh
from numpy.linalg import qr
from pytictoc import TicToc

"-------------------------------------------------------------------------"

def SymmMat(D):
    '''Generates a Random Real Symmetric Matrix of dimensions DxD'''
    A = np.zeros([D,D])
    for i in range(D):
        for j in range(i+1):
            A[i,j] = np.random.random()
            A[j,i] = A[i,j]   #Symmetry condition
    return A
    
"-------------------------------------------------------------------------"

def IPI(A, maxiter=1000, tol=1E-06):
    '''Obtain smallest eigenvalue and corresponding eigenvector via Inverse Power Iteration with Shift'''
    n = A.shape[0]
    #xprev = np.random.randn(n)        #Initial Vector for I.P.I
    x = np.ones(n)                #Initial Vector for I.P.I
    s = -3000                      #Shift (Want smallest eigenvalue, but not necessarily in magnitude)
    B = A - s*np.identity(n)          #Shifted matrix (Positive Definite)
    for j in range(1,maxiter):
        u = x/norm(x)     #Normalize the vector x
        x = solve(B,u)
        mu = u.T @ x         #Smallest magnitude eigenvalue of B=A-sI
    
    lam = 1/mu + s    #Minimum eigenvalue of A
    return(lam)
    
"-------------------------------------------------------------------------"

def LanczosTri(A):
    '''Tridiagonalize Matrix A via Lanczos Iterations'''
    
    #Check if A is symmetric
    if((A.transpose() != A).all()):
        print("WARNING: Input matrix is not symmetric")
    n = A.shape[0]
    x = np.ones(n)                     #Random Initial Vector
    V = np.zeros(n*n).reshape(n,n)     #Tridiagonalizing Matrix
    #Begin Lanczos Iteration
    q = x/np.linalg.norm(x)
    V[:,0] = q
    r = A @ q
    a1 = q.T @ r
    r = r - a1*q
    b1 = norm(r)
    s_min = 0     #Initialize minimum eigenvalue
    #print("a1 = %.12f, b1 = %.12f"%(a1,b1))
    for j in range(2,n+1):
        v = q
        q = r/b1
        V[:,j-1] = q
        r = A @ q - b1*v
        a1 = q.T @ r
        r = r - a1*q
        b1 = norm(r)
        if b1 == 0: break #Need to reorthonormalize
            
    #Tridiagonal matrix similar to A
    T = V.T @ A @ V
    
    #Normalize via Frobenius Norm
    alpha = norm(T)/norm(A)
    T = T/alpha
        
    return T
"-------------------------------------------------------------------------" 

def main():
    #Create the Matrix to be tri-diagonalized
    #n = 12                       #Size of input matrix (nxn)
    #A = SymmMat(n)              #Input matrix. (Hermitian)
    
    #Hamiltonian of tV Model for L = 4, N = 2, ell=2
    A = -1.0*np.array(((0,1,0,1,0,0),
                       (1,0,1,0,1,1),
                       (0,1,0,1,0,0),
                       (1,0,1,0,1,1),
                       (0,1,0,1,0,0),
                       (0,1,0,1,0,0)))
    
    #A = -1.0*np.array(((0,0,1,0),
    #                   (0,0,1,0),
    #                   (1,1,0,1),
    #                   (0,0,1,0)))
    
    
    #Change print format to decimal instead of scientific notation
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    
    #Transform the matrix A to tridiagonal form via Lanczos
    T = LanczosTri(A)
    
    #Find Eigenvalues for Real, Symmetric, Tridiagonal Matrix via QR Iteration
    t2 = TicToc()
    t2.tic()
    lam = IPI(T,maxiter=50000)
    t2.toc()
    print("Eigs(T): ", lam)
    
    #Get eigenpairs of untransformed hermitian matrix A and time the process using blackbox function
    t1 = TicToc()
    t1.tic()
    e_gs_A, gs_A = eigsh(A,k=1,which='SA',maxiter=1000)
    #e_gs_A = NSI(A,maxiter=1000)
    t1.toc()
    print("Eigs(A): ", e_gs_A[0])
    #print("Eigs(A): ",np.sort(e_gs_A[:-1]))
    
    
    
if __name__ == '__main__':
    main()




