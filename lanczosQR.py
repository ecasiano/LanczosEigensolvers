#Emanuel Casiano-Diaz
#Tridiagonalizes a real symmetric matrix through Lanczos iterations #then finds its eigenvalues via QR Algorithm

#References:
#http://qubit-ulm.com/wp-content/uploads/2012/04/Lanczos_Algebra.pdf
#http://users.ece.utexas.edu/~sanghavi/courses/scribed_notes/Lecture_13_and_14_Scribe_Notes.pdf
#Numerical Analysis, T.Sauer, Chapter 12

import numpy as np
from scipy.linalg import norm
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

def NSI(A, tol=1E-13, maxiter=5000):
    '''Obtain Eigendecomposition of matrix A via Normalized Simultaneous QR Iteration in k steps'''
    
    #Get Eigenvalues, Eigenvectors
    m = A.shape[0]
    Q = np.identity(m)
    residual = 1
    lprev = np.ones(m)
    ctr = 0
    while norm(residual) > tol:
        Q,R = qr(A@Q)
        lam = np.diagonal(Q.T @ A @ Q) #Rayleigh Quotient Matrix
        residual = norm(lprev - np.sort(lam))
        lprev = np.sort(lam)
        ctr += 1
        if ctr == maxiter: break
    #print(ctr)
        
    return(lam)
    
"-------------------------------------------------------------------------"

def LanczosTri(A):
    '''Tridiagonalize Matrix A via Lanczos Iterations'''
    
    #Check if A is symmetric
    if((A.transpose() != A).any()):
        print("WARNING: Input matrix is not symmetric")
    n = A.shape[0]
    x = np.ones(n)              #Random Initial Vector
    V = np.zeros((n,1))                 #Tridiagonalizing Matrix

    #Begin Lanczos Iteration
    q = x/norm(x)
    V[:,0] = q
    r = A @ q
    a1 = q.T @ r
    r = r - a1*q
    b1 = norm(r)
    b1=1
    ctr = 0
    #print("a1 = %.12f, b1 = %.12f"%(a1,b1))
    for j in range(2,n+1):
        v = q
        q = r/b1
        #V[:,j-1] = q
        #V = np.hstack((V,np.reshape(q,(n,1))))
        r = A @ q - b1*v
        a1 = q.T @ r
        r = r - a1*q
   
        V = np.hstack((V,np.reshape(q,(n,1))))

        #Reorthogonalize
        V = qr(V)[0]

        b1 = norm(r)
        ctr+=1
        print("|V.T@V - I| = ")
        print(np.abs((V.T@V)-np.eye(j)))
        if b1 == 0: 
            print("WARNING: Lanczos ended due to b1 = 0")
            return V #Need to reorthonormalize
        
        #print(np.trace(V.T@V)/j)
    #Check if V is orthonormal
    print("|V.T@V - I| = ")
    print(np.abs((V.T@V)-np.eye(n)))
    if((V.T@V != np.eye(n)).any()):
        print("WARNING: V.T @ V != I: Orthonormality of Transform Lost")
        
    #Reorthonormalize
    #V = qr(V)[0]
    
    #Tridiagonal matrix similar to A
    T = V.T @ A @ V
    #print(T)
        
    #Normalize via Frobenius Norm
    #alpha = norm(T)/norm(A)
    #T = T/alpha
    
    return T
"-------------------------------------------------------------------------" 

def main():
    #Create the Matrix to be tri-diagonalized
    n = 50                       #Size of input matrix (nxn)
    A = SymmMat(n)               #Input matrix. (Hermitian)
    
    #Hamiltonian of tV Model for L = 4, N = 2, ell=2
    #A = -1.0*np.array(((0,1,0,1,0,0),
    #                   (1,0,1,0,1,1),
    #                   (0,1,0,1,0,0),
    #                   (1,0,1,0,1,1),
    #                   (0,1,0,1,0,0),
    #                   (0,1,0,1,0,0)))
    
    #Test Sparse Matrix
    #A = 1.0*np.diag((1,2,3,4,5,6))
    #A[-1,0] = 5
    #A[0,-1] = 5
    
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
    lam = NSI(T)

    t2.toc()
    print("Eigs(T) (NSI): ", np.sort(lam)[:-1][0])
    
    #Get eigenpairs of untransformed hermitian matrix A and time the process using blackbox function
    t1 = TicToc()
    t1.tic()
    e_gs_T, gs_T = eigsh(T,k=n-1,which='SA',maxiter=1000)
    #e_gs_A = NSI(A,maxiter=1000)
    t1.toc()
    print("Eigs(T) (np.eigsh): ", e_gs_T[0])
    #print("Eigs(A): ",np.sort(e_gs_A[:-1]))
    
    t3 = TicToc()
    t3.tic()
    e_gs_A, gs_A = eigsh(A,k=n-1,which='SA',maxiter=1000)
    #e_gs_A = NSI(A,maxiter=1000)
    t3.toc()
    print("Eigs(A) (np.eigsh): ", e_gs_A[0])
    #print("Eigs(A): ",np.sort(e_gs_A[:-1]))
        
if __name__ == '__main__':
    main()