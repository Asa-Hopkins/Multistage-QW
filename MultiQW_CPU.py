import pandas as pd
import itertools
import time
import cProfile
import pickle
import matplotlib.pyplot as plt

import numpy as np
import scipy

def grid(n):
    #Construct adjacency matrix for a grid of dimension n
    #It does this by reproducing a pattern that occurs down the diagonals

    #For dimension n, there are n diagonals with nonzero elements
    #These are the 2**i'th off-diagonals for every integer i up to n
    #The pattern is then that for the j'th diagonal, there are j -1's followed by j 0's
    #This repeats until the end of the diagonal
    N = 2**n
    diags = []
    offset = []
    for i in range(0,n):
        diags.append(-(np.indices([N - i - 1], dtype=np.int32) + 2**i & 2**i) // 2**i)
        offset.append(2**i)
    A = scipy.sparse.diags(diags,offsets = offset,format = 'csr', shape = (N, N), dtype = np.float32)
    A += A.T
    return A

def all_states(n):
    #Generates all possible states for a system of n spins.
    #Returns a 2D array of shape (2**n, n), where each row represents a different state.
    lists = []
    for i in range(n):
        lists.append([1,-1])
    return np.fromiter(itertools.chain.from_iterable(itertools.product(*lists)),int).reshape(-1,n)

def Cheb(f,n):
    #Calculates the nth order chebyshev interpolant of a given f
    x = np.cos((2*np.arange(1,n+1, dtype=np.longdouble) - 1) * np.pi/2/n)
    points = np.concatenate((f(x), np.zeros(n)))
    b = scipy.fft.ifft(points)
    b = 4*np.exp(1j*np.pi*np.arange(0,n)/2/n)*b[:n]
    b[0] /= 2
    return b[:n].real

def to_list(c, prec = 1e-16):
    #Truncates a polynomial to the desired precision
    last = 0
    for i in range(len(c)):
        if abs(c[i]) > prec:
            last = i
    return np.complex64(c[:last + 1])

def clenshawMt(c,x,v,t):
    #Evaluates p(x*t) @ v, where p is a polynomial in chebyshev form
    n = len(c)
    if len(v.shape) != 2:
      v2 = v[:,None] + t[None,:]*0
    else:
      v2 = v.copy()
    b1 = c[n-1]*v2
    b2 = b1*0
    for r in range(n-2, -1, -1):
        b1, b2 = (2 - (r == 0))*(x@(b1 * t[None,:])) - b2, b1
        b1 += v2*c[r]
    
    return b1

def iexpm2(A,v,t, onenorm = None):
    #Use 1-norm as upper bound on spectral radius
    if onenorm == None:
      B = A.copy()
      B.data = np.abs(B.data)
      onenorm = B.sum(axis=0).max()
      del(B)
    
    onenorm = float(np.max(t)*onenorm)

    f = lambda x: np.cos(x*onenorm)
    #Heuristic for how many points we need in our fft
    fft = 1<<int(onenorm*1.1+64).bit_length()
    cos = to_list(Cheb(f, fft))
    
    while len(cos) > fft - 4:
        #If too few terms were truncated, redo with more points
        fft *= 2
        cos = to_list(Cheb(f, fft))
        
    f = lambda x: np.sin(x*onenorm)
    sin = Cheb(f, fft)
    iexp = to_list([cos[i] + 1j*sin[i] for i in range(0,len(cos))])

    A.data = A.data/onenorm
    return clenshawMt(iexp,A,v,t)


def P_all(t, H_P, H_G, Psi0, e0, gammas, norms = None, measure = False):
  #Handles fixed time, infinite time, single stage and multi stage
  m = len(gammas)
  temp = np.copy(Psi0)
  
  if type(t) != type('inf'):
    #t contains a set of times to sample at
    temp = np.vstack([Psi0]*len(t[0])).T

    if Psi0.size < 2**9:
      #Diagonalisation is faster here
      for i in range(m):
        H = (gammas[i]*H_G.toarray() + H_P.toarray())/np.sqrt(1 + gammas[i]**2)
        diag, eigs = np.linalg.eigh(H)
        eigs = eigs.T
        temp = eigs @ temp
        d = np.exp(-1j * t[i,:,None] * diag[None,:]).T #Calculate diagonals for H at each t
        temp = (d * temp) #Apply H to Psi0
        temp = eigs.T @ temp
      res = np.abs(temp[e0])**2

    else:
      #Sparse methods are faster here
      for i in range(m):
        norm = (gammas[i]*norms[1] + norms[0])/np.sqrt(1 + gammas[i]**2)
        temp = iexpm2(scipy.sparse.csr_matrix((gammas[i]*H_G + H_P)/np.sqrt(1 + gammas[i]**2), dtype=np.float32), np.array(temp, dtype=np.complex64), np.array(t[i],dtype=np.float32))
        res = temp[e0].imag**2
        res += temp[e0].real**2
  
    if measure != False:
      E = np.sum(temp.conj() * (H @ temp), axis=0)
      return np.mean(res), np.mean(np.real(E))
    
    return np.mean(res)
  
  else:
    #Have to diagonalise for infinite time
    H = (H_P.toarray()[None,:,:] + H_G.toarray()[None,:,:]*gammas[:,None,None]) / np.sqrt(1 + gammas[:,None,None]**2)
    vals, eigs = np.linalg.eigh(H)
    
    eigs = eigs.transpose(0,2,1)
    
    if m == 1:
        #Infinite time, single stage probability
        #We can use the formula from the Callison paper to calculate P_inf
        res = eigs[0][:,e0] * (eigs[0] @ Psi0)
        res = np.sum(np.abs(res)**2)
        return res

      #vals = np.abs(eigs[0] @ Psi0)**2
      #return np.sum(vals * np.abs(eigs[0][:,e0]**2))
              
    vals = np.abs(eigs[0] @ Psi0)**2  
    mat = eigs[0].T @ (vals[:,None] * eigs[0])

    for n in range(1,m-1):
        vals = np.sum(eigs[n] * (eigs[n] @ mat.T), axis=1)
        mat = eigs[n].T @ (vals[:,None] * eigs[n])
    
    vals = np.sum(eigs[-1] * (eigs[-1] @ mat.T), axis=1)
    return np.sum(vals * np.abs(eigs[-1][:,e0]**2))

def heur(n):
  #Heuristic gamma for n qubits, from the Callison paper
  return 0.887 * (2 ** (1/2) * (n * (n+3))**0.5) * scipy.special.erfinv(1 - 1/2**n) / 2 / n


def SpinGlass2(n,m, inf_time = True, plot_energy = False, write = True, num = 2000):
    #n is the number of qubits in the spin glass
    #m is number of stages of the walk
    #inf_time chooses between infinite time averages and short time averages
    #plot_energy chooses between plotting the energy or not
    #write chooses between actually writing the output files or not
    N = 2**n
    H_G = grid(n)

    Psi0 = 1/np.sqrt(N)*np.ones(N)

    entries = pd.read_csv('./qwspinglass_data/sk_instances.csv')
    entries = np.array(entries)
    
    probs = []
    params = []
    times = []
    t = time.time()
    for numero in np.arange((n-5)*10000,(n-5)*10000 + num):
        
        J = np.load("./qwspinglass_data/sk_instances/"+entries[numero,0]+".Jmat.npy")
        h = -np.load("./qwspinglass_data/sk_instances/"+entries[numero,0]+".hvec.npy")
        J = -np.tril(J + J.T)/2
        states = all_states(n).T
        
        #For a given state, represented by a column vector, state.T @ J @ state gives its energy
        #Here we have all the states stacked into a matrix, if we did a normal matrix multiplications
        #we'd be doing unecessary calculations (e.g state[a].T @ J @ state[b], where a!=b)
        #To avoid this, the formula used below only evaluates the diagonal elements of the matrix multiplication
        
        H_P = np.sum(states * (J @ states), axis=0) + np.sum(states*h[:,None],axis = 0)
      
        arg = np.argmin(H_P) #Get index of lowest energy
        evl = H_P[arg] #The element at that index is the eigenenergy
        onenorm = np.max(np.abs(H_P)) #Used for calculating 1-norm
      
        state0 = np.zeros(N) #The vector with a 1 in that position is the eigenvector
        state0[arg] = 1
        
        H_P = scipy.sparse.diags(H_P, format = 'csr')
        Es = []
        points = []
        times = []
        
        gammas = heur(n)/np.tan(np.arange(1,m+1)*np.pi/(2*(m+1)))
    
        #For a given set of gammas, find the success probability
        Psit = np.copy(Psi0)
        Prob = []

        if inf_time:
            probs.append(P_all('inf',H_P, H_G, Psi0, arg, gammas, norms = [onenorm,n]))
        else:
            #Calculate the finite time average using the scaling calculated in my notes
            #The time for each stage is randomly chosen on the interval [scale, 2*scale]
            point = np.sqrt(2/(n+1))/np.tan(np.pi/(2*(m+1)))
            lower = point
            upper = lower
            times = lower + upper * np.random.rand(m,100)
            
            #probs.append(P4(times, vals, eigs.transpose(0,2,1), Psi0, arg, H_G)[0])

            probs.append(P_all(times,H_P, H_G, Psi0, arg, gammas, norms = [onenorm,n]))
        if numero & 7 == 0:
          print(numero, time.time() - t, probs[-1])
          t = time.time()
            
    if write == True:
        if inf_time == False:
            f1 = open(f'./Probs/shortavg{n}.{m}.arr','wb')
        else:
            f1 = open(f'./Probs/hprobs{n}.{m}.arr','wb')
        pickle.dump(probs,f1)
        f1.close()
    return np.mean(probs)

x = SpinGlass2(5,1, inf_time=False, write=False, num = 100)
