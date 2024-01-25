import pandas as pd
import itertools
import time
import cProfile
import pickle
import matplotlib.pyplot as plt

use_cupy = True

import numpy as np
import scipy

if use_cupy:
  import cupy as cp
  import cupyx.scipy as cscipy

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

def P(t,diag,eigs,Psi0,e0,H):
    #Fixed time, single stage probability 
    temp = eigs @ Psi0 #Rotate Psi0 into basis where H is diagonal
    #Note that because H is symmetric, the eigenbasis is orthogonal, so its inverse is its transpose
    diags = np.exp(-1j * t[:,None] * diag[None,:]) #Calculate diagonals for H at each t
    temp = (diags * temp[None,:]).T #Apply H to Psi0
    
    psit = eigs.T @ temp #rotate back, with a trick for optimisation    
    res = np.abs(psit[e0])**2
    
    E = np.sum(psit.conj() * (H @ psit), axis=0)
    return res, np.real(E)

def P2(eigs,Psi0,e0):
    #Infinite time, single stage probability
    #We can use the formula from the Callison paper to calculate P_inf
    res = eigs[:,e0] * (eigs @ Psi0)
    res = np.sum(np.abs(res)**2)
    return res

def P3(eigs,Psi0,e0):
    #Infinite time, multi stage probability
    #Uses the formula I derive in my LaTeX notes
    res = 0
    m = len(eigs) #Number of stages

    if m==1:
      return P2(eigs[0], Psi0, e0)

    #The formula here boils down to calculating eigenvalues, then constructing a matrix with those eigenvalues
    #Since we know the eigenvectors of the matrix, we rotate into a basis where it is diagonal
    #Special cases for first and last stage
    
    vals = np.abs(eigs[0] @ Psi0)**2
    mat = eigs[0].T @ (vals[:,None] * eigs[0])

    for n in range(1,m-1):        
        vals = np.sum(eigs[n] * (eigs[n] @ mat.T), axis=1)
        mat = eigs[n].T @ (vals[:,None] * eigs[n])
    
    vals = np.sum(eigs[-1] * (eigs[-1] @ mat.T), axis=1)
    return np.sum(vals * np.abs(eigs[-1][:,e0]**2))

def P4(t,diag,eigs,Psi0,e0,H):
    #Fixed time, multi stage probability
    m = len(eigs)
    temp = np.copy(Psi0)
    temp = np.vstack([Psi0]*len(t[0])).T
    for i in range(0,m):
        temp = eigs[i] @ temp
        d = np.exp(-1j * t[i,:,None] * diag[i][None,:]).T #Calculate diagonals for H at each t
        temp = (d * temp) #Apply H to Psi0
        temp = eigs[i].T @ temp
        res = np.abs(temp[e0])**2
    res = np.abs(temp[e0])**2
    
    E = np.sum(temp.conj() * (H @ temp), axis=0)
    return np.mean(res), np.mean(np.real(E))

def Cheb(f,n):
    #Calculates the nth order chebyshev interpolant of a given f
    x = np.cos((2*np.arange(1,n+1, dtype=np.longdouble) - 1) * np.pi/2/n)
    points = np.concatenate((f(x), np.zeros(n)))
    b = scipy.fft.ifft(points)
    b = 4*np.exp(1j*np.pi*np.arange(0,n)/2/n)*b[:n]
    b[0] /= 2
    return b[:n].real

def to_list(c, prec = 1e-16):
    last = 0
    for i in range(len(c)):
        if abs(c[i]) > prec:
            last = i
    return np.complex64(c[:last + 1])
@prof
def clenshawMt(c,x,v,t):
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
  
@prof
def iexpm2(A,v,t, onenorm = None):
    #Use 1-norm as upper bound on spectral radius
    if onenorm == None:
      B = A.copy()
      B.data = cp.abs(B.data)
      onenorm = B.sum(axis=0).max()
      del(B)
    
    onenorm = float(cp.max(t)*onenorm)

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
  
@prof
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
        temp = iexpm2(cscipy.sparse.csr_matrix((gammas[i]*H_G + H_P)/np.sqrt(1 + gammas[i]**2), dtype=cp.float32), cp.array(temp, dtype=cp.complex64), cp.array(t[i],dtype=cp.float32))
        #temp = expm(cscipy.sparse.csr_matrix(1j*(gammas[i]*H_G + H_P)/np.sqrt(1 + gammas[i]**2), dtype=cp.complex64), cp.array(temp, dtype=cp.complex64), cp.array(t[i],dtype=cp.float32))
        #temp = iexpm2(cscipy.sparse.csr_matrix((gammas[i]*H_G + H_P)/np.sqrt(1 + gammas[i]**2)), cp.array(temp), cp.array(t[i]))
        res = temp[e0].imag**2
        res += temp[e0].real**2
      res = temp.get()
    
    if measure != False:
      E = np.sum(temp.conj() * (H @ temp), axis=0)
      return np.mean(res), np.mean(np.real(E))
    
    return np.mean(res)
  
  else:
    #Have to diagonalise for infinite time
    H = (H_P[None,:,:] + H_G[None,:,:]*gammas[:,None,None]) / np.sqrt(1 + gammas[:,None,None]**2)
    vals, eigs = np.linalg.eigh(H)
    
    if m == 1:
      eigs = eigs.T
      vals = np.abs(eigs @ Psi0)**2
      return np.sum(vals * np.abs(eigs[:,e0]**2))
    
    eigs.transpose(0,2,1)
              
    vals = np.abs(eigs[0] @ Psi0)**2  
    mat = eigs[0].T @ (vals[:,None] * eigs[0])

    for n in range(1,m-1):
        vals = np.sum(eigs[n] * (eigs[n] @ mat.T), axis=1)
        mat = eigs[n].T @ (vals[:,None] * eigs[n])
    
    vals = np.sum(eigs[-1] * (eigs[-1] @ mat.T), axis=1)
    return np.sum(vals * np.abs(eigs[-1][:,e0]**2))
  
def QAOA(t,diag,eigs,Psi0,e0,H_P):
    #Uses QAOA instead of quantum walks
    temp = np.copy(Psi0)
    temp = np.vstack([Psi0]*len(t[0][0])).T
    for alpha, beta in t:
        temp = (np.exp(-1j * beta[None,:] * H_P[:,None]) * temp) #Apply H_P
        d = np.exp(-1j * alpha[None,:] * diag[:,None])
        temp = eigs @ temp
        temp = d*temp
        temp = eigs.T @ temp #Apply H_G
    E = np.sum(temp.conj() * (H_P[:,None] * temp), axis=0)
    return np.abs(temp[e0])**2, np.real(E)

def heur(n):
  #Heuristic gamma for n qubits, from the Callison paper
  return 0.887 * (2 ** (1/2) * (n * (n+3))**0.5) * scipy.special.erfinv(1 - 1/2**n) / 2 / n

@prof
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
        
        if plot_energy:
            for gamma in gammas:
              H = (gamma*H_G + H_P)/np.sqrt(1 + gamma**2)
              vals, eigs = np.linalg.eigh(H)
              point = np.sqrt(2/(n+1))
              if len(times)!=0:
                times.extend(times[-1] + np.linspace(0,point,100))
              else:
                times.extend(np.linspace(0,point,100))
              Es.extend(E(np.linspace(0,point,100),vals, eigs.T, Psit,arg,H_G))
              Prob.extend(P(np.linspace(0,point,100),vals, eigs.T, Psit,arg))
              plt.axvline(times[-1], linestyle = 'dashed', alpha = 0.6)
              points.append(point)
              Psit = eigs @ (np.exp(-1j * vals * point) * (eigs.T @ Psit))
            plt.plot(times,Es, label = "Energy")
            plt.plot(times, np.log2(Prob), label = "Log probability")
            plt.legend()
            plt.show()
            
        #H = (H_P.toarray()[None,:,:] + H_G.toarray()[None,:,:]*gammas[:,None,None]) / np.sqrt(1 + gammas[:,None,None]**2)
        #vals, eigs = np.linalg.eigh(H)

        if inf_time:
            probs.append(P3(eigs.transpose(0,2,1),Psi0,arg))
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
#    return np.mean(probs)

#MakePlot(10,5, use_QAOA = True, num = 50)
#cProfile.run('SpinGlass2(15,20, inf_time = False, write = False)', sort='cumtime')
SpinGlass2(17,3, inf_time=False, write=False, num = 10)
#for m in [1,2,3,4,5]:
#  for n in [16,17,18,19,20]:
#    print(n,m)
#    SpinGlass2(n,m, inf_time = False, write=True, num = 100)
