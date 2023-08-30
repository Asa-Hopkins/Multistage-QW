import pandas as pd
import itertools
import time
import cProfile
import pickle
import matplotlib.pyplot as plt

use_cupy = False
use_sparse_approx = False
#Switching to sparse matrices is faster at n = 9 and above for me
#Setting both the above to True doesn't work, cupy doesn't have the sparse.linalg.expm_multiply function

import numpy
if use_cupy:
  import cupy as np
  import cupyx.scipy as scipy
else:
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

def P(t,diag,eigs,Psi0,e0,H):
    temp = eigs @ Psi0 #Rotate Psi0 into basis where H is diagonal
    #Note that because H is symmetric, the eigenbasis is orthogonal, so its inverse is its transpose
    diags = np.exp(-1j * t[:,None] * diag[None,:]) #Calculate diagonals for H at each t
    temp = (diags * temp[None,:]).T #Apply H to Psi0
    
    psit = eigs.T @ temp #rotate back, with a trick for optimisation    
    res = np.abs(psit[e0])**2
    
    E = np.sum(psit.conj() * (H @ psit), axis=0)
    return res, np.real(E)

def P2(eigs,Psi0,e0):
    #We can use the formula from the Callison paper to calculate P_inf
    res = eigs[:,e0] * (eigs @ Psi0)
    res = np.sum(np.abs(res)**2)
    return res

def P3(eigs,Psi0,e0):
    #This calculates the infinite time average for the multi-stage walk
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
    #Does a multi-stage quantum walk but using fixed times
    m = len(eigs)
    temp = np.copy(Psi0)
    temp = np.vstack([Psi0]*len(t[0])).T
    for i in range(0,m):
        temp = eigs[i] @ temp
        d = np.exp(-1j * t[i][:,None] * diag[i][None,:]).T #Calculate diagonals for H at each t
        temp = (d * temp) #Apply H to Psi0
        temp = eigs[i].T @ temp
    res = np.abs(temp[e0])**2
    
    E = np.sum(temp.conj() * (H @ temp), axis=0)
    return np.mean(res), np.mean(np.real(E))

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

def find_smallest_input(func, target):
    # Start by doubling/halving N until the output is below the target value
    a = 1
    start = func(a)
    d = start < target
    while d == (start < target):
        a *= 2 - 1.5*d
        start = func(a)
    a *= 1 + d

    # Now use binary search to find the smallest input for which the output is below the target value
    low, high = a/2, a
    while (high - low)/high > 0.2: #We want 10% error on the result
        mid = (low + high) / 2
        if func(mid) >= target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def SpinGlass2(n,m, heuristic = True, inf_time = True, plot_energy = False, write = True):
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
    if not use_sparse_approx:
        H_G = np.array(H_G.toarray(), np.float32)
                
    probs = []
    params = []
    times = []
    for numero in np.arange((n-5)*10000,(n-5)*10000 + 2000):
        
        J = np.load("./qwspinglass_data/sk_instances/"+entries[numero,0]+".Jmat.npy")
        h = -np.load("./qwspinglass_data/sk_instances/"+entries[numero,0]+".hvec.npy")
        J = -np.tril(J + J.T)/2
        states = all_states(n).T
        
        #For a given state, represented by a column vector, state.T @ J @ state gives its energy
        #Here we have all the states stacked into a matrix, if we did a normal matrix multiplications
        #we'd be doing unecessary calculations (e.g state[a].T @ J @ state[b], where a!=b)
        #To avoid this, the formula used below only evaluates the diagonal elements of the matrix multiplication
        
        H_P = np.sum(states * (J @ states), axis=0) + np.sum(states*h[:,None],axis = 0)
        #scale = (n * np.sum(H_P**2)/N + np.sum(H_P @ H_G @ H_P)/N)
      
        arg = np.argmin(H_P) #Get index of lowest energy
        evl = H_P[arg] #The element at that index is the eigenenergy
      
        state0 = np.zeros(N) #The vector with a 1 in that position is the eigenvector
        state0[arg] = 1
        
        if not use_sparse_approx:
            H_P = np.array(np.diag(H_P), dtype = np.float32)
        else:
            H_P = scipy.sparse.diags(H_P, format = 'csr')
        Es = []
        points = []
        times = []
        def obj(gammas):
            #For a given set of gammas, return the success probability
            #We in fact return -Prob as this can then be fed into an optimiser
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
            H = (H_P[None,:,:] + H_G[None,:,:]*gammas[:,None,None]) / np.sqrt(1 + gammas[:,None,None]**2)
            vals, eigs = np.linalg.eigh(H)

            if inf_time:
                return -P3(eigs.transpose(0,2,1),Psi0,arg)
            else:
                #Calculate the finite time average using the scaling calculated in my notes
                #The time for each stage is randomly chosen on the interval [scale, 2*scale]
                point = np.sqrt(2/(n+1))
                lower = point
                upper = lower
                Prob.append(P4(lower + upper * np.random.rand(100,m), vals, eigs.transpose(0,2,1), Psi0, arg))
                return -Prob[-1]
        
        if heuristic:
            g = heur(n)/np.tan(np.arange(1,m+1)*np.pi/(2*(m+1)))
            probs.append(-obj(g))
        else:
            gbounds = [*[(0,3)]*m]
            bounds = [*gbounds]
            bounds = list(zip(*bounds))
            bounds = scipy.optimize.Bounds(*bounds)
            x = scipy.optimize.shgo(obj, bounds, iters=3)
            probs.append(-x['fun'])
            params.append(x['x'])
        print(probs[-1])
        die
    if write == True:
        if inf_time == False:
            f1 = open(f'./Probs/shortavg{n}.{m}.arr','wb')
        elif heuristic == True:
            f1 = open(f'./Probs/hprobs{n}.{m}.arr','wb')
        else:
            f1 = open(f'./Probs/probs{n}.{m}.arr','wb')
            f2 = open(f'./Probs/params{n}.{m}.arr','wb')
            pickle.dump(params, f2)
            f2.close()
        pickle.dump(probs,f1)
        f1.close()
    return np.mean(probs)

def MakePlot(n, m, use_QAOA = False, exact = True, samples = 50, num = 100):
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
    if not use_sparse_approx:
        H_G = np.array(H_G.toarray(), np.float32)
                
    probs = []
    params = []
    bounds = []
    
    for numero in np.arange((n-5)*10000,(n-5)*10000 + 1):
        
        J = np.load("./qwspinglass_data/sk_instances/"+entries[numero,0]+".Jmat.npy")
        h = -np.load("./qwspinglass_data/sk_instances/"+entries[numero,0]+".hvec.npy")
        print(entries[numero,:])
        J = -np.tril(J + J.T)/2
        states = all_states(n).T
        
        #For a given state, represented by a column vector, state.T @ J @ state gives its energy
        #Here we have all the states stacked into a matrix, if we did a normal matrix multiplications
        #we'd be doing unecessary calculations (e.g state[a].T @ J @ state[b], where a!=b)
        #To avoid this, the formula used below only evaluates the diagonal elements of the matrix multiplication
        
        H_P = np.sum(states * (J @ states), axis=0) + np.sum(states*h[:,None],axis = 0)
        #scale = (n * np.sum(H_P**2)/N + np.sum(H_P @ H_G @ H_P)/N)
      
        arg = np.argmin(H_P) #Get index of lowest energy
        evl = H_P[arg] #The element at that index is the eigenenergy
      
        state0 = np.zeros(N) #The vector with a 1 in that position is the eigenvector
        state0[arg] = 1
        
        if not use_sparse_approx:
            H_P = np.array(np.diag(H_P), dtype = np.float32)
        else:
            H_P = scipy.sparse.diags(H_P, format = 'csr')
        points = []
        times = []
        N = num
        
        probs = np.zeros((N,N))
        E = np.zeros((N,N))
        
        if use_QAOA:
            if m == 1:
                alphas = x = np.linspace(0,np.pi,N)
                betas = y = np.linspace(0,np.pi,N)
                eigs = np.linalg.eigh(H_G)
                for i in range(0,N):
                    print(i)
                    t = [[np.ones(N)*alphas[i],betas]]
                    probs[:,i], E[:,i] = QAOA(t, eigs[0], eigs[1].T, Psi0, arg, np.diag(H_P))
            else:
                #Seems to stretch exponentially in the x axis for larger number of stages
                #May indicate a more natural choice of heuristic?
                g0 = x = np.logspace(0,3,N)
                t0 = y = np.linspace(0,3*np.pi,N)
                eigs = np.linalg.eigh(H_G)
                for i in range(0,N):
                  print(i)
                  t = []
                  gamma = g0[i]
                  #t_driver = gamma * t_problem
                  #t0 = t_driver + t_problem = t_problem * (1 + gamma)
                  for j in range(0,m):
                      gamma = g0[i] * m/(j+1)
                      t_problem = t0/(1 + gamma)
                      t_driver = gamma * t_problem
                      t.append([t_driver, t_problem])
                  probs[:,i], E[:,i] = QAOA(t, eigs[0], eigs[1].T, Psi0, arg, np.diag(H_P))
                  
                      
        else:
            if m==1:
                gammas = np.linspace(0,10,N)
                x = gammas
                y = gammas
                
                for i in range(0,N):
                  print(i)
                  eigs = np.linalg.eigh(H_G*i*10/N + H_P)
                  probs[i,:], E[i,:] = P(np.linspace(0,10,N),eigs[0],eigs[1].T,Psi0,arg,H_P)
            
            if m == 2:
              gammas = x = np.linspace(0,10,N)
              y = gammas[:N//2]
              H = (H_G[None,:,:] + H_P[None,:,:]*gammas[:,None,None])
              eigs = np.linalg.eigh(H)
              vals, eigs = eigs[0], eigs[1].transpose(0,2,1)
              probs = np.zeros((N//2,N))
              E = np.zeros((N//2,N))
              for i in range(0,N):
                print(i)
                for j in range(0,N//2):
                  Psit = np.copy(Psi0)
                  
                  var1 = np.stack([eigs[j],eigs[i]])
                  var2 = np.stack([vals[j],vals[i]])
                  if exact:
                      probs[j,i] = P3(var1,Psi0,arg)
                  else:
                      upper = np.random.rand(m,samples)*10
                      probs[j,i], E[j,i] = P4(upper, var2,var1, Psi0, arg, H_P)

            if m > 2:
                g = x = np.linspace(0,10,N)
                dg = y = np.linspace(0,1,N)
                
                upper = np.random.rand(m,50)*10
                for i in range(0,N):
                  print(i)
                  for j in range(0,N):
                    gammas = g[i]*(1-dg[j])**np.arange(0,m)
                    gammas = gammas[::-1]
                    Psit = np.copy(Psi0)
                    H = (H_G[None,:,:] + H_P[None,:,:]*gammas[:,None,None])
                    eigs = np.linalg.eigh(H)
                    vals, eigs = eigs[0], eigs[1].transpose(0,2,1)
                    if exact:
                        probs[j,i] = P3(eigs,Psi0,arg)
                    else:
                        upper = np.random.rand(m,100)*10
                        probs[j,i], E[j,i] = P4(upper, vals, eigs, Psi0, arg, H_P)
        plt.contourf(x,y,probs, levels = 10)
        plt.colorbar(label="Success Probability")
        if m == 2 and use_QAOA == False:
            x1, y1 =  heur(n)/np.tan(np.arange(1,2+1)*np.pi/(2*(2+1)))
            plt.scatter(x1,y1,marker='x', color="black", alpha=0.5, linewidths=1)
        if m > 1 and use_QAOA == True:
            plt.semilogx()
        plt.show()
        
        plt.contourf(x,y,E, levels = 10)
        plt.colorbar(label="Energy")
        plt.show()

MakePlot(10,5, use_QAOA = True, num = 50)
#SpinGlass2(5,5)
