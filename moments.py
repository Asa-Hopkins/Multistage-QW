import numpy as np
import time
import itertools

#The moment to be calculated
m = 3

def all_states(n):
    #Generates all possible states for a system of n spins.
    #Returns a 2D array of shape (2**n, n), where each row represents a different state.
    lists = []
    for i in range(n):
        lists.append([1,-1])
    return np.fromiter(itertools.chain.from_iterable(itertools.product(*lists)),int).reshape(-1,n)

def int_part(n):
    #Integer partition code from the user skovorodkin on stackoverflow
    #https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield tuple(a[: k + 2])
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield tuple(a[: k + 1])

def flatten(xss):
    return [x for xs in xss for x in xs]

#Idea - instead of pairing with all numbers, we can start always with a 0 in the first pair
#then, all unused indices of equal repetition are completely equivalent, so we pick the lowest one
        
def generate_unique_pairings(numbers):
    results = []
    def backtrack(remaining, current_pairs, used):
        # If no numbers are left, we've created a complete grouping
        if sum(remaining) == 0:
            # Sort pairs and make them canonical
            results.append(current_pairs)
            return
        
        # Take the first number
        first = 0
        while remaining[first] == 0:
            first += 1
        remaining[first] -= 1
        used[first] = 1
            
        # Try pairing it with each other number
        for i in range(first + 1, len(remaining)):
            second = i
            if used[second] == 0 and remaining[second] == remaining[second - 1]:
                continue
            
            if remaining[second] == 0:
                continue
            
            new_pairs = current_pairs + [(first, second)]
            remaining[second] -= 1

            
            old = used[second]
            used[second] = 1
            backtrack(np.copy(remaining), new_pairs, np.copy(used))
            used[second] = old
            remaining[second] += 1
            
    used = numbers*0
    backtrack(numbers, [], used)
    return results

pairings = []
t = time.time()

for ints in int_part(m):
    if ints[-1] > m//2:
        #Impossible to avoid self-pairing
        continue
    
    numbers = np.array(ints)*2
    
    pairings.extend(generate_unique_pairings(numbers))
    
print(time.time() - t, len(pairings))
t = time.time()
unique = {}

#Generate an Ising problem
#We restrict values to keep condition number small if possible
J = np.random.randint(-2,2,(m,m))
J = J + J.T
J -= np.diag(np.diag(J))

states = all_states(m).T

for pairing in pairings:
    args = [a for pair in pairing
            for a in [J,pair]]
    result = int(np.einsum(*args, optimize=True))
    unique[result] = pairing

print(time.time() - t, len(unique))
t = time.time()

#We now have only the unique pairings, most likely
num = len(unique)

#Finally, we set up a least squares problem to find the contribution from each
#We recycle the one we just calculated too
lsq_l = np.zeros((num,num))
lsq_r = np.zeros(num)

lsq_l[0] = list(unique)

H_P = np.sum(states * (J @ states), axis=0)
lsq_r[0] = np.sum(H_P**m)

for i in range(1,num):
    #Generate problem
    J = np.random.randint(-2,2,(m,m))
    J = J + J.T
    J -= np.diag(np.diag(J))

    H_P = np.sum(states * (J @ states), axis=0)
    lsq_r[i] = np.sum(H_P**m)
    #Calculate all the valid pairings
    for n,j in enumerate(unique):
        args = [a for pair in unique[j]
            for a in [J,pair]]
        result = np.int64(np.einsum(*args, optimize=True))
        lsq_l[i,n] = result
print(time.time() - t, len(pairings))
t = time.time()
       
a = np.linalg.solve(lsq_l, lsq_r)

print(time.time() - t)
t = time.time()

#print(a / 2**m)
print(np.int64(np.rint(a/2**m)))
print(unique)
