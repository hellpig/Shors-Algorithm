#!/usr/bin/env python3.8
#
# For trying to figure out circuits to do times-a-mod-N math.
# This code is not a sophisticated approach! I do a brute-force search of all
#   NOT and SWAP gates and all ways of controlling NOT and SWAP gates.
# The code finds all ways of doing this that take the same minimal amount of gates.
# In this code, a CX gate (controlled NOT) on bits 0 and 2 and an X gate (NOT)
#   on bit 1 are counted as 2 gates even though, probably depending on specific
#   architecture of the quantum computer, they could be done at the same time
#   because they are done on different bits.
#
# The hope is to find some patterns that can be used for quickly
# finding large-N circuits.
#
# In the future, I may extend the code to do sqrt(X) and sqrt(X)-dagger gates.
# But I have no desire to reduce the number of bits by using patterns in the cycles
#   because why do Shor's algorithm if you already know the cycle lengths?
# Since the times-a-mod-N gates are to be controlled, I could also play around
#   with the idea of only controlling some of the gates in the times-a-mod-N gate.
#
# I believe SWAP and NOT gates and all the ways of controlling them are enough to
#   realize any times-a-mod-N gate.
#   The following is a complete set of transformations...
#      1↔2, 2↔3, 3↔4, ..., (N-2)↔(N-1)
#   as doing multiple transformations in a row allows us to shift the 1's and 0's
#      around in all necessary ways.
#   For example, for N=7, which requires 3 bits, 1↔2 would be the almost-diagonal matrix...
#      [[1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,0,1,0,0,0,0], ...]
#   Still looking at N=7, ...
#      1↔2 can be done with a SWAP and a CSWAP
#      2↔3 can be done with a CX and CCX
#      3↔4 can be done with a SWAP, CSWAP, CX, CCX, SWAP, and CSWAP
#      4↔5 can be done with a CX and CCX
#      5↔6 can be done with a CSWAP
#   The above transformations aren't practical, but it's proof that N < 8
#      can be done with SWAP and NOT gates. Though it may be more efficient to use
#      other gates also!




import numpy as np
from itertools import compress, product, combinations
np.set_printoptions(threshold=np.inf)


################################################
### set the following!
################################################

N = 15   # odd. N > 4
a = 2    # 1 < a < N-1   and   gcd(a,N) == 1

# Set j for a^(2^j) operators. I think of j=0 as the basic operator.
j = 0

# Try using an extra qubit? Probably never allows for using fewer gates??
extraBit = False



################################################
### do some math to find U
###   and to print useful things
################################################

print(" for N =", N, "and a =", a, "and j =", j)

# lazy redefinition of a
a = a**(1 << j)

# get bits needed for N
bits = 0
n = N
while n:
  bits += 1
  n >>= 1

if extraBit:
  bits += 1

'''
# make a matrix of size bits-by-bits to maybe find some patterns ???
M = np.zeros((bits,bits))
for i in range( bits - int(extraBit) ):
    start = 1 << i
    j = bits - i - 1
    M[:,j] = list("{:b}".format((start*a)%N).rjust(bits, '0'))
print(M)
'''

# print all the before and after states in binary form and calculate indices[]
U = np.eye(1 << bits)
indices = np.asarray([i for i in range(1 << bits)]) 
for start in range(1,N):
    final = (start*a)%N
    print(" ", "{:b}".format(start).rjust(bits, '0'), 'to', "{:b}".format(final).rjust(bits, '0') )
    indices[final] = start





################################################
### try to reproduce matrix U from SWAP and NOT
###   and various controlled versions of SWAP and NOT
################################################


# https://stackoverflow.com/a/6542458
def combinationsAll(items):
    return ( set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) )


# controlled-NOT gate
# Uses global variable bits
# None of the bits in function arguments should equal each other

def mcx(target, controlList=()):

    # count bits from the other direction
    target = bits - 1 - target
    controlList = [bits - 1 - i for i in controlList]

    indices = np.asarray([i for i in range(1 << bits)]) 
    for i in range( (1 << bits) - 1 ):

        go = True
        for b in controlList:
            if not ((i >> b) & 1):
                go = False

        if go and not ((i >> target) & 1):
            j = i + (1 << target)
            indices[[i, j]] = indices[[j, i]]

    return indices





# controlled-SWAP gate
# Uses global variable bits
# None of the bits in function arguments should equal each other

def mcswap(targetPair, controlList=()):

    # count bits from the other direction
    targetPair = [bits - 1 - i for i in targetPair]
    controlList = [bits - 1 - i for i in controlList]

    indices = np.asarray([i for i in range(1 << bits)]) 
    for i in range( (1 << bits) - 1 ):

        go = True
        for b in controlList:
            if not ((i >> b) & 1):
                go = False

        if go and not ((i >> targetPair[0]) & 1) and ((i >> targetPair[1]) & 1):
            j = i + (1 << targetPair[0]) - (1 << targetPair[1])
            indices[[i, j]] = indices[[j, i]]

    return indices



# lists of all possible gates (depends on bits)
controls = []
targets = []
gates = []

# mcx gates
for i in range(bits):   # loop over targets
    remainingBits = [j for j in range(bits) if j!=i]
    for j in combinationsAll(remainingBits):
        controls.append(j)
        targets.append(i)
        gates.append(mcx(i, j))

print('  number of possible mcx gates =', len(gates))

# mcswap gates
for i in combinations(range(bits), 2):   # loop over targets
    remainingBits = [j for j in range(bits) if j!=i[0] and j!=i[1]]
    for j in combinationsAll(remainingBits):
        controls.append(j)
        targets.append(i)
        gates.append(mcswap(i, j))

length = len(gates)
print('  number of possible gates =', length)

#print( [(controls[i], targets[i]) for i in range(len(targets))] )
#print(gates)




# brute force it!

indices = indices[1:N]  # only elements 1,2,...,N-1 need to match

if np.all( indices == np.asarray([i for i in range(1,N)])  ):
    print('  No gates needed other than the identity matrix.')
    exit()

# might be useful
#controlLengths = [len(i) for i in controls]
#isSWAP = [type(i) is tuple for i in targets]

num = 1  # initial number of gates to try
while num > 0:
    print('  num =', num)
    for arrangement in product(range(length), repeat=num):

        # Two gates in a row undo each other.
        # I should probably do more tests like this to remove more arrangements!
        if any(np.diff(arrangement)==0):    # to slightly help runtime
            continue

        guess = np.asarray([i for i in range(1 << bits)]) 
        for i in arrangement:
            guess = guess[gates[i]]
        if np.all( guess[1:N] == indices ):
            print(arrangement)
            for i in arrangement:
                print(' ', list(controls[i]), targets[i])
            num = -1

    num += 1







'''
I grabbed the following Qiskit codes from IBM Quantum Composer.
In their notation, the final argument(s) are the target,
  just like how I *print* it.


__mod5 for a=2__
circuit.x(qreg_q[0])
circuit.x(qreg_q[1])
circuit.cx(qreg_q[2], qreg_q[0])
circuit.cx(qreg_q[1], qreg_q[2])


__mod5 for a=3__
circuit.cx(qreg_q[1], qreg_q[2])
circuit.cx(qreg_q[2], qreg_q[0])
circuit.x(qreg_q[1])
circuit.x(qreg_q[0])


__mod5 for a = 2 or 3 when j=1__
circuit.x(qreg_q[0])
circuit.x(qreg_q[2])
circuit.cx(qreg_q[1], qreg_q[0])


__mod5 for j>1__
circuit.id(qreg_q[0])
circuit.id(qreg_q[1])
circuit.id(qreg_q[2])



__mod15 for a=2__
circuit.swap(qreg_q[0], qreg_q[1])
circuit.swap(qreg_q[1], qreg_q[2])
circuit.swap(qreg_q[2], qreg_q[3])


__mod15 for a=2 when j=1 (aka a=4 and j=0)__
circuit.swap(qreg_q[0], qreg_q[2])
circuit.swap(qreg_q[1], qreg_q[3])


__mod15 for a=2 when j>1__
circuit.id(qreg_q[0])
circuit.id(qreg_q[1])
circuit.id(qreg_q[2])
circuit.id(qreg_q[3])



__mod63 for a=2__    <-- I did this one by hand (not from the brute-force search)
circuit.swap(qreg_q[0], qreg_q[1])
circuit.swap(qreg_q[1], qreg_q[2])
circuit.swap(qreg_q[2], qreg_q[3])
circuit.swap(qreg_q[3], qreg_q[4])
circuit.swap(qreg_q[4], qreg_q[5])
'''


