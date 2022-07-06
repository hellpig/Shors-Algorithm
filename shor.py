#!/usr/bin/env python3.8
#
# This code finds non-trivial factors of N
# using Shor's algorithm...
#   https://www.youtube.com/watch?v=lvTqbM5Dq4Q
# where the above video's g is my a, and their p is my r.
#
# This code just uses NumPy (not Qiskit)!
# This causes it to run very fast with little RAM if not using many qubits,
#   but it's RAM usage and runtime increase faster with more qubits.
# Unlike Qiskit, this code doesn't do any approximations when adding a control
#   to a multiple-qubit gate defined by a unitary matrix.
#
# The following is good at providing the circuit,
# but it doesn't really explain how it works,
# and it writes a^2^j when it means a^(2^j),
# because a^2^j = (a^2)^j = a^(2*j).
#   https://qiskit.org/textbook/ch-algorithms/shor.html
# Changes I made to their circuit...
#  - I added in a test for if r is odd.
#  - My code also now works for a=1 and a = N-1.
#    Note that a=1 gives phase = 0, so r=1 is odd,
#    so it can be skipped.
#    Also, gcd(N,N) = N and gcd(N-2, N) = 1
#       https://math.stackexchange.com/questions/3038877/
#    so a=N-1 can always be skipped.
#  - You can now run any N!
#  - I removed the swaps from the QFTdagger
#    and reversed the order that the control bits connect
#
# 4 auxiliary qubits are needed to do mod15 arithmetic.
# These 4 qubits can do up to mod16 arithmetic (2^4 = 16).
# If they weren't entangled via control qubits that are
#   run through Hadamard gates,
#   these auxiliary qubits would be completely classical.
# Auxiliary register starts in 0b0001 = 1
# If control qubits are...
#    0b0...000, auxiliary becomes 1
#    0b0...001, auxiliary becomes a
#    0b0...010, auxiliary becomes a^2 % N
#    0b0...011, auxiliary becomes a^3 % N
#    etc.
# For a=7 and N=15, control bits after the first two don't entangle
#    anything because 7^4 % 15 = 1
#    so 7^8 = (7^4)^2 = 1, etc.
#    so n_count = 2 is enough.
# Since cycle lengths for N=15 are either 1, 2, or 4,
#    n_count = 2 is always enough for N=15.
# In general, a slightly bigger n_count is needed.
#    For example, 4^n % 9 makes a cycle of length 3,
#    causing you to get many possible measurements!
#
# Choose a,N that are coprime.
#    That is, a,N shouldn't share non-trivial prime factors.
#    If they aren't coprime, a^(2^j) % N will never be 1,
#    so a period doesn't really exist.
#    See the converse of Euler's theorem...
#      https://en.wikipedia.org/wiki/Euler%27s_theorem
#      https://en.wikipedia.org/wiki/Carmichael_function
#    Note that, for primes p1 and p2,
#      N = p1 * p2 is only coprime with a = p1 and a = p2,
#      so all you'd have to do is calculate N/a beforehand.
#      If N/a is an integer, you've found p1 and p2!
# In general, gcd(a,N) == 1 should be performed first on a. If it isn't 1,
#    you've found a factor, gcd(a,N). Luckily, gcd() is fast...
#      https://en.wikipedia.org/wiki/Euclidean_algorithm



import numpy as np
from fractions import Fraction
import time





# Set N, an odd integer such that N > 2
# Not prime N less than 128 are: 9, 15, 21, 25, 27, 33, 35, 39, 45, 49, 51, 55, 57, 63,
#   65, 69, 75, 77, 81, 85, 87, 91, 93, 95, 99, 105, 111, 115, 117, 119, 121, 123, 125
# If using 7+6 qubits, the above won't take more than 1.55 GiB.
#   The 13 qubits make a matrix, so 2^26 numbers.
#   On my computer, each np.csingle or np.double is 2^3 bytes,
#     so 2^29 bytes = 0.5 GiB for one matrix.
#   In general, RAM should be proportional to 4^(bits + n_count),
#     and be approximately 3 * 2^( 2 * (bits + n_count) + 3),
#     though NumPy will sometimes use some RAM-saving tricks for its arrays!
#   And 7+6 takes 6 minutes for each of the coprime a's
#     on my crappy 2-physical-core x86-64 computer.
#     Runtime seems to get be proportional to 8^(bits + n_count).

N = 63



# calculate bits needed for N
bits = 0
temp = N
while temp:
  bits += 1
  temp >>= 1




# Set the number of counting qubits.
# The ideal number for highest probability of success
#   SEEMS to be int(log2(N)) = bits - 1, based on my experimentation.
# Though, for N=15, n_counts >= 2 are all identical (bits - 1 = 3),
#    and, for N=51, n_counts >= 4 are all identical (bits - 1 = 5),
#    and, for N=85, n_counts >= 4 are all identical (bits - 1 = 6),
#    and, for N=255, n_counts >= 4 are all identical (bits - 1 = 7)
# The following N have bits-1 be the ideal for a=2: 9, 21, 25, 27, 33, 63, 65, 125
# Other than the n_count-greater-than-x-are-identical situation, all tested N agree.
# For many reasons, you might want less than bits - 1...
#  - a quantum computer will get too many errors with too many gates
#  - probability of success is not very low for n_count < bits - 1,
#    but it gets low very quickly after bits - 1
#  - a classical computer can use too much RAM and take too much time

n_count = bits - 1






# gates on a single qubit (global variables)
H = np.array([[1,1],[1,-1]]) / np.sqrt(2)  # Hadamard
I = np.array([[1,0],[0,1]]) * 1.0          # Identity
X = np.array([[0,1],[1,0]]) * 1.0          # NOT gate




# Unlike every other part of the circuit, this function is "magic" in that it
#   doesn't need to know the exact quantum logic gates needed to make the matrix.
#   The next step for me would be to make a general algorithm to create
#   np.linalg.matrix_power(unitaryM, power) from actual gates.
# In fact, using the fewest quantum logic gates may likely return a different matrix
#   with a different 0th column and with different Nth, (N+1)th, etc. columns.
#   Note that these columns are never actually used.

def amodN(a, power, bits, N):
    unitaryM = np.eye(1 << bits)
    for start in range(1,N):  # going all the way to (1 << bits) prevents unitaryM from being unitary
        final = (start*a)%N
        if start != final:   # does this IF statement reduce runtime?
            unitaryM[start, start] = 0.0
            unitaryM[final, start] = 1.0
    return np.linalg.matrix_power(unitaryM, power)



# Returns the matrix of a Controlled-Phase gate
# It's not a general function since it assumes the target is the final qubit
def CPhaseGate(theta, control, target):
    size = 1 << (target+1)
    output = np.eye( size ) * (1.0 + 0j)
    value = np.exp(1j * theta)
    for i in range(1, size, 2):
        if ( i >> (target - control) ) & 1:
            # will be true half the time, so 1/4 of nonzero elements of output[]
            output[i,i] = value
    return output


# Returns the matrix of the initial Hadamard gates
def make_multiH(n_count):
    output = H
    for i in range(1, n_count):
        output = np.kron(output, H)
    return output
#    return np.kron( np.eye( 1 << (n_count-1) ), X)
#    return np.kron(np.kron( np.eye( 1 << (n_count-2) ), X), np.eye(1 << 1))
#    return np.kron( X, np.eye( 1 << (n_count-1) ))



# Returns the matrix of the initial state being 0...001
def make_oneState(bits):
    return np.kron( np.eye( 1 << (bits-1) ), X)



# Returns the matrix of the QFT-dagger gate
# I don't do the SWAP gates because it's easier to simply
#   swap the order of the qubits that control the amodN gates.
def make_qft_dagger(n_count):    # n-qubit QFTdagger
    output = 1. + 0j
    for k in range(n_count):
        output = np.kron( output, I )
        for m in range(k):
            output = CPhaseGate(-np.pi/(1 << (k-m)), m, k) @ output
        output = np.kron( np.eye(1 << k), H ) @ output
    return output



# Returns the matrix of SWAP gates that flip an entire set of qubits.
# My circuit never uses this.
# It is used later for conveniently analyzing the results of the n_count qubits.
def getSwaps(n_count):
    length = 1 << n_count

    labels = [['0']*n_count for i in range(length)]
    backward = [['0']*n_count for i in range(length)]
    for b in range(n_count):  # b is binary digit
        for i in range(length):
            if ( (i >> b) & 1 ):
                labels[i][-b-1] = '1'
                backward[i][b] = '1'

    swaps = np.eye(length)
    for i in range(length-1):
        if (labels[i] == backward[i]):  # do this for speed
            continue
        for j in range(i+1,length):
            if (labels[i] == backward[j]):
                swaps[[i, j], :] = swaps[[j, i], :]
                break

    return swaps




def simulateShor(a, n_count, bits, N):

    print(' --starting simulation--', time.time(), flush=True)

    bigArray = np.kron(multiH, oneState)

    length = 1 << n_count
    step = 1 << bits
    for q in range(n_count):

        U = amodN(a, 1 << q, bits, N)

        # make bigNext be block-diagonal with Identities and U's
        bigNext = np.eye(1 << (n_count+bits))
        for i in range(length):
            if ( i >> q ) & 1:   # control U's starting at the bottom of the n_count qubits
                bigNext[ i*step:(i+1)*step, i*step:(i+1)*step ] = U

        # requires a lot of runtime!
        bigArray = bigNext @ bigArray

    # Complex numbers are now needed for only a tiny amount of runtime.
    #   np.cdouble (complex128 on my computer) uses more RAM,
    #   and it is slower than np.csingle (complex64 on my computer)
    #   perhaps due to RAM bandwidth
    bigArray = bigArray[:,0].astype(np.csingle)   # actually only the first column of bigArray (to speed things up!)
    bigNext = np.kron(qft_dagger.astype(np.csingle), np.eye(step, dtype=np.csingle))
    state = bigNext @ bigArray    # I don't need to grab the first column because bigArray is only a column!

    '''
    # For testing purposes...
    np.set_printoptions(threshold=np.inf)
    print(state)
    '''

    probs = np.absolute(state) ** 2         # now becomes np.double again

    # We now need to add up the probs of all auxiliary qubits
    #   then put the labels as right-to-left using swaps[]
    # We are imagining that the n_count qubits are being measured,
    #   and that the other qubits are not being measured.
    probs = swaps @ [sum( probs[ i*step : (i+1)*step ] ) for i in range(length)]

    print(' --simulation finished--', time.time(), flush=True)

    # qft_dagger can introduce very small numbers in place of what should be 0's
    return np.around(probs, 30)







print("\n N =", N)
print(" n_count =", n_count)


# make some relatively-small global arrays
multiH = make_multiH(n_count)
qft_dagger = make_qft_dagger(n_count)
swaps = getSwaps(n_count)
oneState = make_oneState(bits)



'''
# For testing purposes...
#matrix = CPhaseGate(np.pi, 0, n_count-1)
matrix = make_qft_dagger(n_count)
np.set_printoptions(threshold=np.inf)
print(np.around( matrix, 5))
exit()
'''


# In reality, you'd probably stop the algorithm once a factor was found,
#   but I make a loop over all a's (and all non-zero in probs[]) for analysis.
# Note that, for N=25, some a's give 0 probability of success
#   regardless of n_count (ones that have a cycle length of 4).
#   N=49 = 7^2 has many a's that give zero probability if n_count < bits,
#     and I read that you should check if N = prime^n before using Shor's,
#     else you aren't guaranteed that there will be any a-value that works.
#   N=121 = 11^2 also has many a's that give zero probability of success.
#     a's with cycle lengths of 110, 55, 10, and 5 give zero probability.
#   N=253 has many zero-probability-for-success a's
#     (whenever cycle length is 55 or 110).

#for a in [2]:
for a in range(2, N-1):

    print('\n\n      a =', a, '---------------------------------------', flush=True)

    # a,N must be coprime
    test = np.gcd(a,N)
    if test != 1:
        print("\n*** Non-trivial factor found: gcd(a,N) = %i ***" % test)
        continue

    probs = simulateShor(a, n_count, bits, N)

    '''
    # for testing purposes...
    np.set_printoptions(threshold=np.inf)
    print( np.around(probs, 5) )
    '''

    probSuccess = 0.0
    for i,p in enumerate(probs):
        if p > 0.0:

            phase = i/(1 << n_count)
            frac = Fraction(phase).limit_denominator(N-1)
            r = frac.denominator

            if r & 1:  # we only want even r
                continue

            #print("\n Phase =", frac)
            #print(" Probability =", np.around(p, 5))
            #print(" Result: r =", r)

            success = False
            guesses = [np.gcd(a**(r//2)-1, N), np.gcd(a**(r//2)+1, N)]
            print(" Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
            for guess in guesses:
                if guess not in [1,N] and (N % guess) == 0:
                    success = True
                    print("*** Non-trivial factor found: %i ***" % guess)

            if success:
                probSuccess += p

    print('\n for a =', a, ':', np.around(probSuccess, 5), 'were successful')
