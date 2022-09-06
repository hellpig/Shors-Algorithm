#!/usr/bin/env python3.8
#
# This code finds non-trivial factors of N
# using Shor's algorithm...
#   https://www.youtube.com/watch?v=lvTqbM5Dq4Q
# where the above video's g is my a, and their p is my r.
#
# This code uses only NumPy (not Qiskit) to simulate a quantum computer!
# This makes the code much faster with much less RAM.
# Unlike Qiskit, this code doesn't do any approximations when adding a control
#   to a multiple-qubit gate defined by a unitary matrix.
#   Qiskit's approximate circuits were very complicated and would give
#   relative phases, which either had no effect on probability of finding a
#   factor or, believe it or not, increased it. I suppose these approximate
#   circuits were not polynomial in logN as Shor's algorithm should be.
#
# The following link is good at providing the circuit,
# but it doesn't really explain how it works,
# and note that a^2^j means a^(2^j).
# I used the following link to build my circuit...
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
#    then reversed the order that the control bits connect
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



import numpy as np    # pip install numpy
from fractions import Fraction
import time

from numba import jit, prange   # pip install numba

from math import gcd, log
# np.gcd(A,B) has a bug for A or B in interval [2^63, 2^64)
#   that causes the code to crash, so use math.gcd()


# Import the is_strong_bpsw_prp(N) function for probable prime testing.
# https://www.youtube.com/watch?v=jbiaz_aHHUQ&t=0s
# https://gmpy2.readthedocs.io/en/latest/advmpz.html#advanced-number-theory-functions
from gmpy2 import is_strong_bpsw_prp     # pip install gmpy2

# I'd use the ECPP method or APR-CL method if I wanted a proven prime test.
# Here is the only Python 3 code I could find...
#   https://github.com/wacchoz/APR_CL
#   from APR_CL import APRtest



####################################
## Set parameters
####################################


# Set N, an odd integer such that N > 2
# Not-prime odd N less than 128 are: 9, 15, 21, 25, 27, 33, 35, 39, 45, 49, 51, 55, 57, 63,
#   65, 69, 75, 77, 81, 85, 87, 91, 93, 95, 99, 105, 111, 115, 117, 119, 121, 123, 125
# If using 10+9 qubits (bits + n_count)...
#   The 19 qubits make a vector of 2^19 numbers.
#   On my computer, each np.csingle or np.double is 2^3 bytes,
#     so 2^22 bytes = 4 MiB for one vector.
#   In general, RAM should be proportional to 2^(bits + n_count).
#   And 10+9 takes 1 second for each of the coprime a's
#     on my crappy dual-core x86-64 CPU core.

N = 123



# calculate bits needed for N
if N <= 1 or not isinstance(N, int):   # make sure N isn't ridiculous
    print(' Error: do better!')
    exit()
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




# Should we "cheat" by classically finding the period for each of the a's?
# For a period T, this allows for a simpler faster cycle...
#   0,1,2,...,(T-1),0,1,...
# Of course, a quantum computer wouldn't use this speedup,
#   but simulating a quantum computer with a classical computer could.
# This "cheat" also reduces RAM usage.
# The resulting probabilities are not affected.

cheat = True



####################################
## Define functions
####################################


# Returns indices for performing the times-a-to-power-mod-N operator.
# This is for when cheat=False
def amodN(a, power, bits, N):
    a = (a ** power)%N
    indices = np.asarray([i for i in range(1 << bits)]) 
    for start in range(1,N):
        #indices[start] = (start*a)%N   # reverse cycling shouldn't make a difference
        indices[(start*a)%N] = start
    return indices



# Compile and multithread the calculations that take a lot of runtime.
# The first call to do_fast() will take longer due to compile time.
# Takes a bit more RAM though.
@jit(nopython=True, parallel=True)
def do_fast(state, stateOld, mult, length, step):
    for i in prange(length):     # rows of QFTdagger
        for j in range(length):  # columns of QFTdagger
            state[ i*step:(i+1)*step ] += np.exp(mult*i*j) * stateOld[ j*step:(j+1)*step ]



# for not "cheating"
def simulateShor(a, n_count, bits, N):

    print(' --starting simulation--', time.time(), flush=True)

    # take care of initial Hadamard gates and NOT gate
    state = np.zeros( 1 << (bits + n_count) )
    state[1::(1 << bits)] = 1.0 / np.sqrt(2) ** n_count

    # do the amodN() part of circuit
    length = 1 << n_count
    step = 1 << bits
    for q in range(n_count):

        U = amodN(a, 1 << q, bits, N)

        for i in range(length):
            if ( i >> q ) & 1:   # control U's starting at the bottom of the n_count qubits
                state[ i*step:(i+1)*step ] = state[ i*step:(i+1)*step ][U]

    del U

    # Complex numbers are now needed for the QFTdagger.
    #   np.cdouble (complex128 on my computer) uses more RAM,
    #   and it is slower than np.csingle (complex64 on my computer)
    #   perhaps due to RAM bandwidth
    # The double FOR loop is slow in Python.
    stateOld = state.astype(np.csingle)
    state = np.zeros(len(state), dtype=np.csingle)
    mult = np.array(-np.pi * 1.0j / (1 << (n_count - 1))).astype(np.csingle)
    do_fast(state, stateOld, mult, length, step)  # use Numba to fill state[]
    del stateOld
    state /= np.sqrt(length)

    '''
    # For testing purposes...
    np.set_printoptions(threshold=np.inf)
    print(state)
    '''

    probs = np.absolute(state).astype(np.double) ** 2       # now becomes np.double again

    # We now need to add up the probs of all auxiliary qubits
    # We are imagining that the n_count qubits are being measured,
    #   and that the other qubits are not being measured.
    probs = [sum( probs[ i*step : (i+1)*step ] ) for i in range(length)]

    print(' --simulation finished--', time.time(), flush=True)

    # qft_dagger can introduce very small numbers in place of what should be 0's
    return np.around(probs, 6)



# for "cheating"
# Compile and multithread the calculations that take a lot of runtime.
# The first call to do_fast_cheat() will take longer due to compile time.
# Takes a bit more RAM though.
@jit(nopython=True, parallel=True)
def do_fast_cheat(state, stateReduced, mult, length, T):
    for i in prange(length):     # rows of QFTdagger
        for j in range(length):  # columns of QFTdagger
            state[ i*T + stateReduced[j] ] += np.exp(mult*i*j)



# for "cheating"
def simulateShorCheat(a, n_count, N):

    print(' --starting simulation--', time.time(), flush=True)

    length = 1 << n_count

    # get period T
    x = [1,a]
    last = a
    while True:
        last = (last*a) % N
        x.append( last )
        if (last == 1):
            break
    T = len(x) - 1
    del x

    # I don't expect the following error to occur!
    if T >= (1 << 32):
        print("  ERROR: uint32 is not large enough.")
        exit()

    # take care of initial Hadamard gates and NOT gate
    stateReduced = np.zeros( length, dtype=np.uint32 )   # takes values in the interval [0,T)
    value = 1.0 / np.sqrt(2) ** n_count

    # do the times-a-to-power-mod-N part of circuit
    for q in range(n_count):   # 2^q is the power
        for i in range(length):
            if ( i >> q ) & 1:   # control U's starting at the bottom of the n_count qubits
                stateReduced[i] = ( stateReduced[i] + (1 << q) ) % T

    # Complex numbers are needed for the QFTdagger.
    #   np.cdouble (complex128 on my computer) uses more RAM,
    #   and it is slower than np.csingle (complex64 on my computer)
    #   perhaps due to RAM bandwidth
    # The double FOR loop is slow in Python.
    state = np.zeros(length * T, dtype=np.csingle)
    mult = np.array(-np.pi * 1.0j / (1 << (n_count - 1))).astype(np.csingle)
    T_numba = np.array(T).astype(np.uint32)
    do_fast_cheat(state, stateReduced, mult, length, T_numba)  # use Numba to fill state[]
    del stateReduced
    state *= value / np.sqrt(length)

    '''
    # For testing purposes...
    np.set_printoptions(threshold=np.inf)
    print(state)
    '''

    probs = np.absolute(state).astype(np.double) ** 2       # now becomes np.double again

    # We now need to add up the probs of all auxiliary qubits
    # We are imagining that the n_count qubits are being measured,
    #   and that the other qubits are not being measured.
    probs = [sum( probs[ i*T : (i+1)*T ] ) for i in range(length)]

    print(' --simulation finished--', time.time(), flush=True)

    # qft_dagger can introduce very small numbers in place of what should be 0's
    return np.around(probs, 6)



####################################
## Do Shor's Algorithm!
####################################

# N should not equal prime^k
# It's easier to check that N isn't any integer to a power, so I do that.
for k in range(2, int(log(N,3)) + 1):   # k=2 is all that is necessary if N = p1 * p2
    temp = round( N ** (1.0/k) )
    if N == temp**k:
        print('\n  ***** N =', temp, '^', k, '*****')
        exit()

# It is wise to first test if N is prime!
# N is provably not prime if the following function returns False.
if is_strong_bpsw_prp(N):
    print(' *** N =', N, 'is almost certainly prime ***')
    exit()

# N cannot be even.
if not N&1:
    print(' *** 2 is a factor of N ***')
    exit()

print("\n N =", N)
print(" n_count =", n_count)





# In practice, you'd probably stop the algorithm once a factor was found,
#   but, for analysis, I make a loop over all a's and all non-zero in probs[].
# If you instead want to get specific results,
#   do something like the following to get i and p from probs[]...
#     i = np.random.choice(range(len(probs)), p = probs/sum(probs))
#     p = probs[i]
#
# N=65 has a = 8,18,47,57 give 0 probability of success for n_count = bits - 1.
# N=77 has many a's that give 0 probability of success for n_count = bits - 1.
#   This happens if and only if the cycle length is 15 or 30.
#   n_count >= bits starts to get better probability of success for these a's,
#   and I read somewhere that n_count should approximately be 2*bits,
#   though they didn't say why.
#   I tested n_count >= bits for a = 3,4,17, but no higher than n_count=17...
#    - a=4 has r=15 (odd), and it peaks at n_count=8 with 4% probability of success
#    - a=17 has r=30, but the r,a pair doesn't produce any primes via gcd().
#        It seems to asymptote to 20% probability
#    - a=3 has r=30, and this r,a pair produces primes.
#        It seems to asymptote to 46.7% probability

#for a in [2]:
#for a in [3,4,17]:
for a in range(2, N-1):

    print('\n\n      a =', a, '---------------------------------------', flush=True)

    # a,N must be coprime
    test = gcd(a,N)
    if test != 1:
        print("\n*** Non-trivial factor found: gcd(a,N) = %i ***" % test)
        continue

    if cheat:
        probs = simulateShorCheat(a, n_count, N)
    else:
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
            guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
            #print(" Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
            for guess in guesses:
                if guess not in [1,N] and (N % guess) == 0:
                    success = True
                    #print("*** Non-trivial factor found: %i ***" % guess)

            if success:
                probSuccess += p

    print('\n for a =', a, ':', np.around(probSuccess, 5), 'were successful')
