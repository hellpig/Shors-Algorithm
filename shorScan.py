#!/usr/bin/env python3.8
#
# Classically find cycle lengths of various N's and a's
#   instead of using Shor's quantum algorithm.
#
# For even cycle lengths, guess factors of N.
#   You can verify that this doesn't work for N = prime^n
#
# If any factors are found, calculate the minimum n_count
#   that would be necessary to find r using a quantum computer.
#   You might need even more than this if the probability of
#   getting certain states is zero (or very low), but you
#   need at least that amount. Of course, just because the
#   quantum computer cannot find r doesn't mean that it cannot
#   find primes.


from math import gcd
from fractions import Fraction


# https://en.wikipedia.org/wiki/Carmichael_function
# https://stackoverflow.com/questions/47761383/proper-carmichael-function
def carmichael(n):
        n=int(n)
        k=2
        a=1
        alist=[]

        while not ((gcd(a,n))==1):
            a=a+1

        while ((gcd(a,n))==1) & (a<=n) :
            alist.append(a)
            a=a+1
            while not ((gcd(a,n))==1):
                a=a+1

        timer=len(alist)
        while timer>=0:
            for a in alist:
                if (a**k)%n==1:
                    timer=timer-1
                    if timer <0:
                        break
                    pass
                else:
                    timer=len(alist)
                    k=k+1
        return k



# https://stackoverflow.com/a/22808285
def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors




# find minimum n_count necessary
def getNcount(r, N):
    n_count = 1
    for n_count in range(1,10000):
        maxResult = 1 << n_count
        for i in range(1, maxResult):
            if Fraction(i/maxResult).limit_denominator(N-1).denominator == r:
                return n_count
    return -1  # should never happen




for N in range(3, 1 << 8, 2):  # only odd N's

    factN = prime_factors(N)

    # skip the prime N's
    if len(factN) == 1:
        continue

    print('\n prime factors of N =', N, ':', factN , ", Carmichael's function:", carmichael(N))

    for a in range(2,N-1):

        # get cycle length if a,N are coprime
        '''
        factA = prime_factors(a)
        if (not (set(factA) & set(factN))):
        '''
        if ( gcd(a,N) == 1 ):

            # find period r
            r = 2
            last = a
            while True:
                last = (last*a) % N
                if (last == 1):
                    break
                r += 1

            # see if r gives factors and print
            if r&1:
                print('a =', a, ':', r)
            else:
                factorsObtained = []
                guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
                for guess in guesses:
                    if guess not in [1,N] and (N % guess) == 0:
                        factorsObtained.append(guess)
                if factorsObtained:
                    print('a =', a, ':', r, factorsObtained, ' n_count =', getNcount(r,N))
                else:
                    print('a =', a, ':', r, factorsObtained)

