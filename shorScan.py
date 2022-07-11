#!/usr/bin/env python3.8
#
# Classically find cycle lengths of various N's and a's
#   instead of using Shor's quantum algorithm.
#
# For even cycle lengths, I also try to guess factors of N.
#   You can verify that this doesn't work for N = prime^n


from math import gcd


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

            # find period
            x = [1,a]
            last = a
            while True:
                last = (last*a) % N
                x.append( last )
                if (last == 1):
                    break
            r = len(x) - 1   # period

            # see if r gives factors and print
            if r&1:
                print('a =', a, ':', r)
            else:
                factorsObtained = []
                guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
                for guess in guesses:
                    if guess not in [1,N] and (N % guess) == 0:
                        factorsObtained.append(guess)
                print('a =', a, ':', r, factorsObtained)

