# Shor's Algorithm
Use NumPy instead of Qiskit to simulate a quantum computer that performs Shor's algorithm on any suitable integer.

Each Python file is standalone. **See the comments at the top of each file!**
* shor.py: simulates the quantum computer
* shorScan.py: classically finds cycle lengths of various N's (and *a*'s) instead of using Shor's quantum algorithm
* shorModCircuit.py: for trying to figure out quantum circuits to do times-*a*-mod-N math

Symbols I use...
* N is the integer to be factored
* *a* is less than N and coprime with N
* r is the minimum power greater than 0 such that a^power mod N equals 1. r is the cycle length (aka the period).
