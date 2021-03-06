# Shor's Algorithm
Use NumPy instead of Qiskit to simulate a quantum computer to perform Shor's algorithm on any suitable integer. My code uses much less RAM and runtime than Qiskit!

I initially used NumPy because I found a bug in Qiskit (introducing relative complex phases even if gates involve no complex numbers), though NumPy and Qiskit codes give the same results in special cases where I could avoid Qiskit's bug (the bug occurs when multi-qubit gates defined by a unitary matrix are controlled by another qubit). Testing Qiskit with NumPy has allowed me to greatly understand the math of quantum computers and create some powerful code!

Each Python file is standalone. **See the comments at the top of each file! Especially read the top of shor.py for lots of good information!**
* **shor.py**: simulates the quantum computer to perform Shor's algorithm. I use the Numba module to make it go very fast. I also let you "cheat" to make it even faster (see comments in the file).
* **shorSlow.py**: is an old version of shor.py that calculates and stores huge matrices (whereas shor.py does not create any 2D arrays). I provide it because, if you want to understand shor.py, you need to understand this first. I wrote what is now shorSlow.py based on Qiskit code (linked to below), then the modern shor.py took the readable shorSlow.py and made it *fast*.
* **shorScan.py**: classically finds cycle lengths of various N's (and *a*'s) instead of using Shor's quantum algorithm. This file is useful for understanding the results of Shor's algorithm.
* **shorModCircuit.py**: for trying to figure out quantum circuits to do times-*a*-mod-N math. You can happily ignore this file!

Variables I use...
* N is the integer to be factored
* *a* is less than N and coprime with N
* r is the minimum power greater than 0 such that a^power mod N equals 1. r is the cycle length (aka the period).
* *bits* is the minimum number of qubits needed to represent N. For example, 8 < N < 16 have bits=4.
* n\_count is the number of qubits used in the inverse quantum Fourier transform. The total qubits is *bits* + n\_count.

Great resources...
* [https://qiskit.org/textbook/ch-algorithms/shor.html](https://qiskit.org/textbook/ch-algorithms/shor.html)
* [https://en.wikipedia.org/wiki/Quantum_logic_gate](https://en.wikipedia.org/wiki/Quantum_logic_gate)
