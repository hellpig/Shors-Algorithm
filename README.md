# Shor's Algorithm
Use NumPy instead of Qiskit to simulate a quantum computer to perform Shor's algorithm on any suitable integer.

I used NumPy because I found a bug in Qiskit, though NumPy and Qiskit codes give the same results in special cases where I could avoid Qiskit's bug. Testing Qiskit with NumPy has allowed me to greatly understand the math of quantum computers!

Each Python file is standalone. **See the comments at the top of each file! Especially read the top of shor.py for lots of good information!**
* **shor.py**: simulates the quantum computer to perform Shor's algorithm
* **shorScan.py**: classically finds cycle lengths of various N's (and *a*'s) instead of using Shor's quantum algorithm. This file is useful for understanding the results of Shor's algorithm
* **shorModCircuit.py**: for trying to figure out quantum circuits to do times-*a*-mod-N math. You can happily ignore this file!

Variables I use...
* N is the integer to be factored
* *a* is less than N and coprime with N
* r is the minimum power greater than 0 such that a^power mod N equals 1. r is the cycle length (aka the period).
* *bits* is the minimum number of qubits needed to represent N. For example, 8 < N < 16 have bits=4.
* n\_count is the number of qubits used in the inverse quantum Fourier transform. The total qubits is *bits* + n\_count

Great resources...
* [https://qiskit.org/textbook/ch-algorithms/shor.html](https://qiskit.org/textbook/ch-algorithms/shor.html)
* [https://en.wikipedia.org/wiki/Quantum_logic_gate](https://en.wikipedia.org/wiki/Quantum_logic_gate)
