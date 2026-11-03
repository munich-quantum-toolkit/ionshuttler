OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[2] c;
h q[0];
h q[1];
c[0] = measure q[0];
c[1] = measure q[1];
if (c == 3) { x q[2]; }
if (c[0] != c[1]) { z q[2]; }
