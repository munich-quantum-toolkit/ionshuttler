OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[2] c;
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
c[0] = measure q[0];
c[1] = measure q[1];
if (c[1]) { x q[2]; }
if (c[0]) { z q[2]; }
