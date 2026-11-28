OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
reset q[0];
h q[0];
cx q[0], q[1];
c[1] = measure q[0];
if (c[0]) { x q[1]; }
