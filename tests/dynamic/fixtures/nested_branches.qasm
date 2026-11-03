OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[2] c = "00";
h q[0];
c[0] = measure q[0];
if (c[0]) {
  h q[1];
  c[1] = measure q[1];
  if (c[1]) { x q[2]; } else { cx q[1], q[2]; }
}
cx q[0], q[2];
