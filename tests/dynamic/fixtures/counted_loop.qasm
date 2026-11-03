OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[2] c = "00";
for int r in [0:2] {
  reset q[2];
  cx q[0], q[2];
  cx q[1], q[2];
  c[0] = measure q[2];
  if (c[0]) { x q[0]; }
}
