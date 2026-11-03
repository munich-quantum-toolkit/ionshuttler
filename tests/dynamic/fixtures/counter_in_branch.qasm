OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[2] m = "00";
int[8] n = 0;
bit stop = 0;
m[0] = measure q[0];
if (m[0]) { stop = 1; }
if (stop) { x q[2]; } else {
  m[1] = measure q[1];
  n = n + 1;
  if (m[1]) { n = n + 1; }
  if (n >= 2) { x q[2]; }
}
