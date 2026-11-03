OPENQASM 3.0;
include "stdgates.inc";
qubit[19] q;
bit[9] a;
bit[9] b;
a[0] = measure q[0];
a[1] = measure q[1];
a[2] = measure q[2];
a[3] = measure q[3];
a[4] = measure q[4];
a[5] = measure q[5];
a[6] = measure q[6];
a[7] = measure q[7];
a[8] = measure q[8];
b[0] = measure q[9];
b[1] = measure q[10];
b[2] = measure q[11];
b[3] = measure q[12];
b[4] = measure q[13];
b[5] = measure q[14];
b[6] = measure q[15];
b[7] = measure q[16];
b[8] = measure q[17];
int[8] ones = 0;
if (a[0]) { ones += 1; }
if (a[1]) { ones += 1; }
if (a[2]) { ones += 1; }
if (a[3]) { ones += 1; }
if (a[4]) { ones += 1; }
if (a[5]) { ones += 1; }
if (a[6]) { ones += 1; }
if (a[7]) { ones += 1; }
if (a[8]) { ones += 1; }
if (b[0]) { ones += 1; }
if (b[1]) { ones += 1; }
if (b[2]) { ones += 1; }
if (b[3]) { ones += 1; }
if (b[4]) { ones += 1; }
if (b[5]) { ones += 1; }
if (b[6]) { ones += 1; }
if (b[7]) { ones += 1; }
if (b[8]) { ones += 1; }
if (ones >= 9) { x q[18]; }
if (a == b) { x q[18]; }
if (ones - 3 < 2) { x q[18]; }
