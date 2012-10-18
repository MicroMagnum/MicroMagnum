#ifndef DEMAG_OLD_H
#define DEMAG_OLD_H

#include "config.h"
#include "matrix/matty.h"

Matrix calculateDemagTensor_old(long double lx, long double ly, long double lz, int nx, int ny, int nz, int ex, int ey, int ez, int repeat_x, int repeat_y, int repeat_z);

#endif
