from math import sqrt, log, atan, pi

def f(x,y,z):
  x, y, z = abs(x), abs(y), abs(z)
  x2, y2, z2 = x*x, y*y, z*z
  R = sqrt(x2 + y2 + z2)
  
  res = 0.0
  if (x2 + z2 > 0): res += 0.5*y * (z2-x2) * log((y+R) / sqrt(x2+z2))
  if (x2 + y2 > 0): res += 0.5*z * (y2-x2) * log((z+R) / sqrt(x2+y2))
  if (x*R > 0): res -= x * y * z * atan((y * z) / (x * R))
  res += 1.0/6.0 * (2*x2-y2-z2) * R
  return res

def g(x,y,z):
  z = abs(z)
  x2, y2, z2 = x*x, y*y, z*z
  R = sqrt(x2 + y2 + z2)

  res = -(x * y * R / 3);
  if (x2+y2 > 0): res += (x * y * z)             * log((z+R) / sqrt(x2+y2));
  if (x2+z2 > 0): res += (x / 6) * (3 * z2 - x2) * log((y+R) / sqrt(x2+z2));
  if (y2+z2 > 0): res += (y / 6) * (3 * z2 - y2) * log((x+R) / sqrt(y2+z2));
  if (abs(x * R) > 0): res -= ((z * x2) / 2) * atan((y * z) / (x * R));
  if (abs(y * R) > 0): res -= ((z * y2) / 2) * atan((x * z) / (y * R));
  if (abs(z * R) > 0): res -= ((z2 * z) / 6) * atan((x * y) / (z * R));
  return res

def Nxx(x,y,z,dx,dy,dz):
  return sum([
    -1 * f(x+dx,y+dy,z+dz),
    -1 * f(x+dx,y-dy,z+dz),
    -1 * f(x+dx,y-dy,z-dz),
    -1 * f(x+dx,y+dy,z-dz),
    -1 * f(x-dx,y+dy,z-dz),
    -1 * f(x-dx,y+dy,z+dz),
    -1 * f(x-dx,y-dy,z+dz),
    -1 * f(x-dx,y-dy,z-dz),
     2 * f(x,y-dy,z-dz),
     2 * f(x,y-dy,z+dz),
     2 * f(x,y+dy,z+dz),
     2 * f(x,y+dy,z-dz),
     2 * f(x+dx,y+dy,z),
     2 * f(x+dx,y,z+dz),
     2 * f(x+dx,y,z-dz),
     2 * f(x+dx,y-dy,z),
     2 * f(x-dx,y-dy,z),
     2 * f(x-dx,y,z+dz),
     2 * f(x-dx,y,z-dz),
     2 * f(x-dx,y+dy,z),
    -4 * f(x,y-dy,z),
    -4 * f(x,y+dy,z),
    -4 * f(x,y,z-dz),
    -4 * f(x,y,z+dz),
    -4 * f(x+dx,y,z),
    -4 * f(x-dx,y,z),
     8 * f(x,y,z)
  ]) / (4.0 * pi * dx * dy * dz);

def Nyy(x,y,z,dx,dy,dz):
  return Nxx(y, x, z, dy, dx, dz)

def Nzz(x,y,z,dx,dy,dz):
  return Nxx(z, y, x, dz, dy, dx)

def Nxy(x,y,z,dx,dy,dz):
  return sum([
    -1 * g(x-dx,y-dy,z-dz) +
    -1 * g(x-dx,y-dy,z+dz) +
    -1 * g(x+dx,y-dy,z+dz) +
    -1 * g(x+dx,y-dy,z-dz) +
    -1 * g(x+dx,y+dy,z-dz) +
    -1 * g(x+dx,y+dy,z+dz) +
    -1 * g(x-dx,y+dy,z+dz) +
    -1 * g(x-dx,y+dy,z-dz) +
     2 * g(x,y+dy,z-dz) +
     2 * g(x,y+dy,z+dz) +
     2 * g(x,y-dy,z+dz) +
     2 * g(x,y-dy,z-dz) +
     2 * g(x-dx,y-dy,z) +
     2 * g(x-dx,y+dy,z) +
     2 * g(x-dx,y,z-dz) +
     2 * g(x-dx,y,z+dz) +
     2 * g(x+dx,y,z+dz) +
     2 * g(x+dx,y,z-dz) +
     2 * g(x+dx,y-dy,z) +
     2 * g(x+dx,y+dy,z) +
    -4 * g(x-dx,y,z) +
    -4 * g(x+dx,y,z) +
    -4 * g(x,y,z+dz) +
    -4 * g(x,y,z-dz) +
    -4 * g(x,y-dy,z) +
    -4 * g(x,y+dy,z) +
     8 * g(x,y,z)
  ]) / (4.0 * pi * dx * dy * dz)

def Nxz(x,y,z,dx,dy,dz):
  return Nxy(x, z, y, dx, dz, dy)

def Nyz(x,y,z,dx,dy,dz):
  return Nxy(y, z, x, dy, dz, dx)
