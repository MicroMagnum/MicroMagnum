import numpy as np
import numpy.fft

class Sample: pass
topo = Sample()
topo.n = (32, 32, 1) # number of cells
topo.d = (1e-9, 1e-9, 1e-9) # cell size
topo.m = tuple(map(lambda i: 1 if i == 1 else 2*i, topo.n)) # size of zero-padded magnetization array.

def precompute_demag_tensor():
  n = topo.n
  m = topo.m
  dx, dy, dz = topo.d

  Nxx, Nxy, Nxz, Nyy, Nyz, Nzz = (np.zeros(m, np.float64) for _ in range(6))

  assert m[0] == 2*n[0]
  assert m[1] == 2*n[1]
  assert m[2] == n[2]

  import newell
  import itertools
  for nx, ny, nz in itertools.product(*map(range, m)):
    x = ((nx+n[0]) % (2*n[0])) - n[0]
    y = ((ny+n[1]) % (2*n[1])) - n[1]
    z = ((nz+n[2]) % (2*n[2])) - n[2]
    if abs(x) >= n[0] or abs(y) >= n[1] or abs(z) >= n[2]: continue

    Nxx[nx,ny,nz] = newell.Nxx(dx*x,dy*y,dz*z,dx,dy,dz)
    Nxy[nx,ny,nz] = newell.Nxy(dx*x,dy*y,dz*z,dx,dy,dz)
    Nxz[nx,ny,nz] = newell.Nxz(dx*x,dy*y,dz*z,dx,dy,dz)
    Nyy[nx,ny,nz] = newell.Nyy(dx*x,dy*y,dz*z,dx,dy,dz)
    Nyz[nx,ny,nz] = newell.Nyz(dx*x,dy*y,dz*z,dx,dy,dz)
    Nzz[nx,ny,nz] = newell.Nzz(dx*x,dy*y,dz*z,dx,dy,dz)

  topo.Bxx, topo.Bxy, topo.Bxz, topo.Byy, topo.Byz, topo.Bzz = map(numpy.fft.fftn, (Nxx, Nxy, Nxz, Nyy, Nyz, Nzz))
  print("real/imag absmaxs:")
  for B in (topo.Bxx, topo.Bxy, topo.Bxz, topo.Byy, topo.Byz, topo.Bzz):
    print(np.max(np.abs(np.real(B))), np.max(np.abs(np.imag(B))))

def stray_field(Mx, My, Mz):

  def zeropad(A):
    B = np.zeros((2*topo.n[0], 2*topo.n[1], 2*topo.n[2]), np.float32)
    B[0:topo.n[0], 0:topo.n[1], 0:topo.n[2]] = A[:,:,:]
    return B

  def unpad(B):
    A = np.zeros((topo.n[0], topo.n[1], topo.n[2]), np.float32)
    A[:,:,:] = np.real(B[0:topo.n[0], 0:topo.n[1], 0:topo.n[2]])
    return A

  Ex, Ey, Ez = map(zeropad, (Mx, My, Mz))
  Ex, Ey, Ez = map(numpy.fft.fftn, (Ex, Ey, Ez))
  Ex, Ey, Ez = (
    Ex*topo.Bxx + Ey*topo.Bxy + Ez*topo.Bxz,
    Ex*topo.Bxy + Ey*topo.Byy + Ez*topo.Byz,
    Ex*topo.Bxz + Ey*topo.Byz + Ez*topo.Bzz
  )
  Ex, Ey, Ez = map(numpy.fft.ifftn, (Ex, Ey, Ez))
  Hx, Hy, Hz = map(unpad, (Ex, Ey, Ez))
  return Hx, Hy, Hz

def exchange_field(Mx, My, Mz):
  pass

precompute_demag_tensor()
Mx, My, Mz = (np.ones(topo.n, np.float64) for x in range(3))
Hx, Hy, Hz = stray_field(Mx, My, Mz)
