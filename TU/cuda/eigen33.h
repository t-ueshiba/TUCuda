#pragma once

#include "TU/cuda/Array++.h"
#include "TU/cuda/vec.h"
#include <thrust/device_vector.h>

namespace TU
{
namespace cuda
{
#if defined(__NVCC__)
namespace device
{
template <class T>
static constexpr T	epsilon = std::numeric_limits<T>::epsilon();
    
__device__ inline float	 sqr(float x)		    { return x*x; }
__device__ inline float	 abs(float x)		    { return fabsf(x); }
__device__ inline float	 sqrt(float x)		    { return sqrtf(x); }
__device__ inline float	 rsqrt(float x)		    { return rsqrtf(x); }
__device__ inline float	 sin(float x)		    { return sinf(x); }
__device__ inline float	 cos(float x)		    { return cosf(x); }
__device__ inline float	 atan2(float y, float x)    { return atan2f(y, x); }

__device__ inline double sqr(double x)		    { return x*x; }
__device__ inline double abs(double x)		    { return fabs(x); }
__device__ inline double sqrt(double x)		    { return sqrt(x); }
__device__ inline double rsqrt(double x)	    { return rsqrt(x); }
__device__ inline double sin(double x)		    { return sin(x); }
__device__ inline double cos(double x)		    { return cos(x); }
__device__ inline double atan2(double y, double x)  { return atan2(y, x); }

template <class T> __device__ inline vec<T, 3>
cardano(const mat3x<T, 3>& A)
{
  // Determine coefficients of characteristic poynomial. We write
  //       | a   d   f  |
  //  A =  | d*  b   e  |
  //       | f*  e*  c  |
    const T de = A.x.y * A.y.z;	    // d * e
    const T dd = sqr(A.x.y);	    // d^2
    const T ee = sqr(A.y.z);	    // e^2
    const T ff = sqr(A.x.z);	    // f^2
    const T m  = A.x.x + A.y.y + A.z.z;
    const T c1 = (A.x.x*A.y.y + A.x.x*A.z.z + A.y.y*A.z.z)
	       - (dd + ee + ff);    // a*b + a*c + b*c - d^2 - e^2 - f^2
    const T c0 = A.z.z*dd + A.x.x*ee + A.y.y*ff - A.x.x*A.y.y*A.z.z
	       - T(2) * A.x.z*de;   // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

    const T p = m*m - T(3)*c1;
    const T q = m*(p - (T(3)/T(2))*c1) - (T(27)/T(2))*c0;
    const T sqrt_p = sqrt(abs(p));

    T phi = T(27) * (T(0.25)*sqr(c1)*(p - c1) + c0*(q + T(27)/T(4)*c0));
    phi = (T(1)/T(3)) * atan2(sqrt(abs(phi)), q);

    constexpr T	M_SQRT3 = 1.73205080756887729352744634151;	// sqrt(3)
    const T	c = sqrt_p*cos(phi);
    const T	s = (T(1)/M_SQRT3)*sqrt_p*sin(phi);

    vec<T, 3>	w;
    w.y  = (T(1)/T(3))*(m - c);
    w.z  = w.y + s;
    w.x  = w.y + c;
    w.y -= s;

    return w;
}

template <class T> __device__ void
tridiagonal33(const mat3x<T, 3>& A,
	      mat3x<T, 3>& Qt, vec<T, 3>& d, vec<T, 3>& e)
// ----------------------------------------------------------------------------
// Reduces a symmetric 3x3 matrix to tridiagonal form by applying
// (unitary) Householder transformations:
//            [ d[0]  e[0]       ]
//    A = Q . [ e[0]  d[1]  e[1] ] . Q^T
//            [       e[1]  d[2] ]
// The function accesses only the diagonal and upper triangular parts of
// A. The access is read-only.
// ---------------------------------------------------------------------------
{
    set_zero(Qt);
    Qt.x.x = Qt.y.y = Qt.z.z = T(1);

  // Bring first row and column to the desired form
    const T	h = sqr(A.x.y) + sqr(A.x.z);
    const T	g = (A.x.y > 0 ? -sqrt(h) : sqrt(h));
    e.x = g;

    T		f = g * A.x.y;
    T		omega = h - f;
    if (omega > T(0))
    {
	const T	uy = A.x.y - g;
	const T	uz = A.x.z;

	omega = T(1) / omega;

	f    = A.y.y * uy + A.y.z * uz;
	T qy = omega * f;			// p
	T K  = uy * f;				// u* A u
	f    = A.y.z * uy + A.z.z * uz;
	T qz = omega * f;			// p
	K   += uz * f;				// u* A u

	K  *= T(0.5) * sqr(omega);
	qy -= K * uy;
	qz -= K * uz;

	d.x = A.x.x;
	d.y = A.y.y - T(2)*qy*uy;
	d.z = A.z.z - T(2)*qz*uz;

      // Store inverse Householder transformation in Q
	f = omega * uy;
	Qt.y.y -= f*uy;
	Qt.z.y -= f*uz;
	f = omega * uz;
	Qt.y.z -= f*uy;
	Qt.z.z -= f*uz;

      // Calculate updated A.y.z and store it in e.y
	e.y = A.y.z - qy*uz - uy*qz;
    }
    else
    {
	d.x = A.x.x;
	d.y = A.y.y;
	d.z = A.z.z;

	e.y = A.y.z;
    }
}

namespace detail
{
  template <class T> __device__ inline bool
  is_zero(T e, T g)
  {
      return abs(e) + g == g;
  }

  template <class T> __device__ T
  init_offdiagonal(T w, T w0, T w1, T e0)
  {
      const auto	t = (w1 - w0)/(e0 + e0);
      const auto	r = sqrt(sqr(t) + T(1));
      return w - w0 + e0/(t + (t > 0 ? r : -r));
  }
    
  template <class T> __device__ inline void
  diagonalize(vec<T, 3>& w, vec<T, 3>& e, vec<T, 3>& q0, vec<T, 3>& q1,
	      T& c, T& s, T vec<T, 3>::* i0, T vec<T, 3>::* i1)
  {
      const auto	x = e.*i1;
      const auto	y = s*e.*i0;
      const auto	z = c*e.*i0;
      if (abs(x) > abs(y))
      {
	  const auto	t = y/x;
	  const auto	r = sqrt(sqr(t) + T(1));
	  e.*i1 = x*r;
	  c     = T(1)/r;
	  s     = c*t;
      }
      else
      {
	  const auto	t = x/y;
	  const auto	r = sqrt(sqr(t) + T(1));
	  e.*i1 = y*r;
	  s     = T(1)/r;
	  c     = s*t;
      }

      const auto	v = s*(w.*i0 - w.*i1) + T(2)*c*z;
      const auto	p = s*v;
      w.*i0 -= p;
      w.*i1 += p;
      e.*i0  = c*v - z;

    // Form eigenvectors
      const auto	q_tmp = q0;
      (q0 *= c) -= s*q1;
      (q1 *= c) += s*q_tmp;
  }

  template <class T> __device__ inline vec<T, 3>
  eigen_vector(vec<T, 3> a, vec<T, 3> b, vec<T, 3> w, T vec<T, 3>::* xyz)
  {
      a.x -= w.*xyz;
      b.y -= w.*xyz;
      return cross(a, b);
  }
}	// namespace detail
    
template <class T> __device__ bool
qr33(const mat3x<T, 3>& A, mat3x<T, 3>& Qt, vec<T, 3>& w)
{
  // Transform A to real tridiagonal form by the Householder method
    vec<T, 3>	e;	// The third element is used only as temporary
    device::tridiagonal33(A, Qt, w, e);

  // Calculate eigensystem of the remaining real symmetric tridiagonal matrix
  // with the QL method
  //
  // Loop over all off-diagonal elements
    for (int n = 0; n < 2; ++n)	// n = 0, 1
    {
	for (int nIter = 0; ; )
	{
	    int	i = n;
	    if (i == 0)
		if (detail::is_zero(e.x, abs(w.x) + abs(w.y)))
		    e.x = T(0);
		else
		    ++i;
	    if (i == 1)
		if (detail::is_zero(e.y, abs(w.y) + abs(w.z)))
		    e.y = T(0);
		else
		    ++i;
	    if (i == n)
		break;
	    
	    if (nIter++ >= 30)
		return false;

	  // [n = 0]: i = 1, 2; [n = 1]: i = 2
	    if (n == 0)
		if (i == 1)
		    e.y = detail::init_offdiagonal(w.y, w.x, w.y, e.x);
		else
		    e.z = detail::init_offdiagonal(w.z, w.x, w.y, e.x);
	    else
		e.z = detail::init_offdiagonal(w.z, w.y, w.z, e.y);

	    auto	s = T(1);
	    auto	c = T(1);
	  // [n = 0, i = 1]: i = 0; [n = 0, i = 2]: i = 1, 0
	  // [n = 1, i = 2]: i = 1
	    while (--i >= n)
		if (i == 1)
		    detail::diagonalize(w, e, Qt.y, Qt.z, c, s,
					&vec<T, 3>::y, &vec<T, 3>::z);
		else
		    detail::diagonalize(w, e, Qt.x, Qt.y, c, s,
					&vec<T, 3>::x, &vec<T, 3>::y);
	}
    }

    return true;
}

template <class T> __device__  inline bool
eigen33(const mat3x<T, 3>& A, mat3x<T, 3>& Qt, vec<T, 3>& w)
{
    w = cardano(A);		// Calculate eigenvalues

    auto	t = abs(w.x);
    auto	u = abs(w.y);
    if (u > t)
	t = u;
    if ((u = abs(w.z)) > t)
	t = u;
    if (t < T(1.0))
	u = t;
    else
	u = sqr(t);
    const auto	error = T(256) * epsilon<T> * sqr(u);

    Qt.x = detail::eigen_vector(A.x, A.y, w, &vec<T, 3>::x);
    auto	norm = dot(Qt.x, Qt.x);

  // If vectors are nearly linearly dependent, or if there might have
  // been large cancellations in the calculation of A[i][i] - w.x, fall
  // back to QL algorithm
  // Note that this simultaneously ensures that multiple eigenvalues do
  // not cause problems: If w.x = w.y, then A - w.x * I has rank 1,
  // i.e. all columns of A - w.x * I are linearly dependent.
    if (norm <= error)
    	return qr33(A, Qt, w);
    else                      // This is the standard branch
    {
	Qt.x *= rsqrt(norm);
    }

  // Calculate second eigenvector by the formula
  //   v.y = (A - w.y).e1 x (A - w.y).e2
    Qt.y = detail::eigen_vector(A.x, A.y, w, &vec<T, 3>::y);
    norm  = dot(Qt.y, Qt.y);

    if (norm <= error)
    	return qr33(A, Qt, w);
    else
    {
	Qt.y *= rsqrt(norm);
    }

  // Calculate third eigenvector according to
  //   v.z = v.x x v.y
    Qt.z = cross(Qt.x, Qt.y);

    return true;
}

/************************************************************************
*  __global__ functions							*
************************************************************************/
template <class T> __global__ void
test_tridiagonal33(const mat3x<T, 3>* A,
		   mat3x<T, 3>* Qt, vec<T, 3>* d, vec<T, 3>* e)
{
    device::tridiagonal33(*A, *Qt, *d, *e);
}

template <class T> __global__ void
test_qr33(const mat3x<T, 3>* A, mat3x<T, 3>* Qt, vec<T, 3>* w)
{
    device::qr33(*A, *Qt, *w);
}

template <class T> __global__ void
test_eigen33(const mat3x<T, 3>* A, mat3x<T, 3>* Qt, vec<T, 3>* w)
{
    device::eigen33(*A, *Qt, *w);
}

}	// namespace device
#endif

template <class T> vec<T, 3>
tridiagonal33(const mat3x<T, 3>& A,
	      mat3x<T, 3>& Qt, vec<T, 3>& d, vec<T, 3>& e)
{
    using mat_t	 = mat3x<T, 3>;
    using vec_t	 = vec<T, 3>;

    thrust::device_vector<mat_t>	A_d{1, A};
    thrust::device_vector<mat_t>	Qt_d(1);
    thrust::device_vector<vec_t>	d_d(1);
    thrust::device_vector<vec_t>	e_d(1);

    device::test_tridiagonal33<<<1, 1>>>(get(&A_d[0]), get(&Qt_d[0]),
					 get(&d_d[0]), get(&e_d[0]));

    thrust::copy(Qt_d.begin(), Qt_d.end(), &Qt);
    d = d_d[0];
    e = e_d[0];
}

template <class T> vec<T, 3>
qr33(const mat3x<T, 3>& A, mat3x<T, 3>& Qt)
{
    using mat_t	= mat3x<T, 3>;
    using vec_t	= vec<T, 3>;

    thrust::device_vector<mat_t>	A_d(1, A);
    thrust::device_vector<mat_t>	Qt_d(1);
    thrust::device_vector<vec_t>	w_d(1);

    device::test_qr33<<<1, 1>>>(get(&A_d[0]), get(&Qt_d[0]), get(&w_d[0]));

    thrust::copy(Qt_d.begin(), Qt_d.end(), &Qt);
    vec_t	w;
    thrust::copy(w_d.begin(), w_d.end(), &w);

    return w;
}

template <class T> vec<T, 3>
eigen33(const mat3x<T, 3>& A, mat3x<T, 3>& Qt)
{
    using mat_t	= mat3x<T, 3>;
    using vec_t	= vec<T, 3>;

    thrust::device_vector<mat_t>	A_d(1, A);
    thrust::device_vector<mat_t>	Qt_d(1);
    thrust::device_vector<vec_t>	w_d(1);

    device::test_eigen33<<<1, 1>>>(get(&A_d[0]), get(&Qt_d[0]), get(&w_d[0]));

    thrust::copy(Qt_d.begin(), Qt_d.end(), &Qt);
    vec_t	w;
    thrust::copy(w_d.begin(), w_d.end(), &w);

    return w;
}

}	// namespace cuda
}	// namespace TU
