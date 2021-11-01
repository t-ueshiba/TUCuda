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
    
    
__host__ __device__ inline float	 sqr(float x)		    { return x*x; }
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

template <class T> __host__ inline vec<T, 3>
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
	       - T(2.0) * A.x.z*de;   // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

    const T p = m*m - T(3)*c1;
    const T q = m*(p - (T(3)/T(2))*c1) - (T(27)/T(2))*c0;
    const T sqrt_p = std::sqrt(std::abs(p));

    T phi = T(27) * (T(0.25)*sqr(c1)*(p - c1) + c0*(q + T(27)/T(4)*c0));
    phi = (T(1)/T(3)) * std::atan2(std::sqrt(std::abs(phi)), q);

    constexpr T	M_SQRT3 = 1.73205080756887729352744634151;	// sqrt(3)
    const T	c = sqrt_p*std::cos(phi);
    const T	s = (T(1)/M_SQRT3)*sqrt_p*std::sin(phi);

    vec<T, 3>	w;
    w.y  = (T(1)/T(3))*(m - c);
    w.z  = w.y + s;
    w.x  = w.y + c;
    w.y -= s;

    return w;
}

template <class T> __host__ __device__ void
tridiagonal33(const mat3x<T, 3>& A,
	      mat3x<T, 3>& Qt, vec<T, 3>& d, vec<T, 2>& e)
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
	T	K  = 0;

	omega = T(1) / omega;

	f    = A.y.y * uy + A.y.z * uz;
	T qy = omega * f;			// p
	K   += uy * f;				// u* A u
	f    = A.y.z * uy + A.z.z * uz;
	T qz = omega * f;			// p
	K   += uz * f;				// u* A u

	K *= T(0.5) * sqr(omega);

	qy -= K * uy;
	qz -= K * uz;

	d.x = A.x.x;
	d.y = A.y.y - 2.0*qy*uy;
	d.z = A.z.z - 2.0*qz*uz;

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
/*
template <class T> bool
qr33(const mat3x<T, 3>& A, mat3x<T, 3>& Q, vec<T, 3>& w)
{
  // Transform A to real tridiagonal form by the Householder method
    vec<T, 2>	e;
    tridiagonal33(A, Q, w, e);

  // Calculate eigensystem of the remaining real symmetric tridiagonal matrix
  // with the QL method
  //
  // Loop over all off-diagonal elements
    for (int l = 0; l < 2; ++l)
    {
	int	nIter = 0;
	for (;;)
	{
	  // Check for convergence and exit iteration loop if off-diagonal
	  // element e(l) is zero
	    if (l == 0)
	    {
		g = abs(w.x) + abs(w.y);
		if (abs(e.x) + g == g)
		    break;
	    }
	    g = abs(w.y) + abs(w.z);
	    if (abs(e.y) + g == g && l == 1)
		break;

	    if (nIter++ >= 30)
		return false;

	  // Calculate g = d_m - k
	    if (l == 0)
	    {
		g = (w.y - w.x) / (e.x + e.x);
		r = std::sqrt(SQR(g) + 1.0);
		if (g > 0)
		    g = w[m] - w[l] + e[l]/(g + r);
		else
		    g = w[m] - w[l] + e[l]/(g - r);
	    }

	    s = c = 1.0;
	    p = 0.0;
	    for (int i=m-1; i >= l; i--)
	    {
		f = s * e[i];
		b = c * e[i];
		if (std::abs(f) > std::abs(g))
		{
		    c      = g / f;
		    r      = std::sqrt(SQR(c) + 1.0);
		    e[i+1] = f * r;
		    c     *= (s = 1.0/r);
		}
		else
		{
		    s      = f / g;
		    r      = std::sqrt(SQR(s) + 1.0);
		    e[i+1] = g * r;
		    s     *= (c = 1.0/r);
		}
		
		g = w[i+1] - p;
		r = (w[i] - g)*s + 2.0*c*b;
		p = s * r;
		w[i+1] = g + p;
		g = c*r - b;

	      // Form eigenvectors
		for (int k=0; k < n; k++)
		{
		    t = Q[k][i+1];
		    Q[k][i+1] = s*Q[k][i] + c*t;
		    Q[k][i]   = c*Q[k][i] - s*t;
		}
	    }
	    w[l] -= p;
	    e[l]  = g;
	    e[m]  = 0.0;
	}
    }

    return 0;
}
*/
template <class T> __host__  inline bool
eigen33(const mat3x<T, 3>& A, mat3x<T, 3>& Qt, vec<T, 3>& w)
{
    w = cardano(A);		// Calculate eigenvalues

    T	t = std::abs(w.x), u = std::abs(w.y);
    if (u > t)
	t = u;
    if ((u = std::abs(w.z)) > t)
	t = u;
    if (t < T(1.0))
	u = t;
    else
	u = sqr(t);
    const T	error = T(256) * epsilon<T> * sqr(u);

    // auto	ax = A.x;
    // auto	ay = A.y;
    // ax.x -= w.x;
    // ay.y -= w.x;
    // Qt.x  = cross(ax, ay);
    
    // Qt.y.x = A.x.y*A.y.z - A.x.z*A.y.y;
    // Qt.y.y = A.x.z*A.x.y - A.x.x*A.y.z;
    // Qt.y.z = sqr(A.x.y);

  // Calculate first eigenvector by the formula
  //   v.x = (A - w.x).e1 x (A - w.x).e2
    // Qt.x.x = Qt.y.x + A.x.z*w.x;
    // Qt.x.y = Qt.y.y + A.y.z*w.x;
    // Qt.x.z = (A.x.x - w.x) * (A.y.y - w.x) - Qt.y.z;
    T	norm = dot(Qt.x, Qt.x);

  // If vectors are nearly linearly dependent, or if there might have
  // been large cancellations in the calculation of A[i][i] - w.x, fall
  // back to QL algorithm
  // Note that this simultaneously ensures that multiple eigenvalues do
  // not cause problems: If w.x = w.y, then A - w.x * I has rank 1,
  // i.e. all columns of A - w.x * I are linearly dependent.
    // if (norm <= error)
    // 	return qr33(A, Qt, w);
    // else                      // This is the standard branch
    {
	Qt.x *= 1.0/std::sqrt(norm);
    }

  // Calculate second eigenvector by the formula
  //   v.y = (A - w.y).e1 x (A - w.y).e2
    Qt.y.x = Qt.y.x + A.x.z*w.y;
    Qt.y.y = Qt.y.y + A.y.z*w.y;
    Qt.y.z = (A.x.x - w.y) * (A.y.y - w.y) - Qt.y.z;
    norm   = dot(Qt.y, Qt.y);
    // if (norm <= error)
    // 	return qr33(A, Qt, w);
    // else
    {
	Qt.y *= 1.0/std::sqrt(norm);
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
test_eigen33(const mat3x<T, 3>* A, mat3x<T, 3>* Qt, vec<T, 3>* w)
{
    device::eigen33(*A, *Qt, *w);
}

template <class T> __global__ void
test_tridiagonal33(const mat3x<T, 3>* A,
		   mat3x<T, 3>* Qt, vec<T, 3>* d, vec<T, 2>* e)
{
    device::tridiagonal33(*A, *Qt, *d, *e);
}

}	// namespace device
#endif

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

template <class T> vec<T, 3>
tridiagonal33(const mat3x<T, 3>& A,
	      mat3x<T, 3>& Qt, vec<T, 3>& d, vec<T, 2>& e)
{
    using mat_t	 = mat3x<T, 3>;
    using vec_t	 = vec<T, 3>;
    using vec2_t = vec<T, 2>;

    thrust::device_vector<mat_t>	A_d{1, A};
    thrust::device_vector<mat_t>	Qt_d(1);
    thrust::device_vector<vec_t>	d_d(1);
    thrust::device_vector<vec2_t>	e_d(1);

    device::test_tridiagonal33<<<1, 1>>>(get(&A_d[0]), get(&Qt_d[0]),
					 get(&d_d[0]), get(&e_d[0]));

    thrust::copy(Qt_d.begin(), Qt_d.end(), &Qt);
    d = d_d[0];
    e = e_d[0];
}

}	// namespace cuda
}	// namespace TU
