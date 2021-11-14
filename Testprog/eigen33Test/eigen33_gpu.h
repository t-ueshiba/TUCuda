#pragma once

#include "TU/cuda/functional.h"
#include <thrust/device_vector.h>

namespace TU
{
namespace cuda
{
#if defined(__NVCC__)
namespace device
{
  /**********************************************************************
  *  __global__ functions						*
  **********************************************************************/
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
