/*
 *  $Id$
 */
/*!
  \file		Normal.h
  \brief	depth normalフィルタの定義と実装
*/
#pragma once

#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"
#include "TU/cuda/vec.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  class Undistort<T>							*
************************************************************************/
template <class T=float>
class Undistort
{
  private:

  public:
		Undistort()						;

    Undistort&	initialize(const Affinity<T, 3, 3>& K,
			   const vec<T, 4>& d)				;
    vec<T, 3>	operator ()(int u, int v)			const	;

  private:
    Affinity<T, 3, 3>	_K;

    vec<T, 4>		_d;
    vec<T, 2>		_finv;

    mutable mat<T, 3, 3>	_K;
};

#if defined(__NVCC__)
namespace device
{
  __device__ void
  undistort(const vec<T, 2>& u, const mat3x<T, 3>& K, const vec<T, 4>& d,
	    const Projectivity<T, 3, 3>& R, int max_iter=5)
  {
      const vec<T, 2>	f{K.x.x, K.y.y};
      const vec<T, 2>	finv = T(1)/f;
      const vec<T, 2>	u0{K.x.z, K.y.z};

      auto		x  = (u - u0)*finv;
      const auto	x0 = x;
      auto		error = std::numeric_limits<T>::max();

    // compensate distortion iteratively
      for (int n = 0; n < max_iter; ++n)
      {
	  const auto	r2     = square(x);
	  const auto	icdist = T(1) / (T(1) + (d.x + d.y*r2)*r2);
	  if (icdist < 0)  // test: undistortPoints.regression_14583
	      break;

	  const auto		a = T(2)*x.x*x.y;
	  const vec<T, 2>	delta{d.z*a + d.w*(r2 + T(2)*x.x*x.x),
				      d.z*(r2 + T(2)*x.y*x.y) + d.w*a};
	  x = (x0 - delta)*icdist;	// compensate lens distortion

	  if (max_iter == 0)
	  {
	      r2 = square(x);
	      a  = T(2)*x.x*x.y;
	      const auto	cdist = T(1) + (d.x*r2 + d.y*r2)*r2;
	      const vec<T, 2>	xd{x*cdist + d.z*a + d.w*(r2 + T(2)*x.x*x.x),
				   y*cdist + d.z*(r2 + T(2)*x.y*x.y) + d.w*a};

	      const auto	u_proj = f*xd + u0;

	      error = square(u_proj - u);
	  }
      }

      return R(x);
  }
}	// namespace device
#endif


}	// namespace cuda
}	// namespace TU
