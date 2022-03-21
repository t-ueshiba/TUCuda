// Software License Agreement (BSD License)
//
// Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of National Institute of Advanced Industrial
//    Science and Technology (AIST) nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Toshio Ueshiba
//
/*!
  \file		functional.h
  \brief	CUDAデバイス上で実行される関数オブジェクトの定義と実装
*/
#pragma once

#include <cmath>
#include <thrust/functional.h>
#include "TU/cu/vec.h"

namespace TU
{
namespace cu
{
/************************************************************************
*  1x3, 3x1 and 3x3  operators						*
************************************************************************/
template <size_t OPERATOR_SIZE_Y, size_t OPERATOR_SIZE_X>
struct OperatorTraits
{
    constexpr static size_t	OperatorSizeY = OPERATOR_SIZE_Y;
    constexpr static size_t	OperatorSizeX = OPERATOR_SIZE_X;
};
    
//! 横方向1階微分オペレータを表す関数オブジェクト
template <class T>
struct diffH1x3 : public OperatorTraits<1, 3>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	return T(0.5)*(in[y][x+1] - in[y][x-1]);
    }

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < x && x + 1 < ncol ? (*this)(y, x, in) : T(0));
    }
};

//! 縦方向1階微分オペレータを表す関数オブジェクト
template <class T>
struct diffV3x1 : public OperatorTraits<3, 1>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	return T(0.5)*(in[y+1][x] - in[y-1][x]);
    }

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < y && y + 1 < nrow ? (*this)(y, x, in) : T(0));
    }
};

//! 横方向2階微分オペレータを表す関数オブジェクト
template <class T>
struct diffHH1x3 : public OperatorTraits<1, 3>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	return in[y][x-1] - 2*in[y][x] + in[y][x+1];
    }

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < x && x + 1 < ncol ? (*this)(y, x, in) : T(0));
    }
};

//! 縦方向2階微分オペレータを表す関数オブジェクト
template <class T>
struct diffVV3x1 : public OperatorTraits<3, 1>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	return in[y-1][x] - 2*in[y][x] + in[y+1][x];
    }

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < y && y + 1 < nrow ? (*this)(y, x, in) : T(0));
    }
};

//! 縦横両方向2階微分オペレータを表す関数オブジェクト
template <class T>
struct diffHV3x3 : public OperatorTraits<3, 3>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	return T(0.25)*(in[y-1][x-1] - in[y-1][x+1] -
			in[y+1][x-1] + in[y+1][x+1]);
    }

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < y && y + 1 < nrow && 0 < x && x + 1 < ncol ?
		(*this)(y, x, in) : T(0));
    }
};

//! 横方向1階微分Sobelオペレータを表す関数オブジェクト
template <class T>
struct sobelH3x3 : public OperatorTraits<3, 3>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	return T(0.125)*(in[y-1][x+1] - in[y-1][x-1] +
			 in[y+1][x+1] - in[y+1][x-1]) +
	       T(0.250)*(in[y  ][x+1] - in[y  ][x-1]);
	
    }
    
    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < y && y + 1 < nrow && 0 < x && x + 1 < ncol ?
		(*this)(y, x, in) : T(0));
    }
};

//! 縦方向1階微分Sobelオペレータを表す関数オブジェクト
template <class T>
struct sobelV3x3 : public OperatorTraits<3, 3>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	return T(0.125)*(in[y+1][x-1] - in[y-1][x-1] +
			 in[y+1][x+1] - in[y-1][x+1]) +
	       T(0.250)*(in[y+1][x  ] - in[y-1][x  ]);
    }
    
    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < y && y + 1 < nrow && 0 < x && x + 1 < ncol ?
		(*this)(y, x, in) : T(0));
    }
};

//! 1階微分Sobelオペレータの縦横両方向出力の絶対値の和を表す関数オブジェクト
template <class T>
struct sobelAbs3x3 : public OperatorTraits<3, 3>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	return abs(sobelH3x3<T>()(y, x, in)) + abs(sobelV3x3<T>()(y, x, in));
    }

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < y && y + 1 < nrow && 0 < x && x + 1 < ncol ?
		(*this)(y, x, in) : T(0));
    }
};

//! ラプラシアンオペレータを表す関数オブジェクト
template <class T>
struct laplacian3x3 : public OperatorTraits<3, 3>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	return in[y][x-1] + in[y][x+1] +
	       in[y-1][x] + in[y+1][x] - T(4)*in[y][x];
    }

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < y && y + 1 < nrow && 0 < x && x + 1 < ncol ?
		(*this)(y, x, in) : T(0));
    }
};

//! ヘッセ行列式オペレータを表す関数オブジェクト
template <class T>
struct det3x3 : public OperatorTraits<3, 3>
{
    using result_type = T;

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, T_ in[][W_]) const
    {
	const auto	dxy = diffHV3x3<T>()(y, x, in);

	return diffHH1x3<T>()(y, x, in) * diffVV3x1<T>()(y, x, in) - dxy * dxy;
    }

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (0 < y && y + 1 < nrow && 0 < x && x + 1 < ncol ?
		(*this)(y, x, in) : T(0));
    }
};

//! Logical AND operator between a pixel and its 4-neighbors
template <class COMP>
class logical_and4 : public OperatorTraits<3, 3>
{
  public:
    using result_type = bool;

    __host__ __device__
    logical_and4(COMP comp=COMP())	:_comp(comp)		{}

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (y   <= 0    || _comp(in[y][x], in[y-1][x])) &&
	       (y+1 >= nrow || _comp(in[y][x], in[y+1][x])) &&
	       (x   <= 0    || _comp(in[y][x], in[y][x-1])) &&
	       (x   >= ncol || _comp(in[y][x], in[y][x+1]));
    }

  private:
    const COMP	_comp;
};

//! Logical OR operator between a pixel and its 4-neighbors
template <class COMP>
class logical_or4 : public OperatorTraits<3, 3>
{
  public:
    using result_type = bool;

    __host__ __device__
    logical_or4(COMP comp=COMP())	:_comp(comp)		{}

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return (y   > 0    && _comp(in[y][x], in[y-1][x])) ||
	       (y+1 < nrow && _comp(in[y][x], in[y+1][x])) ||
	       (x   > 0    && _comp(in[y][x], in[y][x-1])) ||
	       (x   < ncol && _comp(in[y][x], in[y][x+1]));
    }

  private:
    const COMP	_comp;
};

//! Logical AND operator between a pixel and its 4-neighbors
template <class COMP>
class logical_and8 : public OperatorTraits<3, 3>
{
  public:
    using result_type = bool;

    __host__ __device__
    logical_and8(COMP comp=COMP())	:_comp(comp)		{}

    template <class T, size_t W> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T in[][W]) const
    {
	return logical_and4<COMP>(_comp)(y, x, nrow, ncol, in)  &&
	       (y <= 0       ||
		(x   <= 0    || _comp(in[y][x], in[y-1][x-1]))  &&
		(x+1 >= ncol || _comp(in[y][x], in[y-1][x+1]))) &&
	       (y+1 >= nrow  ||
		(x   <= 0    || _comp(in[y][x], in[y+1][x-1]))  &&
		(x+1 >= ncol || _comp(in[y][x], in[y+1][x+1])));
    }

  private:
    const COMP	_comp;
};

//! Logical OR operator between a pixel and its 4-neighbors
template <class COMP>
class logical_or8 : public OperatorTraits<3, 3>
{
  public:
    using result_type = bool;

    __host__ __device__
    logical_or8(COMP comp=COMP())	:_comp(comp)		{}

    template <class T_, size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T_ in[][W_]) const
    {
	return logical_or4<COMP>(_comp)(y, x, nrow, ncol, in)  ||
	       (y > 0       &&
		(x   > 0    && _comp(in[y][x], in[y-1][x-1]))  ||
		(x+1 < ncol && _comp(in[y][x], in[y-1][x+1]))) ||
	       (y+1 < nrow  &&
		(x   > 0    && _comp(in[y][x], in[y+1][x-1]))  ||
		(x+1 < ncol && _comp(in[y][x], in[y+1][x+1])));
    }

  private:
    const COMP	_comp;
};

//!
template <class T, class LOGICAL_AND>
class extremal3x3 : public OperatorTraits<3, 3>
{
  public:
    using result_type = T;

    __host__ __device__
    extremal3x3(T false_value=0) 
	:_false_value(false_value), _logical_and()		{}
    
    template <size_t W_> __host__ __device__ result_type
    operator ()(int y, int x, int nrow, int ncol, T in[][W_]) const
    {
	return (_logical_and(y, x, nrow, ncol, in) ? in[y][x] : _false_value);
    }
    
  private:
    const T		_false_value;
    const LOGICAL_AND	_logical_and;
};

template <class T>
using maximal4	 = extremal3x3<T, logical_and4<thrust::greater<T> > >;

template <class T>
using minimal4	 = extremal3x3<T, logical_and4<thrust::less<T> > >;

template <class T>
using maximal8	 = extremal3x3<T, logical_and8<thrust::greater<T> > >;

template <class T>
using minimal8	 = extremal3x3<T, logical_and8<thrust::less<T> > >;

template <class T>
using is_border4 = logical_or4<thrust::not_equal_to<T> >;

//! 2つの値の閾値付き差を表す関数オブジェクト
template <class T>
class diff
{
  public:
    using result_type = T;

    __host__ __device__
    diff(T thresh)	:_thresh(thresh)			{}

    __host__ __device__ result_type
    operator ()(T x, T y) const
    {
	return thrust::minimum<T>()((x > y ? x - y : y - x), _thresh);
    }

  private:
    const T	_thresh;
};

template <class T>
class overlay
{
  public:
    using result_type = void;

    __host__ __device__
    overlay(T val)	:_val(val)				{}
    
    template <class T_> __host__ __device__ void
    operator ()(T_&& out, bool draw) const
    {
	if (draw)
	    out = _val;
    }

  private:
    const T	_val;
};
    
/************************************************************************
*  undistortion operator						*
************************************************************************/
template <class T>
struct undistort
{
    template <class ITER_K, class ITER_D> __host__ __device__
    undistort(ITER_K K, ITER_D d)
    {
	_flen.x = *K;
	std::advance(K, 2);
	_uv0.x = *K;
	std::advance(K, 2);
	_flen.y = *K;
	++K;
	_uv0.y = *K;
	_d[0] = *d;
	_d[1] = *++d;
	_d[2] = *++d;
	_d[3] = *++d;
    }
    
    __host__ __device__ vec<T, 2>
    operator ()(T u, T v) const
    {
	constexpr static T	MAX_ERR  = 0.001*0.001;
	constexpr static int	MAX_ITER = 5;

	const vec<T, 2>	uv{u, v};
	auto		xy  = (uv - _uv0)/_flen;
	const auto	xy0 = xy;

      // compensate distortion iteratively
	for (int n = 0; n < MAX_ITER; ++n)
	{
	    const auto	r2 = cu::square(xy);
	    const auto	k  = T(1) + (_d[0] + _d[1]*r2)*r2;
	    if (k < T(0))
		break;

	    const auto	a = T(2)*xy.x*xy.y;
	    vec<T, 2>	delta{_d[2]*a + _d[3]*(r2 + T(2)*xy.x*xy.x),
			      _d[2]*(r2 + T(2)*xy.y*xy.y) + _d[3]*a};
	    const auto	uv_proj = _flen*(k*xy + delta) + _uv0;

	    if (cu::square(uv_proj - uv) < MAX_ERR)
		break;

	    xy = (xy0 - delta)/k;	// compensate lens distortion
	}

	return xy;
    }

    __host__ __device__ vec<T, 3>
    operator ()(T u, T v, T d) const
    {
	const auto	xy = (*this)(u, v);
	return {d*xy.x, d*xy.y, d};
    }

  private:
    vec<T, 2>	_flen;
    vec<T, 2>	_uv0;
    T		_d[4];
};

/************************************************************************
*  operators for fitting plane to 3D points 				*
************************************************************************/
#if defined(__NVCC__)
namespace device
{
  template <class T> __device__ __forceinline__ vec<T, 3>
  cardano(const mat3x<T, 3>& A)
  {
    // Determine coefficients of characteristic poynomial. We write
    //       | a   d   f  |
    //  A =  | d*  b   e  |
    //       | f*  e*  c  |
      const T de = A.x.y * A.y.z;	    // d * e
      const T dd = square(A.x.y);	    // d^2
      const T ee = square(A.y.z);	    // e^2
      const T ff = square(A.x.z);	    // f^2
      const T m  = A.x.x + A.y.y + A.z.z;
      const T c1 = (A.x.x*A.y.y + A.x.x*A.z.z + A.y.y*A.z.z)
	  - (dd + ee + ff);    // a*b + a*c + b*c - d^2 - e^2 - f^2
      const T c0 = A.z.z*dd + A.x.x*ee + A.y.y*ff - A.x.x*A.y.y*A.z.z
	  - T(2) * A.x.z*de;   // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

      const T p = m*m - T(3)*c1;
      const T q = m*(p - (T(3)/T(2))*c1) - (T(27)/T(2))*c0;
      const T sqrt_p = sqrt(abs(p));

      T phi = T(27) * (T(0.25)*square(c1)*(p - c1) + c0*(q + T(27)/T(4)*c0));
      phi = (T(1)/T(3)) * atan2(sqrt(abs(phi)), q);

      constexpr T	M_SQRT3 = 1.73205080756887729352744634151;  // sqrt(3)
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
  {
    // ----------------------------------------------------------------------
    // Reduces a symmetric 3x3 matrix to tridiagonal form by applying
    // (unitary) Householder transformations:
    //             [ d[0]  e[0]       ]
    //    Qt.A.Q = [ e[0]  d[1]  e[1] ]
    //             [       e[1]  d[2] ]
    // The function accesses only the diagonal and upper triangular parts of A.
    // The access is read-only.
    // ----------------------------------------------------------------------
      Qt = T(0);
      Qt.x.x = Qt.y.y = Qt.z.z = T(1);

    // Bring first row and column to the desired form
      const T	h = square(A.x.y) + square(A.x.z);
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
	  T K  = uy * f;			// u* A u
	  f    = A.y.z * uy + A.z.z * uz;
	  T qz = omega * f;			// p
	  K   += uz * f;			// u* A u

	  K  *= T(0.5) * square(omega);
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
    template <class T> __device__ __forceinline__ bool
    is_zero(T e, T g)
    {
	return abs(e) + g == g;
    }

    template <class T> __device__ T
    init_offdiagonal(T w, T w0, T w1, T e0)
    {
	const auto	t = (w1 - w0)/(e0 + e0);
	const auto	r = sqrt(square(t) + T(1));
	return w - w0 + e0/(t + (t > 0 ? r : -r));
    }

    template <size_t I, class T> __device__ __forceinline__ void
    diagonalize(vec<T, 3>& w, vec<T, 3>& e,
		vec<T, 3>& q0, vec<T, 3>& q1, T& c, T& s)
    {
	const auto	x = val<I+1>(e);
	const auto	y = s*val<I>(e);
	const auto	z = c*val<I>(e);
	if (abs(x) > abs(y))
	{
	    const auto	t = y/x;
	    const auto	r = sqrt(square(t) + T(1));
	    val<I+1>(e) = x*r;
	    c 	      = T(1)/r;
	    s	      = c*t;
	}
	else
	{
	    const auto	t = x/y;
	    const auto	r = sqrt(square(t) + T(1));
	    val<I+1>(e) = y*r;
	    s	      = T(1)/r;
	    c	      = s*t;
	}

	const auto	v = s*(val<I>(w) - val<I+1>(w)) + T(2)*c*z;
	const auto	p = s*v;
	val<I  >(w) -= p;
	val<I+1>(w) += p;
	val<I  >(e)  = c*v - z;

      // Form eigenvectors
	const auto	q0_old = q0;
	(q0 *= c) -= s*q1;
	(q1 *= c) += s*q0_old;
    }

    template <class T> __device__ __forceinline__ vec<T, 3>
    eigen_vector(vec<T, 3> a, vec<T, 3> b, T w)
    {
	a.x -= w;
	b.y -= w;
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
		      detail::diagonalize<1>(w, e, Qt.y, Qt.z, c, s);
		  else
		      detail::diagonalize<0>(w, e, Qt.x, Qt.y, c, s);
	  }
      }

      return true;
  }

  template <class T> __device__  __forceinline__ bool
  eigen33(const mat3x<T, 3>& A, mat3x<T, 3>& Qt, vec<T, 3>& w)
  {
      w = cardano(A);		// Calculate eigenvalues

      const auto	t     = min(min(abs(w.x), abs(w.y)), abs(w.z));
      const auto	u     = (t < T(1) ? t : square(t));
      const auto	error = T(256) * epsilon<T> * square(u);

    // 1st eigen vector
      Qt.x = detail::eigen_vector(A.x, A.y, w.x);
      auto	norm = dot(Qt.x, Qt.x);
      if (norm <= error)
	  return qr33(A, Qt, w);
      Qt.x *= rsqrt(norm);

    // 2nd eigen vector
      Qt.y = detail::eigen_vector(A.x, A.y, w.y);
      norm  = dot(Qt.y, Qt.y);
      if (norm <= error)
	  return qr33(A, Qt, w);
      Qt.y *= rsqrt(norm);

    // 3rd eigen vector
      Qt.z = cross(Qt.x, Qt.y);

      return true;
  }
}	// namespace device

template <class T, bool POINT_ARG=false>
struct plane_moment
{
  /*
   * [[  x,   y,   z],
   *  [x*x, x*y, x*z],
   *  [y*y, y*z, z*z],
   *  [  u,   v,   w]]		# w = (z > 0 ? 1 : 0)
   */
    using result_type = mat4x<T, 3>;

    template <bool POINT_ARG_=POINT_ARG>
    __host__ __device__ std::enable_if_t<!POINT_ARG_, result_type>
    operator ()(const vec<T, 3>& point) const
    {
	return (point.z > T(0) ?
	        result_type(
		    point,
		    point.x * point,
		    {point.y * point.y, point.y * point.z, point.z * point.z},
		    {0, 0, 1}) :
		result_type(0));
    }

    template <bool POINT_ARG_=POINT_ARG>
    __host__ __device__ std::enable_if_t<POINT_ARG_, result_type>
    operator ()(int v, int u, const vec<T, 3>& point) const
    {
	return (point.z > T(0) ?
	        result_type(
		    point,
		    point.x * point,
		    {point.y * point.y, point.y * point.z, point.z * point.z},
		    {u, v, 1}) :
		result_type(0));
    }
};

template <class T>
struct plane_estimator
{
  /*
   * [[cx, cy, cz],		# center of sampled points
   *  [nx, ny, nz],		# plane normal
   *  [u, v, mse]]		# mean of 2D sample points, mean-square error
   */
    using result_type = mat3x<T, 3>;

    __host__ __device__ static result_type
    invalid_plane()
    {
	return _invalid_plane;
    }
    
    __host__ __device__ result_type
    operator ()(const mat4x<T, 3>& moment) const
    {
	if (moment.w.z < T(4))	// Four or more points required.
	    return _invalid_plane;

	const auto	sc = T(1)/moment.w.z;
	result_type	plane;
	plane.x = moment.x * sc;

	mat3x<T, 3>	A = {moment.y - moment.x.x * plane.x,
			     {T(0),
			      moment.z.x - moment.x.y * plane.x.y,
			      moment.z.y - moment.x.y * plane.x.z},
			     {T(0),
			      T(0),
			      moment.z.z - moment.x.z * plane.x.z}};
	mat3x<T, 3>	evecs;
	vec<T, 3>	evals;
	device::eigen33(A, evecs, evals);

	T		eval_min;
	if (evals.x < evals.y)
	    if (evals.x < evals.z)
	    {
		eval_min = evals.x;
		plane.y  = evecs.x;
	    }
	    else
	    {
		eval_min = evals.z;
		plane.y  = evecs.z;
	    }
	else
	    if (evals.y < evals.z)
	    {
		eval_min = evals.y;
		plane.y  = evecs.y;
	    }
	    else
	    {
		eval_min = evals.z;
		plane.y  = evecs.z;
	    }

      // enforce dot(normal, center) < 0 so normal points towards camera
	if (dot(plane.x, plane.y) > T(0))
	    plane.y *= T(-1);
	
	plane.z = {moment.w.x*sc, moment.w.y*sc, eval_min*sc};

	return plane;
    }

  private:
    constexpr static result_type _invalid_plane{{0, 0, 0},
						{0, 0, 0},
						{0, 0, device::maxval<T>}};
};

template <class T=vec<uint8_t, 3> >
struct colored_normal
{
    using result_type	= T;

    __host__ __device__ result_type
    operator ()(const vec<float, 3>& normal) const
    {
	return {uint8_t(128 + 127*normal.x),
		uint8_t(128 + 127*normal.y),
		uint8_t(128 + 127*normal.z)};
    }

    template <class T_> __host__ __device__ result_type
    operator ()(const mat3x<T_, 3>& plane) const
    {
	return (*this)(plane.y);
    }
};
#endif
}	// namespace cu
}	// namespace TU
