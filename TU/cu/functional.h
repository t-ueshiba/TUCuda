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

#include <thrust/functional.h>

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

}	// namespace cu
}	// namespace TU
