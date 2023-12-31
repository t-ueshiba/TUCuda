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
  \file		GuidedFilter.h
  \author	Toshio UESHIBA
  \brief	guided filterに関するクラスの定義と実装
*/
#pragma once

#include "TU/cu/tuple.h"
#include "TU/cu/vec.h"
#include "TU/cu/BoxFilter.h"
#include "TU/cu/iterator.h"

namespace TU
{
template <class T>	class TD;

namespace cu
{
namespace device
{
/************************************************************************
*  utility functions							*
************************************************************************/
template <class T>
struct init_params
{
    template <class IN_, class GUIDE_> __host__ __device__
    vec<T, 4>	operator ()(IN_ p, GUIDE_ g) const
		{
		    return {T(p), T(g), T(p)*T(g), T(g)*T(g)};
		}
    template <class IN_> __host__ __device__
    vec<T, 2>	operator ()(IN_ p) const
		{
		    return {T(p), T(p)*T(p)};
		}
};

template <class T>
struct init_coeffs
{
    __host__ __device__
    init_coeffs(size_t n, T e)	:_n(n), _sq_e(e*e)			{}

    __host__ __device__
    vec<T, 2>	operator ()(const vec<T, 4>& params) const
		{
		    vec<T, 2>	coeffs;
		    coeffs.x = (_n*params.z - params.x*params.y)
			     / (_n*(params.w + _n*_sq_e) - params.y*params.y);
		    coeffs.y = (params.x - coeffs.x*params.y)/_n;

		    return coeffs;
		}
    __host__ __device__
    vec<T, 2>	operator ()(const vec<T, 2>& params) const
		{
		    vec<T, 2>	coeffs;
		    const auto	var = _n*params.y - params.x*params.x;

		    coeffs.x = var/(var + _n*_n*_sq_e);
		    coeffs.y = (params.x - coeffs.x*params.y)/_n;

		    return coeffs;
		}

  private:
    size_t	_n;
    T		_sq_e;
};

template <class T>
class trans_guides
{
  public:
    __host__ __device__
    trans_guides(size_t n)	:_n(n)					{}

    template <class GUIDE_, class OUT_> __host__ __device__
    void	operator ()(thrust::tuple<GUIDE_, OUT_>&& t,
			    const vec<T, 2>& coeffs) const
		{
		    using	thrust::get;

		    get<1>(t) = (coeffs.x*get<0>(t) + coeffs.y)/_n;
		}

  private:
    size_t	_n;
};

}	// namespace device

/************************************************************************
*  class GuidedFilter2<T, BLOCK_TRAITS, WMAX, CLOCK>			*
************************************************************************/
//! 2次元guided filterを表すクラス
//! CUDAによる2次元boxフィルタを表すクラス
template <class T=float,
	  class BLOCK_TRAITS=BlockTraits<>, size_t WMAX=23, class CLOCK=void>
class GuidedFilter2 : public BLOCK_TRAITS, public Profiler<CLOCK>
{
  public:
    using element_type	= T;

    using			BLOCK_TRAITS::BlockDimX;
    using			BLOCK_TRAITS::BlockDimY;
    constexpr static size_t	WinSizeMax = WMAX;

  private:
    using params_t	= vec<T, 4>;
    using coeffs_t	= vec<T, 2>;
    using profiler_t	= Profiler<CLOCK>;

  public:
    GuidedFilter2(size_t wrow, size_t wcol, T e)
	:profiler_t(3),
	 _paramsFilter(wrow, wcol), _coeffsFilter(wrow, wcol), _e(e)	{}

  //! guidedフィルタのウィンドウ行幅(高さ)を返す．
  /*!
    \return	guidedフィルタのウィンドウの行幅
   */
    size_t	winSizeV()	const	{ return _paramsFilter.winSizeV(); }

  //! guidedフィルタのウィンドウ列幅(幅)を返す．
  /*!
    \return	guidedフィルタのウィンドウの列幅
   */
    size_t	winSizeH()	const	{ return _paramsFilter.winSizeH(); }

  //! guidedフィルタのウィンドウの行幅(高さ)を設定する．
  /*!
    \param winSizeV	guidedフィルタのウィンドウの行幅
    \return		このguidedフィルタ
   */
    GuidedFilter2&
		setWinSizeV(size_t winSizeV)
		{
		    _paramsFilter.setWinSizeV(winSizeV);
		    _coeffsFilter.setWinSizeV(winSizeV);
		    return *this;
		}

  //! guidedフィルタのウィンドウの列幅(幅)を設定する．
  /*!
    \param winSizeH	guidedフィルタのウィンドウの列幅
    \return		このguidedフィルタ
   */
    GuidedFilter2&
		setWinSizeH(size_t winSizeH)
		{
		    _paramsFilter.setWinSizeH(winSizeH);
		    _coeffsFilter.setWinSizeH(winSizeH);
		    return *this;
		}

    auto	epsilon()		const	{ return _e; }
    auto&	setEpsilon(T e)			{ _e = e; return *this; }

    template <class IN, class GUIDE, class OUT>
    void	convolve(IN ib, IN ie, GUIDE gb, GUIDE ge,
			 OUT out, bool shift=true)		  const	;
    template <class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out, bool shift=true) const	;

    size_t	outSizeH(size_t inSize)	const	{return inSize - 2*offsetH();}
    size_t	outSizeV(size_t inSize)	const	{return inSize - 2*offsetV();}
    size_t	offsetH()		const	{return winSizeH() - 1;}
    size_t	offsetV()		const	{return winSizeV() - 1;}

  private:
    BoxFilter2<device::box_convolver<params_t>,
	       BLOCK_TRAITS, WMAX>	_paramsFilter;
    BoxFilter2<device::box_convolver<coeffs_t>,
	       BLOCK_TRAITS, WMAX>	_coeffsFilter;
    T					_e;
    mutable Array2<coeffs_t>		_c;
};

//! 2次元入力データと2次元ガイドデータにguided filterを適用する
/*!
  \param ib	2次元入力データの先頭の行を示す反復子
  \param ie	2次元入力データの末尾の次の行を示す反復子
  \param gb	2次元ガイドデータの先頭の行を示す反復子
  \param ge	2次元ガイドデータの末尾の次の行を示す反復子
  \param out	guided filterを適用したデータの出力先の先頭行を示す反復子
*/
template <class T, class BLOCK_TRAITS, size_t WMAX, class CLOCK>
template <class IN, class GUIDE, class OUT> void
GuidedFilter2<T, BLOCK_TRAITS, WMAX, CLOCK>::convolve(IN ib, IN ie,
						      GUIDE gb, GUIDE ge,
						      OUT out, bool shift) const
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;
    using	std::size;

    if (ib == ie)
	return;

    profiler_t::start(0);

    const auto	n     = winSizeV() * winSizeH();
    const auto	nrows = std::distance(ib, ie);
    const auto	ncols = size(*ib);

    _c.resize(nrows + 1 - winSizeV(), ncols + 1 - winSizeH());

  // guided filterの2次元係数ベクトルを計算する．
    profiler_t::start(1);
    _paramsFilter.convolve(make_range_iterator(
			       make_map_iterator(device::init_params<T>(),
						 cbegin(*ib),
						 cbegin(*gb)),
			       cu::stride(ib, gb),
			       size(*ib)),
			   make_range_iterator(
			       make_map_iterator(device::init_params<T>(),
						 cbegin(*ie),
						 cbegin(*ge)),
			       cu::stride(ie, ge),
			       size(*ie)),
			   make_range_iterator(
			       make_assignment_iterator(
				   device::init_coeffs<T>(n, _e),
				   _c.begin()->begin()),
			       stride(_c.begin()),
			       _c.ncol()),
			   false);

  // 係数ベクトルの平均値を求め，それによってガイドデータ列を線型変換する．
    profiler_t::start(2);
    gb += offsetV();
    if (shift)
	out += offsetV();
    _coeffsFilter.convolve(_c.cbegin(), _c.cend(),
			   make_range_iterator(
			       make_assignment_iterator(
				   device::trans_guides<T>(n),
				   begin(*gb)  + offsetH(),
				   begin(*out) + (shift ? offsetH() : 0)),
			       cu::stride(gb, out),
			       size(*out)),
			   false);
    profiler_t::nextFrame();
}

//! 2次元入力データにguided filterを適用する
/*!
  ガイドデータは与えられた2次元入力データに同一とする．
  \param ib	2次元入力データの先頭の行を示す反復子
  \param ie	2次元入力データの末尾の次の行を示す反復子
  \param out	guided filterを適用したデータの出力先の先頭行を示す反復子
*/
template <class T, class BLOCK_TRAITS, size_t WMAX, class CLOCK>
template <class IN, class OUT> void
GuidedFilter2<T, BLOCK_TRAITS, WMAX, CLOCK>::convolve(IN ib, IN ie,
						      OUT out, bool shift) const
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;
    using	std::size;

    if (ib == ie)
	return;

    profiler_t::start(0);
    const auto	n     = winSizeV() * winSizeH();
    const auto	nrows = std::distance(ib, ie);
    const auto	ncols = size(*ib);

    _c.resize(nrows + 1 - winSizeV(), ncols + 1 - winSizeH());

  // guided filterの2次元係数ベクトルを計算する．
    profiler_t::start(1);
    _paramsFilter.convolve(make_range_iterator(
			       make_map_iterator(device::init_params<T>(),
						 cbegin(*ib)),
			       stride(ib), size(*ib)),
			   make_range_iterator(
			       make_map_iterator(device::init_params<T>(),
						 cbegin(*ie)),
			       stride(ie), size(*ie)),
			   make_range_iterator(
			       make_assignment_iterator(
				   device::init_coeffs<T>(n, _e),
				   _c.begin()->begin()),
			       stride(_c.begin()), _c.ncol()),
			   false);

  // 係数ベクトルの平均値を求め，それによってガイドデータ列を線型変換する．
    profiler_t::start(2);
    ib += offsetV();
    if (shift)
	out += offsetV();
    _coeffsFilter.convolve(_c.cbegin(), _c.cend(),
			   make_range_iterator(
			       make_assignment_iterator(
				   device::trans_guides<T>(n),
				   begin(*ib)  + offsetH(),
				   begin(*out) + (shift ? offsetH() : 0)),
			       cu::stride(ib, out),
			       size(*out)),
			   false);
    profiler_t::nextFrame();
}

}	// namespace cu
}	// namespace TU
