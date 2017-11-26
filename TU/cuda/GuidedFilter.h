/*!
  \file		GuidedFilter2.h
  \author	Toshio UESHIBA
  \brief	guided filterに関するクラスの定義と実装
*/
#ifndef TU_CUDA_GUIDEDFILTER2_H
#define TU_CUDA_GUIDEDFILTER2_H

#include "TU/cuda/tuple.h"
#include "TU/cuda/vec.h"
#include "TU/cuda/BoxFilter.h"

namespace TU
{
namespace cuda
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
*  class GuidedFilter2<T, CLOCK, WMAX>					*
************************************************************************/
//! 2次元guided filterを表すクラス
//! CUDAによる2次元boxフィルタを表すクラス
template <class T=float, class CLOCK=void, size_t WMAX=23>
class GuidedFilter2 : public Profiler<CLOCK>
{
  public:
    using element_type	= T;
    
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
    size_t	rowWinSize()	const	{ return _paramsFilter.rowWinSize(); }

  //! guidedフィルタのウィンドウ列幅(幅)を返す．
  /*!
    \return	guidedフィルタのウィンドウの列幅
   */
    size_t	colWinSize()	const	{ return _paramsFilter.colWinSize(); }

  //! guidedフィルタのウィンドウの行幅(高さ)を設定する．
  /*!
    \param rowWinSize	guidedフィルタのウィンドウの行幅
    \return		このguidedフィルタ
   */
    GuidedFilter2&
		setRowWinSize(size_t rowWinSize)
		{
		    _paramsFilter.setRowWinSize(rowWinSize);
		    _coeffsFilter.setRowWinSize(rowWinSize);
		    return *this;
		}

  //! guidedフィルタのウィンドウの列幅(幅)を設定する．
  /*!
    \param colWinSize	guidedフィルタのウィンドウの列幅
    \return		このguidedフィルタ
   */
    GuidedFilter2&
		setColWinSize(size_t colWinSize)
		{
		    _paramsFilter.setColWinSize(colWinSize);
		    _coeffsFilter.setColWinSize(colWinSize);
		    return *this;
		}
    
    auto	epsilon()		const	{ return _e; }
    auto&	setEpsilon(T e)			{ _e = e; return *this; }
    
    template <class IN, class GUIDE, class OUT>
    void	convolve(IN ib, IN ie,
			 GUIDE gb, GUIDE ge, OUT out)		const	;
    template <class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out)			const	;
    
  private:
    BoxFilter2<params_t, void, WMAX>	_paramsFilter;
    BoxFilter2<coeffs_t, void, WMAX>	_coeffsFilter;
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
template <class T, class CLOCK, size_t WMAX>
template <class IN, class GUIDE, class OUT> void
GuidedFilter2<T, CLOCK, WMAX>::convolve(IN ib, IN ie,
					GUIDE gb, GUIDE ge, OUT out) const
{
    if (ib == ie)
	return;

    profiler_t::start(0);

    const auto	n     = rowWinSize() * colWinSize();
    const auto	nrows = std::distance(ib, ie);
    const auto	ncols = TU::size(*ib);

    _c.resize(nrows + 1 - rowWinSize(), ncols + 1 - colWinSize());
    
  // guided filterの2次元係数ベクトルを計算する．
    profiler_t::start(1);
    _paramsFilter.convolve(make_range_iterator(
			       make_map_iterator(device::init_params<T>(),
						 std::cbegin(*ib),
						 std::cbegin(*gb)),
			     //thrust::make_tuple(stride(ib), stride(gb)),
			       stride(ib),
			       TU::size(*ib)),
			   make_range_iterator(
			       make_map_iterator(device::init_params<T>(),
						 std::cbegin(*ie),
						 std::cbegin(*ge)),
			     //thrust::make_tuple(stride(ie), stride(ge)),
			       stride(ie),
			       TU::size(*ie)),
			   make_range_iterator(
			       make_assignment_iterator(
				   _c.begin()->begin(),
				   device::init_coeffs<T>(n, _e)),
			       stride(_c.begin()), _c.ncol()));

  // 係数ベクトルの平均値を求め，それによってガイドデータ列を線型変換する．
    profiler_t::start(2);
    gb  += (rowWinSize() - 1);
    out += (rowWinSize() - 1);
    _coeffsFilter.convolve(_c.cbegin(), _c.cend(),
			   make_range_iterator(
			       cuda::make_assignment_iterator(
				   begin(thrust::make_tuple(*gb, *out))
				   += (colWinSize() - 1),
				   device::trans_guides<T>(n)),
			     //thrust::make_tuple(stride(gb), stride(out)),
			       stride(gb),
			       TU::size(*out)));
    profiler_t::nextFrame();
}

//! 2次元入力データにguided filterを適用する
/*!
  ガイドデータは与えられた2次元入力データに同一とする．
  \param ib	2次元入力データの先頭の行を示す反復子
  \param ie	2次元入力データの末尾の次の行を示す反復子
  \param out	guided filterを適用したデータの出力先の先頭行を示す反復子
*/
template <class T, class CLOCK, size_t WMAX>
template <class IN, class OUT> void
GuidedFilter2<T, CLOCK, WMAX>::convolve(IN ib, IN ie, OUT out) const
{
    if (ib == ie)
	return;

    profiler_t::start(0);
    const auto	n     = rowWinSize() * colWinSize();
    const auto	nrows = std::distance(ib, ie);
    const auto	ncols = TU::size(*ib);

    _c.resize(nrows + 1 - rowWinSize(), ncols + 1 - colWinSize());
    
  // guided filterの2次元係数ベクトルを計算する．
    profiler_t::start(1);
    _paramsFilter.convolve(make_range_iterator(
			       make_map_iterator(device::init_params<T>(),
						 std::cbegin(*ib)),
			       stride(ib), TU::size(*ib)),
			   make_range_iterator(
			       make_map_iterator(device::init_params<T>(),
						 std::cbegin(*ie)),
			       stride(ie), TU::size(*ie)),
			   make_range_iterator(
			       make_assignment_iterator(
				   _c.begin()->begin(),
				   device::init_coeffs<T>(n, _e)),
			       stride(_c.begin()), _c.ncol()));

  // 係数ベクトルの平均値を求め，それによってガイドデータ列を線型変換する．
    profiler_t::start(2);
    ib  += (rowWinSize() - 1);
    out += (rowWinSize() - 1);
    _coeffsFilter.convolve(_c.cbegin(), _c.cend(),
			   make_range_iterator(
			       cuda::make_assignment_iterator(
				   begin(thrust::make_tuple(*ib, *out))
				   += (colWinSize() - 1),
				   device::trans_guides<T>(n)),
			     //thrust::make_tuple(stride(gb), stride(out)),
			       stride(ib),
			       TU::size(*out)));
    profiler_t::nextFrame();
}

}	// namespace cuda
}	// namespace TU
#endif	// !TU_CUDA_GUIDEDFILTER2_H
