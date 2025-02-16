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
  \file		ICIA.h
  \author	Toshio UESHIBA
  \brief	クラス TU::cu::ICIA の定義と実装
*/
#pragma once

#include <Eigen/Eigen>
#include "TU/Profiler.h"
#include "TU/cu/FIRGaussianConvolver.h"
#include "TU/cu/BoxFilter.h"
#include "TU/cu/chrono.h"
#include "TU/cu/vec.h"
#include "TU/cu/Texture.h"
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <iomanip>
#include "TU/Image++.h"

namespace TU
{
namespace cu
{
namespace detail
{
  template <class MAP, class C>
  class ICIAColorMoment
  {
    private:
      using image_type	= range<range_iterator<thrust::device_ptr<const C> > >;
      constexpr static size_t	DOF = MAP::DOF;

    public:
      using value_type	= typename MAP::element_type;
      using array_type	= array<value_type, DOF*(DOF+1)/2>;
      using matrix_type	= Eigen::Matrix<value_type, DOF, DOF>;

    public:
      ICIAColorMoment(const Array2<C>& edgeH, const Array2<C>& edgeV)
	  :_edgeH(edgeH.cbegin(), edgeH.nrow()),
	   _edgeV(edgeV.cbegin(), edgeV.nrow())
      {
      }

      template <class C_=C> __host__ __device__
      std::enable_if_t<std::is_arithmetic<C_>::value, array_type>
      operator ()(int i) const
      {
	  const int	v = i / ncol();
	  const int	u = i - (v * ncol());
	  const auto	s = 1 / value_type(max(nrow(), ncol()));

	  return MAP::image_derivative0(s*u, s*v, _edgeH[v][u], _edgeV[v][u])
		.template ext();
      }

      template <class C_=C> __host__ __device__
      std::enable_if_t<!std::is_arithmetic<C_>::value, array_type>
      operator ()(int i) const
      {
	  const int	v  = i / ncol();
	  const int	u  = i - (v * ncol());
	  const C	eH = _edgeH[v][u];
	  const C	eV = _edgeV[v][u];
	  const auto	s  = 1 / value_type(max(nrow(), ncol()));
	  const auto	uf = s * u;
	  const auto	vf = s * v;
	  auto		a  = MAP::image_derivative0(uf, vf, eH.x, eV.x);
	  auto		m  = a.template ext();

	  a  = MAP::image_derivative0(uf, vf, eH.y, eV.y);
	  m += a.template ext();
	  a  = MAP::image_derivative0(uf, vf, eH.z, eV.z);
	  m += a.template ext();

	  return m;
      }

      int
      size() const
      {
	  return nrow() * ncol();
      }

      static matrix_type
      M(const array_type& moment)
      {
	  matrix_type	m;
	  auto		p = moment.data();
	  for (int i = 0; i < m.rows(); ++i)
	      for (int j = i; j < m.cols(); ++j)
		  m(j, i) = m(i, j) = *p++;
	  return m;
      }

    private:
      __host__ __device__ __forceinline__
      int	nrow()		const	{ return _edgeH.size(); }
      __host__ __device__ __forceinline__
      int	ncol()		const	{ return _edgeH.cbegin().size(); }

    private:
      const image_type	_edgeH;		// source horizontal gradient image
      const image_type	_edgeV;		// source vertcial gradient image
  };

  template <class MAP, class C>
  class ICIAColorDeviation
  {
    private:
      using image_type	= range<range_iterator<thrust::device_ptr<const C> > >;
      constexpr static size_t	DOF = MAP::DOF;

    public:
      using value_type	= typename MAP::element_type;
      using array_type	= array<value_type, DOF+2>;
      using vector_type	= Eigen::Matrix<value_type, DOF, 1>;

    public:
      ICIAColorDeviation(const MAP& Mts,
			 const Array2<C>& edgeH, const Array2<C>& edgeV,
			 const Array2<C>& source, const Texture<C>& target,
			 value_type color_thresh)
	  :_Mts(Mts),
	   _edgeH(edgeH.cbegin(), edgeH.nrow()),
	   _edgeV(edgeV.cbegin(), edgeV.nrow()),
	   _source(source.cbegin(), source.nrow()),
	   _target(target),
	   _sqcolor_thresh(color_thresh*color_thresh)
      {
      }

      template <class C_=C> __host__ __device__
      std::enable_if_t<std::is_arithmetic<C_>::value, array_type>
      operator ()(int i) const
      {
	  const int	v    = i / ncol();
	  const int	u    = i - (v * ncol());
	  const auto	uv_t = _Mts(u, v);

	  if (0 <= uv_t.x && uv_t.x < ncol() && 0 <= uv_t.y && uv_t.y < nrow())
	  {
	      const auto	c   = _source[v][u];
	      const auto	c_t = _target(uv_t.x, uv_t.y);
	      const auto	b   = c - c_t;

	      if (c != C(0) && c_t != C(0) && b*b < _sqcolor_thresh)
	      {
		  const auto	s  = 1 / value_type(max(nrow(), ncol()));
		  const auto	ab = MAP::image_derivative0(s*u, s*v,
							    _edgeH[v][u],
							    _edgeV[v][u])
				   * b;
		  auto		d  = ab.template extend<DOF+2>();
		  d[DOF]   = b*b;
		  d[DOF+1] = 1;

		  return d;
	      }
	  }

	  return {0};
      }

      template <class C_=C> __host__ __device__
      std::enable_if_t<!std::is_arithmetic<C_>::value, array_type>
      operator ()(int i) const
      {
	  const int	v    = i / ncol();
	  const int	u    = i - (v * ncol());
	  const auto	uv_t = _Mts(u, v);

	  if (0 <= uv_t.x && uv_t.x < ncol() && 0 <= uv_t.y && uv_t.y < nrow())
	  {
	      const auto	c   = _source[v][u];
	      const auto	c_t = _target(uv_t.x, uv_t.y);
	      const auto	b   = c - c_t;

	      if (valid(c) && valid(c_t) && square(b) < _sqcolor_thresh)
	      {
		  const C	eH = _edgeH[v][u];
		  const C	eV = _edgeV[v][u];
		  const auto	s  = 1 / value_type(max(nrow(), ncol()));
		  const auto	uf = s * u;
		  const auto	vf = s * v;
		  const auto	ab = MAP::image_derivative0(uf, vf, eH.x, eV.x)
				   * b.x
				   + MAP::image_derivative0(uf, vf, eH.y, eV.y)
				   * b.y
				   + MAP::image_derivative0(uf, vf, eH.z, eV.z)
				   * b.z;
		  auto		d  = ab.template extend<DOF+2>();
		  d[DOF]   = square(b);
		  d[DOF+1] = 1;

		  return d;
	      }
	  }

	  return {0};
      }

      int
      size() const
      {
	  return nrow() * ncol();
      }

      vector_type&
      unnormalize_updates(vector_type& updates) const
      {
	  MAP::unnormalize_updates(updates.data(),
				   1 / value_type(max(nrow(), ncol())));
	  return updates;
      }

      static vector_type
      d(const array_type& deviation)
      {
	  vector_type	v;
	  for (int i = 0; i < v.rows(); ++i)
	      v(i) = deviation[i];
	  return v;
      }

      static value_type
      npoints(const array_type& deviation)
      {
	  return deviation[DOF+1];
      }

      static value_type
      sqerr(const array_type& deviation)
      {
	  return deviation[DOF];
      }

      static value_type
      mse(const array_type& deviation)
      {
	  return deviation[DOF] / deviation[DOF+1];
      }

      template <class C_> __host__ __device__ static bool
      valid(const C_& c)
      {
	  return c.x != 0 || c.y != 0 || c.z != 0;
      }

    private:
      __host__ __device__ __forceinline__
      int	nrow()		const	{ return _source.size(); }
      __host__ __device__ __forceinline__
      int	ncol()		const	{ return _source.cbegin().size(); }

    private:
      const MAP		_Mts;		// map from source to destination image
      const image_type	_edgeH;		// source horizontal gradient image
      const image_type	_edgeV;		// source vertcial gradient image
      const image_type	_source;	// source color image
      const Texture<C>	_target;	// target color image
      const value_type	_sqcolor_thresh;
  };
}	// namespace detail

/************************************************************************
*  class ICIA<MAP, C, CLOCK>						*
************************************************************************/
template <class MAP, class C, class CLOCK=void>
class ICIA : public Profiler<CLOCK>
{
  public:
    using map_type	= MAP;
    using color_type	= C;
    using image_type	= Array2<color_type>;
    using value_type	= typename map_type::element_type;

    struct Parameters
    {
	float		sigma		= 2.0;
	value_type	color_thresh	= 20;
	value_type	tol		= 1.0e-4;
	size_t		niter_max	= 100;
    };

  private:
    constexpr static size_t	DOF = map_type::DOF;

    using matrix_type	= Eigen::Matrix<value_type, DOF, DOF>;
    using profiler_type	= Profiler<CLOCK>;

  public:
		ICIA(const Parameters& params=Parameters())
		    :profiler_type(2), _params(params),
		     _source(), _edgeH(), _edgeV(), _M()		{}

    const Parameters&
		getParameters()			const	{ return _params; }
    void	setParameters(const Parameters& params)	{ _params = params; }
    const image_type&
		getSourceImage()		const	{ return _source; }
    const image_type&
		getEdgeH()			const	{ return _edgeH; }
    const image_type&
		getEdgeV()			const	{ return _edgeV; }
    bool	empty()						const	;
    void	clearSourceImage()					;
    void	setSourceImage(const image_type& source)		;
    void	setSourceImage(image_type&& source)			;
    void	swapSourceImage(image_type& source)			;
    value_type	operator ()(const image_type& target, MAP& Mts)	const	;
    value_type	operator ()(const image_type& source,
			    const image_type& target, MAP& Mts)		;

  private:
    void	computeEdgesAndMoment()					;

  private:
    Parameters	_params;
    image_type	_source;	// current reference source image
    image_type	_edgeH;		// horizontal derivative of source image
    image_type	_edgeV;		// vertical derivative of source image
    matrix_type	_M;		// color moment matrix
};

template <class MAP, class C, class CLOCK> bool
ICIA<MAP, C, CLOCK>::empty() const
{
    return _source.nrow() == 0;
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::clearSourceImage()
{
    _source.resize(0, 0);
    _edgeH.resize(0, 0);
    _edgeV.resize(0, 0);
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::setSourceImage(const image_type& source)
{
    _source = source;

    computeEdgesAndMoment();
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::setSourceImage(image_type&& source)
{
    _source = std::move(source);

    computeEdgesAndMoment();
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::swapSourceImage(image_type& source)
{
    _source.swap(source);

    computeEdgesAndMoment();
}

template <class MAP, class C, class CLOCK>
typename ICIA<MAP, C, CLOCK>::value_type
ICIA<MAP, C, CLOCK>::operator ()(const image_type& target, MAP& Mts) const
{
    using deviation_type	= detail::ICIAColorDeviation<MAP, C>;
    using deviation_array_type	= typename deviation_type::array_type;

  // Convert the error moment to a matrix and save its diagonals.
    const Texture<C>	target_tex(target);
    auto		Mts_old = Mts;
    auto		mse_old = std::numeric_limits<value_type>::max();
    auto		mse_prev = mse_old;
    value_type		lambda  = 1.0e-3;
    for (size_t n = 0; n < _params.niter_max; ++n)
    {
      // Compute error derivation vector by parallel reduction.
	const deviation_type		deviation(Mts, _edgeH, _edgeV,
						  _source, target_tex,
						  _params.color_thresh);
	Array<deviation_array_type>	tmp_deviation(1);
	size_t				tmp_size = 0;
	cub::DeviceReduce::Sum(nullptr, tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   deviation),
			       tmp_deviation.begin(), deviation.size());
	Array<uint8_t>	tmp(tmp_size);
	cub::DeviceReduce::Sum(tmp.data().get(), tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   deviation),
			       tmp_deviation.begin(), deviation.size());
	gpuCheckLastError();
	const deviation_array_type	deviation_array = tmp_deviation[0];

      // Evaluate residual mean square_error.
	const auto	mse = deviation_type::mse(deviation_array);
#if !defined(NDEBUG)
	std::cerr << "      mse=" << mse << ", mse_old=" << mse_old
		  << ", mse_absdiff=" << std::abs(mse - mse_old)
		  << ", sqerr="   << deviation_type::sqerr(deviation_array)
		  << ", npoints=" << deviation_type::npoints(deviation_array)
		  << std::endl;
#endif
	if (isnan(mse))
	    return mse;

	if (mse < mse_old)
	{
	    if (std::abs(mse - mse_old) <= _params.tol)
	    {
		return mse;
	    }

	    Mts_old = Mts;
	    mse_old = mse;
	    lambda *= 0.1;
	}
	else
	{
	    if (std::abs(mse - mse_prev) <= _params.tol || lambda < 1.0e-20)
	    {
		Mts = Mts_old;
		return mse_old;
	    }

	    lambda *= 10.0;
	}
	mse_prev = mse;

      // Solve the linear system for updates of transform.
	auto		A = _M;
	for (size_t i = 0; i < A.rows(); ++i)
	    A(i, i) *= (1.0 + lambda);
	const auto	b     = deviation_type::d(deviation_array);
	auto		delta = A.ldlt().solve(b).eval();
	deviation.unnormalize_updates(delta);
	Mts = Mts_old * MAP::exp(delta.data());
#if !defined(NDEBUG)
	std::cerr << "  [" << n << "] err=" << std::sqrt(mse)
		  << ", lambda=" << lambda << std::endl;
#endif
#if defined(DEBUG)
	image_type	source(target.nrow(), target.ncol());
	source = 0;
	warp(target, source.begin(), Mts);
	TU::Image<C>	diff = TU::Array2<C>(_source) - TU::Array2<C>(source);
	diff.saveData(std::cout, ImageFormat::FLOAT);
	usleep(50000);
#endif
    }

    throw std::runtime_error("ICIA::operator (): maximum iteration limit exceeded!");

    return -1.0;
}

template <class MAP, class C, class CLOCK>
typename ICIA<MAP, C, CLOCK>::value_type
ICIA<MAP, C, CLOCK>::operator ()(const image_type& source,
				 const image_type& target, MAP& Mts)
{
#if defined(DEBUG)
    Image<float>	diff(source.ncol(), source.nrow());
    std::cout << 'M' << 1 << std::endl;
    diff.saveHeader(std::cout, ImageFormat::FLOAT);
#endif
    profiler_type::start(0);
    setSourceImage(source);
    profiler_type::start(1);
    const auto	mse = (*this)(target, Mts);
    profiler_type::nextFrame();

    return mse;
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::computeEdgesAndMoment()
{
    using moment_type		= detail::ICIAColorMoment<MAP, C>;
    using moment_array_type	= typename moment_type::array_type;

  // Compute horizontal and vertical image derivatives.
    _edgeH.resize(_source.nrow(), _source.ncol());
    _edgeV.resize(_source.nrow(), _source.ncol());
    FIRGaussianConvolver2<C>	convolver(_params.sigma);
    convolver.diffH(_source.cbegin(), _source.cend(), _edgeH.begin(), true);
    convolver.diffV(_source.cbegin(), _source.cend(), _edgeV.begin(), true);

  // Compute error moment matrix by parallel reduction.
    const moment_type		color_moment(_edgeH, _edgeV);
    Array<moment_array_type>	tmp_moment(1);
    size_t			tmp_size = 0;
    cub::DeviceReduce::Sum(nullptr, tmp_size,
			   thrust::make_transform_iterator(
			       thrust::make_counting_iterator(0),
			       color_moment),
			   tmp_moment.begin(), color_moment.size());
    Array<uint8_t>	tmp(tmp_size);
    cub::DeviceReduce::Sum(tmp.data().get(), tmp_size,
			   thrust::make_transform_iterator(
			       thrust::make_counting_iterator(0),
			       color_moment),
			   tmp_moment.begin(), color_moment.size());
    gpuCheckLastError();

    _M = moment_type::M(tmp_moment[0]);
}

}	// namespace cu
}	// namespace TU
