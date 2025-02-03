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
  class ICIAErrorMoment
  {
    private:
      using colors_type	= range<range_iterator<thrust::device_ptr<const C> > >;
      constexpr static size_t	DOF = MAP::DOF;

    public:
      using value_type		= typename MAP::element_type;
      using moment_type		= array<value_type, DOF*(DOF+1)/2>;
      using moment_matrix_type	= Eigen::Matrix<value_type, DOF, DOF>;

    public:
      ICIAErrorMoment(const Array2<C>& edgeH, const Array2<C>& edgeV)
	  :_edgeH(edgeH.cbegin(), edgeH.nrow()),
	   _edgeV(edgeV.cbegin(), edgeV.nrow())
      {
      }

      template <class C_=C> __host__ __device__
      std::enable_if_t<std::is_arithmetic<C_>::value, moment_type>
      operator ()(int i) const
      {
	  const int	v = i / ncol();
	  const int	u = i - (v * ncol());
	  const auto	s = 1 / value_type(max(nrow(), ncol()));

	  return MAP::image_derivative0(s*u, s*v, _edgeH[v][u], _edgeV[v][u])
		.template ext();
      }

      template <class C_=C> __host__ __device__
      std::enable_if_t<!std::is_arithmetic<C_>::value, moment_type>
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

      static moment_matrix_type
      A(const moment_type& moment)
      {
	  moment_matrix_type	m;
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
      const colors_type	_edgeH;
      const colors_type	_edgeV;
  };

  template <class MAP, class C>
  class ICIAErrorDeviation
  {
    private:
      using colors_type	= range<range_iterator<thrust::device_ptr<const C> > >;
      constexpr static size_t	DOF = MAP::DOF;

    public:
      using value_type			= typename MAP::element_type;
      using deviation_type		= array<value_type, DOF+2>;
      using deviation_vector_type	= Eigen::Matrix<value_type, DOF, 1>;

    public:
      ICIAErrorDeviation(const MAP& map,
			 const Array2<C>& edgeH, const Array2<C>& edgeV,
			 const Array2<C>& colors, const Texture<C>& colors_p,
			 value_type color_thresh)
	  :_map(map),
	   _edgeH(edgeH.cbegin(), edgeH.nrow()),
	   _edgeV(edgeV.cbegin(), edgeV.nrow()),
	   _colors(colors.cbegin(), colors.nrow()),
	   _colors_p(colors_p),
	   _sqcolor_thresh(color_thresh*color_thresh)
      {
      }

      template <class C_=C> __host__ __device__
      std::enable_if_t<std::is_arithmetic<C_>::value, deviation_type>
      operator ()(int i) const
      {
	  const int	v    = i / ncol();
	  const int	u    = i - (v * ncol());
	  const auto	uv_p = _map(u, v);

	  if (0 <= uv_p.x && uv_p.x < ncol() && 0 <= uv_p.y && uv_p.y < nrow())
	  {
	      const auto	c   = _colors[v][u];
	      const auto	c_p = _colors_p(uv_p.x, uv_p.y);
	      const auto	b   = c - c_p;

	      if (c != C(0) && c_p != C(0) && b*b < _sqcolor_thresh)
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
      std::enable_if_t<!std::is_arithmetic<C_>::value, deviation_type>
      operator ()(int i) const
      {
	  const int	v    = i / ncol();
	  const int	u    = i - (v * ncol());
	  const auto	uv_p = _map(u, v);

	  if (0 <= uv_p.x && uv_p.x < ncol() && 0 <= uv_p.y && uv_p.y < nrow())
	  {
	      const auto	c   = _colors[v][u];
	      const auto	c_p = _colors_p(uv_p.x, uv_p.y);
	      const auto	b   = c - c_p;

	      if (valid(c) && valid(c_p) && square(b) < _sqcolor_thresh)
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

      deviation_vector_type&
      unnormalize_updates(deviation_vector_type& updates) const
      {
	  MAP::unnormalize_updates(updates.data(),
				   1 / value_type(max(nrow(), ncol())));
	  return updates;
      }

      static deviation_vector_type
      b(const deviation_type& deviation)
      {
	  deviation_vector_type	v;
	  for (int i = 0; i < v.rows(); ++i)
	      v(i) = deviation[i];
	  return v;
      }

      static value_type
      npoints(const deviation_type& deviation)
      {
	  return deviation[DOF+1];
      }

      static value_type
      sqerr(const deviation_type& deviation)
      {
	  return deviation[DOF];
      }

      static value_type
      mse(const deviation_type& deviation)
      {
	  return deviation[DOF] / deviation[DOF+1];
      }

      template <class C_> __host__ __device__ static bool
      square(const C_& c)
      {
	  return c.x*c.x + c.y*c.y + c.z*c.z;
      }

      template <class C_> __host__ __device__ static bool
      valid(const C_& c)
      {
	  return c.x != 0 || c.y != 0 || c.z != 0;
      }

    private:
      __host__ __device__ __forceinline__
      int	nrow()		const	{ return _colors.size(); }
      __host__ __device__ __forceinline__
      int	ncol()		const	{ return _colors.cbegin().size(); }

    private:
      const MAP		_map;
      const colors_type	_edgeH;
      const colors_type	_edgeV;
      const colors_type	_colors;
      const Texture<C>	_colors_p;
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
    constexpr static size_t	DOF = MAP::DOF;

    using image_type		= Array2<C>;
    using value_type		= typename MAP::element_type;
    using moment_type		= array<value_type, DOF*(DOF+1)/2>;
    using deviation_type	= array<value_type, DOF+2>;

    struct Parameters
    {
	float		sigma		= 2.0;
	value_type	color_thresh	= 20;
	value_type	tol		= 1.0e-4;
	size_t		niter_max	= 100;
    };

  private:
    using profiler_t		= Profiler<CLOCK>;

  public:
		ICIA(const Parameters& params=Parameters())
		    :profiler_t(2), _params(params),
		     _src(), _edgeH(), _edgeV(),
		     _tmp(), _moment(1), _deviation(1)			{}

    const Parameters&
		getParameters()			const	{ return _params; }
    void	setParameters(const Parameters& params)	{ _params = params; }
    const image_type&
		getSourceImage()		const	{ return _src; }
    const image_type&
		getEdgeH()			const	{ return _edgeH; }
    const image_type&
		getEdgeV()			const	{ return _edgeV; }
    bool	empty()						const	;
    void	clearSourceImage()					;
    void	setSourceImage(const image_type& src)			;
    value_type	operator ()(const image_type& dst, MAP& f)	const	;
    value_type	operator ()(const image_type& src,
			    const image_type& dst, MAP& f)		;

  private:
    Parameters				_params;
    image_type				_src;
    image_type				_edgeH;
    image_type				_edgeV;

  // Temporary buffers
    mutable Array<uint8_t>		_tmp;	// for CUB
    mutable Array<moment_type>		_moment;
    mutable Array<deviation_type>	_deviation;
};

template <class MAP, class C, class CLOCK> bool
ICIA<MAP, C, CLOCK>::empty() const
{
    return _src.nrow() == 0;
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::clearSourceImage()
{
    _src.resize(0, 0);
    _edgeH.resize(0, 0);
    _edgeV.resize(0, 0);
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::setSourceImage(const image_type& src)
{
    _src = src;
    _edgeH.resize(src.nrow(), src.ncol());
    _edgeV.resize(src.nrow(), src.ncol());

    FIRGaussianConvolver2<C>	convolver(_params.sigma);
    convolver.diffH(src.cbegin(), src.cend(), _edgeH.begin(), true);
    convolver.diffV(src.cbegin(), src.cend(), _edgeV.begin(), true);
}

template <class MAP, class C, class CLOCK>
typename ICIA<MAP, C, CLOCK>::value_type
ICIA<MAP, C, CLOCK>::operator ()(const image_type& dst, MAP& map) const
{
    using error_moment_type	= detail::ICIAErrorMoment<MAP, C>;
    using error_deviation_type	= detail::ICIAErrorDeviation<MAP, C>;

  // Compute error moment matrix by parallel reduction.
    const error_moment_type	error_moment(_edgeH, _edgeV);
    size_t			tmp_size = 0;
    cub::DeviceReduce::Sum(nullptr, tmp_size,
			   thrust::make_transform_iterator(
			       thrust::make_counting_iterator(0),
			       error_moment),
			   _moment.begin(), error_moment.size());
    if (tmp_size > _tmp.size())
	_tmp.resize(tmp_size);
    cub::DeviceReduce::Sum(_tmp.data().get(), tmp_size,
			   thrust::make_transform_iterator(
			       thrust::make_counting_iterator(0),
			       error_moment),
			   _moment.begin(), error_moment.size());
    gpuCheckLastError();
    const moment_type	moment = _moment[0];

  // Convert the error moment to a matrix and save its diagonals.
    const Texture<C>	dst_tex(dst);
    auto		map_old = map;
    auto		mse_old = std::numeric_limits<value_type>::max();
    auto		mse_prev = mse_old;
    value_type		lambda  = 1.0e-3;
    for (size_t n = 0; n < _params.niter_max; ++n)
    {
      // Compute error derivation vector by parallel reduction.
	const error_deviation_type	error_deviation(map, _edgeH, _edgeV,
							_src, dst_tex,
							_params.color_thresh);
	size_t				tmp_size = 0;
	cub::DeviceReduce::Sum(nullptr, tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   error_deviation),
			       _deviation.begin(), error_deviation.size());
	gpuCheckLastError();
	if (tmp_size > _tmp.size())
	    _tmp.resize(tmp_size);
	cub::DeviceReduce::Sum(_tmp.data().get(), tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   error_deviation),
			       _deviation.begin(), error_deviation.size());
	gpuCheckLastError();
	const deviation_type	deviation = _deviation[0];

      // Evaluate residual mean square_error.
	const auto		mse = error_deviation_type::mse(deviation);
#if !defined(NDEBUG)
	std::cerr << "      mse=" << mse << ", mse_old=" << mse_old
		  << ", mse_absdiff=" << std::abs(mse - mse_old)
		  << ", sqerr="   << error_deviation_type::sqerr(deviation)
		  << ", npoints=" << error_deviation_type::npoints(deviation)
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

	    map_old = map;
	    mse_old = mse;
	    lambda *= 0.1;
	}
	else
	{
	    if (std::abs(mse - mse_prev) <= _params.tol || lambda < 1.0e-20)
	    {
		map = map_old;
		return mse_old;
	    }

	    lambda *= 10.0;
	}
	mse_prev = mse;

      // Solve the linear system for updates of transform.
	auto		A = error_moment_type::A(moment);
	for (size_t i = 0; i < A.rows(); ++i)
	    A(i, i) *= (1.0 + lambda);
	const auto	b     = error_deviation_type::b(deviation);
	auto		delta = A.ldlt().solve(b).eval();
	error_deviation.unnormalize_updates(delta);
	map = map_old * MAP::exp(delta.data());
#if !defined(NDEBUG)
	std::cerr << "  [" << n << "] err=" << std::sqrt(mse)
		  << ", lambda=" << lambda << std::endl;
#endif
#if defined(DEBUG)
	image_type	warped(dst.nrow(), dst.ncol());
	warped = 0;
	warp(dst, warped.begin(), map);
	TU::Image<C>	diff = TU::Array2<C>(_src) - TU::Array2<C>(warped);
	diff.saveData(std::cout, ImageFormat::FLOAT);
	usleep(50000);
#endif
    }

    throw std::runtime_error("ICIA::operator (): maximum iteration limit exceeded!");

    return -1.0;
}

template <class MAP, class C, class CLOCK>
typename ICIA<MAP, C, CLOCK>::value_type
ICIA<MAP, C, CLOCK>::operator ()(const image_type& src,
				 const image_type& dst, MAP& map)
{
#if defined(DEBUG)
    Image<float>	diff(src.ncol(), src.nrow());
    std::cout << 'M' << 1 << std::endl;
    diff.saveHeader(std::cout, ImageFormat::FLOAT);
#endif
    profiler_t::start(0);
    setSourceImage(src);
    profiler_t::start(1);
    const auto	mse = (*this)(dst, map);
    profiler_t::nextFrame();

    return mse;
}

}	// namespace cu
}	// namespace TU
