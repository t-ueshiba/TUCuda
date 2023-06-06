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

	  if (u >= ncol() || v >= nrow())
	      return {0};

	  const auto	s  = 1 / value_type(max(nrow(), ncol()));
	  
	  return MAP::image_derivative0(_edgeH[v][u], _edgeV[v][u], s*u, s*v)
		.template ext();
      }

      template <class C_=C> __host__ __device__
      std::enable_if_t<!std::is_arithmetic<C_>::value, moment_type>
      operator ()(int i) const
      {
	  const int	v = i / ncol();
	  const int	u = i - (v * ncol());

	  if (u >= ncol() || v >= nrow())
	      return {0};

	  const C	eH = _edgeH[v][u];
	  const C	eV = _edgeV[v][u];
	  const auto	s  = 1 / value_type(max(nrow(), ncol()));
	  const auto	uf = s * u;
	  const auto	vf = s * v;
	  auto		a  = MAP::image_derivative0(eH.x, eV.x, uf, vf);
	  auto		m  = a.template ext();
	  
	  a  = MAP::image_derivative0(eH.y, eV.y, uf, vf);
	  m += a.template ext();
	  a  = MAP::image_derivative0(eH.z, eV.z, uf, vf);
	  m += a.template ext();

	  return m;
      }

      __host__ __device__ __forceinline__
      int	size()		const	{ return nrow() * ncol(); }
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
      mse(const deviation_type& deviation)
      {
	  return deviation[DOF] / deviation[DOF+1];
      }

      deviation_vector_type&
      unnormalize_updates(deviation_vector_type& updates) const
      {
	  MAP::unnormalize_updates(updates.data(), 
				   1 / value_type(max(nrow(), ncol())));
	  return updates;
      }
      
      ICIAErrorDeviation(const MAP& map,
			 const Array2<C>& edgeH, const Array2<C>& edgeV,
			 const Array2<C>& colors, cudaTextureObject_t colors_p,
			 value_type sqcolor_thresh)
	  :_map(map),
	   _edgeH(edgeH.cbegin(), edgeH.nrow()),
	   _edgeV(edgeV.cbegin(), edgeV.nrow()),
	   _colors(colors.cbegin(), colors.nrow()),
	   _colors_p(colors_p),
	   _sqcolor_thresh(sqcolor_thresh)
      {
      }
	
      template <class C_=C> __host__ __device__
      std::enable_if_t<std::is_arithmetic<C_>::value, deviation_type>
      operator ()(int i) const
      {
	  const int	v    = i / ncol();
	  const int	u    = i - (v * ncol());
	  const auto	uv_p = _map(u, v);

	  if (u < ncol() && v < nrow() &&
	      0 <= uv_p.x && uv_p.x < ncol() && 0 <= uv_p.y && uv_p.y < nrow())
	  {
	      const auto	b = _colors[v][u] - tex2D<C>(_colors_p,
							     uv_p.x, uv_p.y);

	      if (b*b < _sqcolor_thresh)
	      {
		  const auto	s  = 1 / value_type(max(nrow(), ncol()));
		  const auto	ab = MAP::image_derivative0(_edgeH[v][u],
							    _edgeV[v][u],
							    s*u, s*v)
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

	  if (u < ncol() && v < nrow() &&
	      0 <= uv_p.x && uv_p.x < ncol() && 0 <= uv_p.y && uv_p.y < nrow())
	  {
	      const auto	b = _colors[v][u] - tex2D<C>(_colors_p,
							     uv_p.x, uv_p.y);

	      if (square(b) < _sqcolor_thresh)
	      {
		  const C	eH = _edgeH[v][u];
		  const C	eV = _edgeV[v][u];
		  const auto	s  = 1 / value_type(max(nrow(), ncol()));
		  const auto	uf = s * u;
		  const auto	vf = s * v;
		  const auto	ab = MAP::image_derivative0(eH.x, eV.x, uf, vf)
				   * b.x
				   + MAP::image_derivative0(eH.y, eV.y, uf, vf)
				   * b.y
				   + MAP::image_derivative0(eH.z, eV.z, uf, vf)
				   * b.z;
		  auto		d  = ab.template extend<DOF+2>();
		  d[DOF]   = sqaure(b);
		  d[DOF+1] = 1;
		  
		  return d;
	      }
	  }

	  return {0};
      }

      __host__ __device__ __forceinline__
      int	size()		const	{ return nrow() * ncol(); }
      __host__ __device__ __forceinline__
      int	nrow()		const	{ return _colors.size(); }
      __host__ __device__ __forceinline__
      int	ncol()		const	{ return _colors.cbegin().size(); }

    private:
      const MAP			_map;
      const colors_type		_edgeH;
      const colors_type		_edgeV;
      const colors_type		_colors;
      const cudaTextureObject_t	_colors_p;
      const value_type		_sqcolor_thresh;
  };
}	// namespace detail
    
/************************************************************************
*  class ICIA<MAP, CLOCK>						*
************************************************************************/
template <class MAP, class CLOCK=void>
class ICIA : public Profiler<CLOCK>
{
  public:
    constexpr static size_t	DOF = MAP::DOF;

    using value_type		= typename MAP::element_type;
    using moment_type		= array<value_type, DOF*(DOF+1)/2>;
    using deviation_type	= array<value_type, DOF+2>;

    struct Parameters
    {
	float		sigma		= 2.0;
	value_type	sqcolor_thresh	= 50*50;
	value_type	tol		= 1.0e-2;
	size_t		niter_max	= 100;
    };

  private:
    using profiler_t		= Profiler<CLOCK>;
    
  public:
		ICIA(const Parameters& params=Parameters())
		    :profiler_t(3), _params(params),
		     _tmp(), _moment(1), _deviation(1) 			{}

    template <class C_>
    value_type	operator ()(const Array2<C_>& src,
			    const Array2<C_>& dst, MAP& f)	const	;

  private:
    Parameters				_params;

  // Temporary buffers
    mutable Array<uint8_t>		_tmp;	// for CUB
    mutable Array<moment_type>		_moment;
    mutable Array<deviation_type>	_deviation;
};

template <class MAP, class CLOCK> template <class C_>
typename ICIA<MAP, CLOCK>::value_type
ICIA<MAP, CLOCK>::operator ()(const Array2<C_>& src,
			      const Array2<C_>& dst, MAP& map) const
{
#if !defined(NDEBUG)
    Image<float>	diff(src.ncol(), src.nrow());
    std::cout << 'M' << 1 << std::endl;
    diff.saveHeader(std::cout, ImageFormat::FLOAT);
#endif
    using error_moment_type	= detail::ICIAErrorMoment<MAP, C_>;
    using error_deviation_type	= detail::ICIAErrorDeviation<MAP, C_>;

  // Compute horizontal and vertical derivatives.
    profiler_t::start(0);
    Array2<C_>			edgeH(src.nrow(), src.ncol());
    Array2<C_>			edgeV(src.nrow(), src.ncol());
    FIRGaussianConvolver2<C_>	convolver(_params.sigma);
    convolver.diffH(src.cbegin(), src.cend(), edgeH.begin(), true);
    convolver.diffV(src.cbegin(), src.cend(), edgeV.begin(), true);

  // Allocate temporary storage for parallel reduction.
    profiler_t::start(1);
    error_moment_type		error_moment(edgeH, edgeV);
    size_t			tmp_size = 0;
    cub::DeviceReduce::Sum(nullptr, tmp_size,
			   thrust::make_transform_iterator(
			       thrust::make_counting_iterator(0),
			       error_moment),
			   _moment.begin(), error_moment.size());
    if (tmp_size > _tmp.size())
	_tmp.resize(tmp_size);

  // Perform parallel reduction of moments for source image.
    cub::DeviceReduce::Sum(_tmp.data().get(), tmp_size,
			   thrust::make_transform_iterator(
			       thrust::make_counting_iterator(0),
			       error_moment),
			   _moment.begin(), error_moment.size());
    gpuCheckLastError();

  // Convert the error moment to a matrix and save its diagonals.
    const moment_type		moment = _moment[0];
    auto			A = error_moment_type::A(moment);
    std::array<value_type, DOF>	diagA;
    for (size_t i = 0; i < diagA.size(); ++i)
	diagA[i] = A(i, i);

    profiler_t::start(2);
    Texture<C_>	dst_tex(dst);
    auto	map_old = map;
    auto	mse_old = std::numeric_limits<value_type>::max();
    value_type	lambda  = 1.0e-4;
    for (size_t n = 0; n < _params.niter_max; ++n)
    {
      // Allocate temporary storage for parallel reduction.
	error_deviation_type	error_deviation(map, edgeH, edgeV,
						src, dst_tex.get(),
						_params.sqcolor_thresh);
	size_t			tmp_size = 0;
	cub::DeviceReduce::Sum(nullptr, tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   error_deviation),
			       _deviation.begin(), error_deviation.size());
	if (tmp_size > _tmp.size())
	    _tmp.resize(tmp_size);

      // Perform parallel reduction of deviations between source and
      // destination images.
	cub::DeviceReduce::Sum(_tmp.data().get(), tmp_size,
			       thrust::make_transform_iterator(
				   thrust::make_counting_iterator(0),
				   error_deviation),
			       _deviation.begin(), error_deviation.size());
	gpuCheckLastError();

	const deviation_type	deviation = _deviation[0];
	const auto		mse = error_deviation_type::mse(deviation);

	std::cerr << "err_old=" << std::sqrt(mse_old)
		  << ", err=" << std::sqrt(mse)
		  << std::endl;
	
	if (mse < mse_old)
	{
	    if (std::abs(mse - mse_old) <= _params.tol)
	    {
		profiler_t::nextFrame();
		return mse;
	    }

	    mse_old = mse;
	    lambda *= 0.1;
	}
	else
	{
	    map = map_old;

	    if (lambda < 1.0e-20)
	    {
		profiler_t::nextFrame();
		return mse_old;
	    }

	    lambda *= 10.0;
	}
	
      // Solve the linear system for updates of transform.
	map_old = map;
	for (size_t i = 0; i < diagA.size(); ++i)
	    A(i, i) = (1.0 + lambda) * diagA[i];
	const auto	b     = error_deviation_type::b(deviation);
	auto		delta = A.ldlt().solve(b).eval();
	error_deviation.unnormalize_updates(delta);
	map = map_old * MAP::exp(delta.data());

#if !defined(NDEBUG)
	std::cerr << "  n=" << n << ", err=" << std::sqrt(mse)
		  << ", lambda=" << lambda << std::endl;
	Array2<C_>	warped(dst.nrow(), dst.ncol());
	warp(dst, warped.begin(), map);
	diff = TU::Array2<C_>(src) - TU::Array2<C_>(warped);
	diff.saveData(std::cout, ImageFormat::FLOAT);
	usleep(50000);
#endif
    }

    throw std::runtime_error("ICIA::operator (): maximum iteration limit exceeded!");

    return -1.0;
}

}	// namespace cu
}	// namespace TU
