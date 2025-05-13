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

namespace TU::cu
{
namespace icia
{
  template <class ICIA>
  class ColorMoment
  {
    private:
      using map_type	= typename ICIA::map_type;
      using color_type	= typename ICIA::color_type;
      using slice_type	= typename ICIA::slice_type;

      constexpr static size_t	DOF = map_type::DOF;

    public:
      using value_type	= typename map_type::element_type;
      using array_type	= array<value_type, DOF*(DOF+1)/2>;
      using matrix_type	= Eigen::Matrix<value_type, DOF, DOF>;

    public:
      ColorMoment(const slice_type& edgeH, const slice_type& edgeV)
	  :_edgeH(edgeH), _edgeV(edgeV)
      {
      }

      template <class C=color_type> __host__ __device__
      std::enable_if_t<std::is_arithmetic<C>::value, array_type>
      operator ()(int i) const
      {
	  const int	v = i / ncol();
	  const int	u = i - (v * ncol());
	  const auto	s = 1 / value_type(max(nrow(), ncol()));

	  return map_type::image_derivative0(s*u, s*v,
					     _edgeH[v][u], _edgeV[v][u])
		.template ext();
      }

      template <class C=color_type> __host__ __device__
      std::enable_if_t<!std::is_arithmetic<C>::value, array_type>
      operator ()(int i) const
      {
	  const int	v  = i / ncol();
	  const int	u  = i - (v * ncol());
	  const auto	eH = _edgeH[v][u];
	  const auto	eV = _edgeV[v][u];
	  const auto	s  = 1 / value_type(max(nrow(), ncol()));
	  const auto	uf = s * u;
	  const auto	vf = s * v;
	  auto		a  = map_type::image_derivative0(uf, vf, eH.x, eV.x);
	  auto		m  = a.template ext();

	  a  = map_type::image_derivative0(uf, vf, eH.y, eV.y);
	  m += a.template ext();
	  a  = map_type::image_derivative0(uf, vf, eH.z, eV.z);
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
      const slice_type	_edgeH;		// source horizontal gradient image
      const slice_type	_edgeV;		// source vertcial gradient image
  };

  template <class ICIA>
  class ColorDeviation
  {
    private:
      using map_type	= typename ICIA::map_type;
      using color_type	= typename ICIA::color_type;
      using image_type	= typename ICIA::image_type;
      using slice_type	= typename ICIA::slice_type;

      constexpr static size_t	DOF = map_type::DOF;

    public:
      using value_type	= typename map_type::element_type;
      using array_type	= array<value_type, DOF+2>;
      using vector_type	= Eigen::Matrix<value_type, DOF, 1>;

    public:
      ColorDeviation(const map_type& Mts,
		     const slice_type& edgeH,
		     const slice_type& edgeV,
		     const slice_type& source,
		     const image_type& target,
		     value_type color_thresh)
	  :_Mts(Mts),
	   _edgeH(edgeH),
	   _edgeV(edgeV),
	   _source(source),
	   _target(target),
	   _nrow_t(_target.nrow()),
	   _ncol_t(_target.ncol()),
	   _sqcolor_thresh(color_thresh*color_thresh)
      {
      }

      template <class C=color_type> __host__ __device__
      std::enable_if_t<std::is_arithmetic<C>::value, array_type>
      operator ()(int i) const
      {
	  const int	v    = i / ncol();
	  const int	u    = i - (v * ncol());
	  const auto	uv_t = _Mts(u, v);

	  if (0 <= uv_t.x && uv_t.x < _ncol_t &&
	      0 <= uv_t.y && uv_t.y < _nrow_t)
	  {
	      const auto	c   = _source[v][u];
	      const auto	c_t = _target(uv_t.x, uv_t.y);
	      const auto	b   = c - c_t;

	      if (c != C(0) && c_t != C(0) && b*b < _sqcolor_thresh)
	      {
		  const auto	s  = 1 / value_type(max(nrow(), ncol()));
		  const auto	ab = map_type::image_derivative0(s*u, s*v,
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

      template <class C=color_type> __host__ __device__
      std::enable_if_t<!std::is_arithmetic<C>::value, array_type>
      operator ()(int i) const
      {
	  const int	v    = i / ncol();
	  const int	u    = i - (v * ncol());
	  const auto	uv_t = _Mts(u, v);

	  if (0 <= uv_t.x && uv_t.x < _ncol_t &&
	      0 <= uv_t.y && uv_t.y < _nrow_t)
	  {
	      const auto	c   = _source[v][u];
	      const auto	c_t = _target(uv_t.x, uv_t.y);
	      const auto	b   = c - c_t;

	      if (valid(c) && valid(c_t) && square(b) < _sqcolor_thresh)
	      {
		  const auto	eH = _edgeH[v][u];
		  const auto	eV = _edgeV[v][u];
		  const auto	s  = 1 / value_type(max(nrow(), ncol()));
		  const auto	uf = s * u;
		  const auto	vf = s * v;
		  const auto	ab = map_type::image_derivative0(uf, vf,
								 eH.x, eV.x)
				   * b.x
				   + map_type::image_derivative0(uf, vf,
								 eH.y, eV.y)
				   * b.y
				   + map_type::image_derivative0(uf, vf,
								 eH.z, eV.z)
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
	  map_type::unnormalize_updates(updates.data(),
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

      value_type
      overlap(const array_type& deviation) const
      {
	  return deviation[DOF+1] / size();
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

      template <class C> __host__ __device__ static bool
      valid(const C& c)
      {
	  return c.x != 0 || c.y != 0 || c.z != 0;
      }

    private:
      __host__ __device__ __forceinline__
      int	nrow()		const	{ return _source.size(); }
      __host__ __device__ __forceinline__
      int	ncol()		const	{ return _source.cbegin().size(); }

    private:
      const map_type		_Mts;	// map from source to destination image
      const slice_type		_edgeH;	// source horizontal gradient image
      const slice_type		_edgeV;	// source vertcial gradient image
      const slice_type		_source;	// source color image
      const Texture<color_type>	_target;	// target color image
      const int			_nrow_t;
      const int			_ncol_t;
      const value_type		_sqcolor_thresh;
  };
}	// namespace icia

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
    using slice_type	= range<range_iterator<
			      thrust::device_ptr<const color_type> > >;
    using value_type	= typename map_type::element_type;

    struct Parameters
    {
	float		sigma		= 2.0;
	value_type	color_thresh	= 20;
	value_type	tol		= 1.0e-4;
	size_t		niter_max	= 100;

	friend std::ostream&
	operator <<(std::ostream& out, const Parameters& params)
	{
	    return out << "sigma="	    << params.sigma
		       << ", color_thresh=" << params.color_thresh
		       << ", tol="	    << params.tol
		       << ", niter_max="    << params.niter_max;
	}
    };

    struct Frame
    {
	image_type	image;
	image_type	edgeH;
	image_type	edgeV;

	size_t	nrow()	const	{ return image.nrow(); }
	size_t	ncol()	const	{ return image.ncol(); }
	void	swap(Frame& frame)
		{
		    image.swap(frame.image);
		    edgeH.swap(frame.edgeH);
		    edgeV.swap(frame.edgeV);
		}
	void	clear()
		{
		    image.resize(0, 0);
		    edgeH.resize(0, 0);
		    edgeV.resize(0, 0);
		}
    };
    
    struct Result
    {
	value_type	mse;
	value_type	overlap;
    };

  private:
    constexpr static size_t	DOF = map_type::DOF;

    using matrix_type	= Eigen::Matrix<value_type, DOF, DOF>;
    using profiler_type	= Profiler<CLOCK>;

  public:
		ICIA(const Parameters& params=Parameters())
		    :profiler_type(2), _params(params), _source(),
		     _image(_source.image.cbegin(), _source.image.nrow()),
		     _edgeH(_source.edgeH.cbegin(), _source.edgeH.nrow()),
		     _edgeV(_source.edgeV.cbegin(), _source.edgeV.nrow()),
		     _M()						{}

    const Parameters&
		getParameters()		const	{ return _params; }
    void	setParameters(const Parameters& params)	{ _params = params; }
    bool	empty()			const	{ return _source.nrow() == 0; }
    const Frame&
		getSourceFrame()	const	{ return _source; }
    void	clearSourceFrame()		{ _source.clear(); }
    void	setSourceFrame(const Frame& source)			;
    void	setSourceFrame(Frame&& source)				;
    void	swapSourceFrame(Frame& source)				;
    void	setSourceImage(const image_type& image)			;
    void	setSourceImage(image_type&& image)			;
    void	swapSourceImage(image_type& image)			;
    void	setSourceWindow(size_t v0, size_t winSizeV,
				size_t u0, size_t winSizeH)		;
    Result	operator ()(const image_type& target, MAP& Mts)	const	;
    Result	operator ()(const image_type& source,
			    const image_type& target, MAP& Mts)		;

  private:
    void	computeEdges()						;
    
  private:
    Parameters	_params;
    Frame	_source;	// current reference source image
    slice_type	_image;
    slice_type	_edgeH;
    slice_type	_edgeV;
    matrix_type	_M;		// color moment matrix
};

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::setSourceFrame(const Frame& source)
{
    _source = source;

    setSourceWindow(0, _source.image.nrow(), 0, _source.image.ncol());
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::setSourceFrame(Frame&& source)
{
    _source = std::move(source);

    setSourceWindow(0, _source.image.nrow(), 0, _source.image.ncol());
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::swapSourceFrame(Frame& source)
{
    _source.swap(source);

    setSourceWindow(0, _source.image.nrow(), 0, _source.image.ncol());
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::setSourceImage(const image_type& image)
{
    _source.image = image;

    computeEdges();
    setSourceWindow(0, _source.image.nrow(), 0, _source.image.ncol());
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::setSourceImage(image_type&& image)
{
    _source.image = std::move(image);

    computeEdges();
    setSourceWindow(0, _source.image.nrow(), 0, _source.image.ncol());
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::swapSourceImage(image_type& image)
{
    _source.image.swap(image);

    computeEdges();
    setSourceWindow(0, _source.image.nrow(), 0, _source.image.ncol());
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::setSourceWindow(size_t v0, size_t winSizeV,
				     size_t u0, size_t winSizeH)
{
    using moment_type		= icia::ColorMoment<ICIA>;
    using moment_array_type	= typename moment_type::array_type;

    if (v0 + winSizeV > _source.image.nrow() ||
	u0 + winSizeH > _source.image.ncol())
	throw std::runtime_error("ICIA::setSourceWindow(): illegal window size["
				 + std::to_string(winSizeH) + 'x'
				 + std::to_string(winSizeV) + ']');

    _image = cu::slice(_source.image.cbegin(), v0, winSizeV, u0, winSizeH);
    _edgeH = cu::slice(_source.edgeH.cbegin(), v0, winSizeV, u0, winSizeH);
    _edgeV = cu::slice(_source.edgeV.cbegin(), v0, winSizeV, u0, winSizeH);
    
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

template <class MAP, class C, class CLOCK>
typename ICIA<MAP, C, CLOCK>::Result
ICIA<MAP, C, CLOCK>::operator ()(const image_type& target, MAP& Mts) const
{
    using deviation_type	= icia::ColorDeviation<ICIA>;
    using deviation_array_type	= typename deviation_type::array_type;

  // Convert the error moment to a matrix and save its diagonals.
    auto		Mts_old = Mts;
    auto		mse_old = std::numeric_limits<value_type>::max();
    size_t		overlap_old = 0;
    auto		mse_prev = mse_old;
    value_type		lambda  = 1.0e-3;
    for (size_t n = 0; n < _params.niter_max; ++n)
    {
      // Compute error derivation vector by parallel reduction.
	const deviation_type		deviation(Mts, _image, _edgeH, _edgeV,
						  target,
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
	const auto	mse	= deviation_type::mse(deviation_array);
	const auto	overlap = deviation.overlap(deviation_array);
#if !defined(NDEBUG)
	std::cerr << "      mse=" << mse << ", mse_old=" << mse_old
		  << ", mse_absdiff=" << std::abs(mse - mse_old)
		  << ", sqerr="   << deviation_type::sqerr(deviation_array)
		  << ", overlap=" << overlap
		  << std::endl;
#endif
	if (isnan(mse))
	    return {mse, overlap};

	if (mse < mse_old)
	{
	    if (std::abs(mse - mse_old) <= _params.tol || lambda < 1.0e-15)
	    {
		return {mse, overlap};
	    }

	    Mts_old	= Mts;
	    mse_old	= mse;
	    overlap_old = overlap;
	    lambda     *= 0.1;
	}
	else
	{
	    if (std::abs(mse - mse_prev) <= _params.tol || lambda < 1.0e-15)
	    {
		Mts = Mts_old;
		return {mse_old, overlap_old};
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
	TU::Image<C>	diff = TU::Array2<C>(_source.image)
			     - TU::Array2<C>(source);
	diff.saveData(std::cout, ImageFormat::FLOAT);
	usleep(50000);
#endif
    }

    throw std::runtime_error("ICIA::operator (): maximum iteration limit exceeded!");

    return {-1.0, 0};
}

template <class MAP, class C, class CLOCK>
typename ICIA<MAP, C, CLOCK>::Result
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
    const auto	result = (*this)(target, Mts);
    profiler_type::nextFrame();

    return result;
}

template <class MAP, class C, class CLOCK> void
ICIA<MAP, C, CLOCK>::computeEdges()
{
  // Compute horizontal and vertical image derivatives.
    _source.edgeH.resize(_source.image.nrow(), _source.image.ncol());
    _source.edgeV.resize(_source.image.nrow(), _source.image.ncol());
    FIRGaussianConvolver2<>	convolver(_params.sigma);
    convolver.diffH(_source.image.cbegin(), _source.image.cend(),
		    _source.edgeH.begin(), true);
    convolver.diffV(_source.image.cbegin(), _source.image.cend(),
		    _source.edgeV.begin(), true);
}
}	// namespace TU::cu
