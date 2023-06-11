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
  \file		vec.h
  \author	Toshio UESHIBA
  \brief	cudaベクトルクラスとその演算子の定義
*/
#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <thrust/device_ptr.h>
#include "TU/cu/array.h"
#include "TU/Image++.h"		// for TU::RGB_<E>

namespace TU
{
namespace cu
{
#if defined(__NVCC__)
namespace device
{
  template <class T>
  constexpr static T	minval  = std::numeric_limits<T>::min();
  template <class T>
  constexpr static T	maxval  = std::numeric_limits<T>::max();
  template <class T>
  constexpr static T	epsilon = std::numeric_limits<T>::epsilon();
  template <class T>
  constexpr static T	nan	= std::numeric_limits<T>::quiet_NaN();

  __device__ __forceinline__ float  fma(float x,
					float y,
					float z)	{ return fmaf(x, y, z);}
  __device__ __forceinline__ float  min(float x,
					float y)	{ return fminf(x, y); }
  __device__ __forceinline__ float  max(float x,
					float y)	{ return fmaxf(x, y); }
  __device__ __forceinline__ float  square(float x)	{ return x*x; }
  __device__ __forceinline__ float  abs(float x)	{ return fabsf(x); }
  __device__ __forceinline__ float  sqrt(float x)	{ return sqrtf(x); }
  __device__ __forceinline__ float  rsqrt(float x)	{ return rsqrtf(x); }
  __device__ __forceinline__ float  sin(float x)	{ return __sinf(x); }
  __device__ __forceinline__ float  cos(float x)	{ return __cosf(x); }
  __device__ __forceinline__ float  tan(float x)	{ return __tanf(x); }
  __device__ __forceinline__ float  atan2(float y,
					  float x)	{ return atan2f(y, x); }
  __device__ __forceinline__ float  floor(float x)	{ return floorf(x); }
  __device__ __forceinline__ float  ceil(float x)	{ return ceilf(x); }
  __device__ __forceinline__ float  exp(float x)	{ return __expf(x); }
  __device__ __forceinline__ float  exp10(float x)	{ return __exp10f(x); }
  __device__ __forceinline__ float  log(float x)	{ return __logf(x); }
  __device__ __forceinline__ float  log2(float x)	{ return __log2f(x); }
  __device__ __forceinline__ float  log10(float x)	{ return __log10f(x); }
  __device__ __forceinline__ float  pow(float x,
					float y)	{ return __powf(x, y); }

  __device__ __forceinline__ double min(double x,
					double y)	{ return fmin(x, y); }
  __device__ __forceinline__ double max(double x,
					double y)	{ return fmax(x, y); }
  __device__ __forceinline__ double square(double x)	{ return x*x; }
  __device__ __forceinline__ double abs(double x)	{ return fabs(x); }

  __device__ __forceinline__ int    to_int(float x)	{ return
							   __float2int_rn(x);}
  __device__ __forceinline__ int    to_int(double x)	{ return
							   __double2int_rn(x);}

  template <class T> __device__ __forceinline__ T
  clamp(T val, T low, T high)
  {
      using ::min;
      using ::max;

      return min(max(val, low), high);
  }
}	// namespace device
#endif	// __NVCC__

namespace detail
{
  template <class T, size_t N>	struct base_vec;

  template <>	struct base_vec<int8_t,   1>	{ using type = char1;	};
  template <>	struct base_vec<int8_t,   2>	{ using type = char2;	};
  template <>	struct base_vec<int8_t,   3>	{ using type = char3;	};
  template <>	struct base_vec<int8_t,   4>	{ using type = char4;	};

  template <>	struct base_vec<uint8_t,  1>	{ using type = uchar1;	};
  template <>	struct base_vec<uint8_t,  2>	{ using type = uchar2;	};
  template <>	struct base_vec<uint8_t,  3>	{ using type = uchar3;	};
  template <>	struct base_vec<uint8_t,  4>	{ using type = uchar4;	};

  template <>	struct base_vec<int16_t,  1>	{ using type = short1;	};
  template <>	struct base_vec<int16_t,  2>	{ using type = short2;	};
  template <>	struct base_vec<int16_t,  3>	{ using type = short3;	};
  template <>	struct base_vec<int16_t,  4>	{ using type = short4;	};

  template <>	struct base_vec<uint16_t, 1>	{ using type = ushort1;	};
  template <>	struct base_vec<uint16_t, 2>	{ using type = ushort2;	};
  template <>	struct base_vec<uint16_t, 3>	{ using type = ushort3;	};
  template <>	struct base_vec<uint16_t, 4>	{ using type = ushort4;	};

  template <>	struct base_vec<int32_t,  1>	{ using type = int1;	};
  template <>	struct base_vec<int32_t,  2>	{ using type = int2;	};
  template <>	struct base_vec<int32_t,  3>	{ using type = int3;	};
  template <>	struct base_vec<int32_t,  4>	{ using type = int4;	};

  template <>	struct base_vec<uint32_t, 1>	{ using type = uint1;	};
  template <>	struct base_vec<uint32_t, 2>	{ using type = uint2;	};
  template <>	struct base_vec<uint32_t, 3>	{ using type = uint3;	};
  template <>	struct base_vec<uint32_t, 4>	{ using type = uint4;	};

  template <>	struct base_vec<float,    1>	{ using type = float1;	};
  template <>	struct base_vec<float,    2>	{ using type = float2;	};
  template <>	struct base_vec<float,    3>	{ using type = float3;	};
  template <>	struct base_vec<float,    4>	{ using type = float4;	};

  template <>	struct base_vec<int64_t,  1>	{ using type = longlong1; };
  template <>	struct base_vec<int64_t,  2>	{ using type = longlong2; };

  template <>	struct base_vec<double,   1>	{ using type = double1;	};
  template <>	struct base_vec<double,   2>	{ using type = double2;	};
}	// namespace derail

/************************************************************************
*  struct mat2x<T, C>, mat3x<T, C>, mat4x<T, C>				*
************************************************************************/
template <class T, size_t C>	struct mat2x;

template <class T>
struct mat2x<T, 1> : public detail::base_vec<T, 2>::type
{
    using super		= typename detail::base_vec<T, 2>::type;

    using element_type	= T;
    using value_type	= element_type;

    using super::x;
    using super::y;

    __host__ __device__ constexpr
    static size_t	rank()				{ return 1; }
    __host__ __device__ constexpr
    static size_t	size0()				{ return 2; }
    __host__ __device__ constexpr
    static size_t	size1()				{ return 1; }
    __host__ __device__ constexpr
    static size_t	size()				{ return size0(); }

			mat2x()				= default;
    __host__ __device__ constexpr
    explicit		mat2x(T c)	:super{c, c}	{}
    __host__ __device__ constexpr
    explicit		mat2x(const super& v) :super{v}	{}
    __host__ __device__ constexpr
			mat2x(T x, T y)	:super{x, y}	{}

    __host__ __device__
    mat2x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    return *this;
			}

    __host__ __device__ constexpr
    static mat2x	zero()			{ return {T(0), T(0)}; }

    __device__ void	print() const
			{
			    constexpr auto fmt = (std::is_integral<T>::value ?
						  "[%d,%d]" : "[%f,%f]");
			    printf(fmt, x, y);
			}
};

template <class T, size_t C>	struct mat3x;

template <class T>
struct mat3x<T, 1> : public detail::base_vec<T, 3>::type
{
    using super		= typename detail::base_vec<T, 3>::type;

    using element_type	= T;
    using value_type	= element_type;

    using super::x;
    using super::y;
    using super::z;

    __host__ __device__ constexpr
    static size_t	rank()				{ return 1; }
    __host__ __device__ constexpr
    static size_t	size0()				{ return 3; }
    __host__ __device__ constexpr
    static size_t	size1()				{ return 1; }
    __host__ __device__ constexpr
    static size_t	size()				{ return size0(); }

			mat3x()				= default;
    __host__ __device__ constexpr
    explicit		mat3x(T c)
			    :super{c, c, c}		{}
    __host__ __device__ constexpr
    explicit		mat3x(const super& v)
			    :super{v}			{}
    __host__ __device__ constexpr
			mat3x(T x, T y, T z)
			    :super{x, y, z}		{}

    __host__ __device__
    mat3x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    z = c;
			    return *this;
			}

    __host__ __device__ constexpr
    static mat3x	zero()			{ return {T(0), T(0), T(0)}; }

    __device__ void	print() const
			{
			    constexpr auto fmt = (std::is_integral<T>::value ?
						  "[%d,%d,%d]" : "[%f,%f,%f]");
			    printf(fmt, x, y);
			}
};

template <class T, size_t C>	struct mat4x;

template <class T>
struct mat4x<T, 1> : public detail::base_vec<T, 4>::type
{
    using super		= typename detail::base_vec<T, 4>::type;

    using element_type	= T;
    using value_type	= element_type;

    using super::x;
    using super::y;
    using super::z;
    using super::w;

    __host__ __device__ constexpr
    static size_t	rank()				{ return 1; }
    __host__ __device__ constexpr
    static size_t	size0()				{ return 4; }
    __host__ __device__ constexpr
    static size_t	size1()				{ return 1; }
    __host__ __device__ constexpr
    static size_t	size()				{ return size0(); }

			mat4x()				= default;
    __host__ __device__ constexpr
    explicit		mat4x(T c)
			    :super{c, c, c, c}		{}
    __host__ __device__ constexpr
    explicit		mat4x(const super& v)
			    :super{v}			{}
    __host__ __device__ constexpr
			mat4x(T x, T y, T z, T w)
			    :super{x, y, z, w}		{}

    __host__ __device__
    mat4x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    z = c;
			    w = c;
			    return *this;
			}

    __host__ __device__ constexpr
    static mat4x	zero()		{ return {T(0), T(0), T(0), T(0)}; }

    __device__ void	print() const
			{
			    constexpr auto fmt = (std::is_integral<T>::value ?
						  "[%d,%d,%d,%d]" :
						  "[%f,%f,%f,%f]");
			    printf(fmt, x, y);
			}
};

template <class T, size_t D>
using vec = std::conditional_t<D == 1, T,
	    std::conditional_t<D == 2, mat2x<T, 1>,
	    std::conditional_t<D == 3, mat3x<T, 1>,
	    std::conditional_t<D == 4, mat4x<T, 1>, void> > > >;

template <class T, size_t C>
struct mat2x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

    __host__ __device__ constexpr
    static size_t	rank()				{ return 2; }
    __host__ __device__ constexpr
    static size_t	size0()				{ return 2; }
    __host__ __device__ constexpr
    static size_t	size1()				{ return C; }
    __host__ __device__ constexpr
    static size_t	size()				{ return size0(); }

			mat2x()				= default;
    __host__ __device__	constexpr
			mat2x(const value_type& xx,
			      const value_type& yy)
			    :x(xx), y(yy)		{}
    __host__ __device__	constexpr
    explicit		mat2x(T c) :x(c), y(c)		{}

    __host__ __device__
    mat2x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    return *this;
			}

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 2, mat2x<T, 2> >
    transpose()	const
    {
	return {{x.x, y.x}, {x.y, y.y}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 3, mat3x<T, 2> >
    transpose()	const
    {
	return {{x.x, y.x}, {x.y, y.y}, {x.z, y.z}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 4, mat4x<T, 2> >
    transpose()	const
    {
	return {{x.x, y.x}, {x.y, y.y}, {x.z, y.z}, {x.w, y.w}};
    }

    __host__ __device__ constexpr static mat2x
    zero()
    {
	return {value_type::zero(), value_type::zero()};
    }

    template <size_t C_=C> __host__ __device__
    constexpr static std::enable_if_t<C_ == 2, mat2x<T, 2> >
    identity()
    {
	return {{T(1), T(0)}, {T(0), T(1)}};
    }

    __device__ void	print() const
			{
			    printf("[");
			    x.print();
			    printf(",");
			    y.print();
			    printf("]");
			}

    value_type	x, y;
};

template <class T, size_t C>
struct mat3x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

    __host__ __device__ constexpr
    static size_t	rank()				{ return 2; }
    __host__ __device__ constexpr
    static size_t	size0()				{ return 3; }
    __host__ __device__ constexpr
    static size_t	size1()				{ return C; }
    __host__ __device__ constexpr
    static size_t	size()				{ return size0(); }

			mat3x()				= default;
    __host__ __device__	constexpr
			mat3x(const value_type& xx,
			      const value_type& yy,
			      const value_type& zz)
			    :x(xx), y(yy), z(zz)	{}
    __host__ __device__	constexpr
    explicit		mat3x(T c) :x(c), y(c), z(c)	{}

    __host__ __device__
    mat3x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    z = c;
			    return *this;
			}

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 2, mat2x<T, 3> >
    transpose()	const
    {
	return {{x.x, y.x, z.x}, {x.y, y.y, z.y}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 3, mat3x<T, 3> >
    transpose()	const
    {
	return {{x.x, y.x, z.x}, {x.y, y.y, z.y}, {x.z, y.z, z.z}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 4, mat4x<T, 3> >
    transpose()	const
    {
	return {{x.x, y.x, z.x}, {x.y, y.y, z.y},
		{x.z, y.z, z.z}, {x.w, y.w, z.w}};
    }

    __host__ __device__ constexpr static mat3x
    zero()
    {
	return {value_type::zero(), value_type::zero(), value_type::zero()};
    }

    template <size_t C_=C> __host__ __device__
    constexpr static std::enable_if_t<C_ == 3, mat3x<T, 3> >
    identity()
    {
	return {{T(1), T(0), T(0)}, {T(0), T(1), T(0)}, {T(0), T(0), T(1)}};
    }

    __device__ void	print() const
			{
			    printf("[");
			    x.print();
			    printf(",");
			    y.print();
			    printf(",");
			    z.print();
			    printf("]");
			}

    value_type	x, y, z;
};

template <class T, size_t C>
struct mat4x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

    __host__ __device__ constexpr
    static size_t	rank()				{ return 2; }
    __host__ __device__ constexpr
    static size_t	size0()				{ return 4; }
    __host__ __device__ constexpr
    static size_t	size1()				{ return C; }
    __host__ __device__ constexpr
    static size_t	size()				{ return size0(); }

			mat4x()				= default;
    __host__ __device__	constexpr
			mat4x(const value_type& xx,
			      const value_type& yy,
			      const value_type& zz,
			      const value_type& ww)
			    :x(xx), y(yy), z(zz), w(ww)	{}
    __host__ __device__	constexpr
    explicit		mat4x(T c)
			    :x(c), y(c), z(c), w(c)	{}

    __host__ __device__
    mat4x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    z = c;
			    w = c;
			    return *this;
			}

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 2, mat2x<T, 4> >
    transpose()	const
    {
	return {{x.x, y.x, z.x, w.x}, {x.y, y.y, z.y, w.y}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 3, mat3x<T, 4> >
    transpose()	const
    {
	return {{x.x, y.x, z.x, w.x},
		{x.y, y.y, z.y, w.y}, {x.z, y.z, z.z, w.z}};
    }

    template <size_t C_=C> __host__ __device__
    std::enable_if_t<C_ == 4, mat4x<T, 4> >
    transpose()	const
    {
	return {{x.x, y.x, z.x, w.x}, {x.y, y.y, z.y, w.y},
		{x.z, y.z, z.z, w.z}, {x.w, y.w, z.w, w.w}};
    }

    __host__ __device__ constexpr static mat4x
    zero()
    {
    	return {value_type::zero(), value_type::zero(),
    		value_type::zero(), value_type::zero()};
    }

    template <size_t C_=C> __host__ __device__
    constexpr static std::enable_if_t<C_ == 4, mat4x<T, 4> >
    identity()
    {
	return {{T(1), T(0), T(0), T(0)}, {T(0), T(1), T(0), T(0)},
		{T(0), T(0), T(1), T(0)}, {T(0), T(0), T(0), T(1)}};
    }

    __device__ void	print() const
			{
			    printf("[");
			    x.print();
			    printf(",");
			    y.print();
			    printf(",");
			    z.print();
			    printf(",");
			    w.print();
			    printf("]");
			}

    value_type	x, y, z, w;
};

template <class VM> constexpr size_t
size1()
{
    return VM::size1();
}

// TU::element_t<E> requires begin(std::decval<E>()) be valid.
// We have to override TU::cu::begin() defined in tuple.h,
template <class T, size_t C> const typename mat2x<T, C>::value_type*
begin(const mat2x<T, C>&)						;

template <class T, size_t C> const typename mat3x<T, C>::value_type*
begin(const mat3x<T, C>&)						;

template <class T, size_t C> const typename mat4x<T, C>::value_type*
begin(const mat4x<T, C>&)						;

template <class T, size_t R, size_t C>
using mat = std::conditional_t<R == 1, vec<T, C>,
	    std::conditional_t<R == 2, mat2x<T, C>,
	    std::conditional_t<R == 3, mat3x<T, C>,
	    std::conditional_t<R == 4, mat4x<T, C>, void> > > >;

/************************************************************************
*  Access element by its integral index					*
************************************************************************/
template <size_t I, class VM, std::enable_if_t<I == 0>* = nullptr>
__host__ __device__ __forceinline__ auto	get(const VM& a){ return a.x; }
template <size_t I, class VM, std::enable_if_t<I == 1>* = nullptr>
__host__ __device__ __forceinline__ auto	get(const VM& a){ return a.y; }
template <size_t I, class VM, std::enable_if_t<I == 2>* = nullptr>
__host__ __device__ __forceinline__ auto	get(const VM& a){ return a.z; }
template <size_t I, class VM, std::enable_if_t<I == 3>* = nullptr>
__host__ __device__ __forceinline__ auto	get(const VM& a){ return a.w; }

template <size_t I, class VM, std::enable_if_t<I == 0>* = nullptr>
__host__ __device__ __forceinline__ auto&	get(VM& a)	{ return a.x; }
template <size_t I, class VM, std::enable_if_t<I == 1>* = nullptr>
__host__ __device__ __forceinline__ auto&	get(VM& a)	{ return a.y; }
template <size_t I, class VM, std::enable_if_t<I == 2>* = nullptr>
__host__ __device__ __forceinline__ auto&	get(VM& a)	{ return a.z; }
template <size_t I, class VM, std::enable_if_t<I == 3>* = nullptr>
__host__ __device__ __forceinline__ auto&	get(VM& a)	{ return a.w; }

/************************************************************************
*  2-dimensional vectors or 2-by-C matrices				*
************************************************************************/
template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>
operator -(const mat2x<T, C>& a)
{
    return {-a.x, -a.y};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>&
operator +=(mat2x<T, C>& a, const mat2x<T, C>& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>&
operator -=(mat2x<T, C>& a, const mat2x<T, C>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>&
operator *=(mat2x<T, C>& a, T c)
{
    a.x *= c;
    a.y *= c;
    return a;
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>
operator +(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return {a.x + b.x, a.y + b.y};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>
operator -(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return {a.x - b.x, a.y - b.y};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>
operator *(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return {a.x * b.x, a.y * b.y};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>
operator /(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return {a.x / b.x, a.y / b.y};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>
operator *(const mat2x<T, C>& a, T c)
{
    return {a.x * c, a.y * c};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat2x<T, C>
operator /(T c, const mat2x<T, C>& a)
{
    return {c / a.x, c / a.y};
}

template <class T, size_t C> __host__ __device__ __forceinline__ bool
operator ==(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return a.x == b.x && a.y == b.y;
}

template <class T, size_t C> __host__ __device__ __forceinline__ bool
operator !=(const mat2x<T, C>& a, const mat2x<T, C>& b)
{
    return !(a == b);
}

/************************************************************************
*  3-dimensional vectors or 3-by-C matrices				*
************************************************************************/
template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>
operator -(const mat3x<T, C>& a)
{
    return {-a.x, -a.y, -a.z};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>&
operator +=(mat3x<T, C>& a, const mat3x<T, C>& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>&
operator -=(mat3x<T, C>& a, const mat3x<T, C>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>&
operator *=(mat3x<T, C>& a, T c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>
operator +(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>
operator -(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>
operator *(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>
operator /(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>
operator *(const mat3x<T, C>& a, T c)
{
    return {a.x * c, a.y * c, a.z * c};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat3x<T, C>
operator /(T c, const mat3x<T, C>& a)
{
    return {c / a.x, c / a.y, c / a.z};
}

template <class T, size_t C> __host__ __device__ __forceinline__ bool
operator ==(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

template <class T, size_t C> __host__ __device__ __forceinline__ bool
operator !=(const mat3x<T, C>& a, const mat3x<T, C>& b)
{
    return !(a == b);
}

/************************************************************************
*  4-dimensional vectors or 4-by-C matrices				*
************************************************************************/
template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>
operator -(const mat4x<T, C>& a)
{
    return {-a.x, -a.y, -a.z, -a.w};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>&
operator +=(mat4x<T, C>& a, const mat4x<T, C>& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>&
operator -=(mat4x<T, C>& a, const mat4x<T, C>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>&
operator *=(mat4x<T, C>& a, T c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    a.w *= c;
    return a;
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>
operator +(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>
operator -(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>
operator *(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>
operator /(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>
operator *(const mat4x<T, C>& a, T c)
{
    return {a.x * c, a.y * c, a.z * c, a.w * c};
}

template <class T, size_t C> __host__ __device__ __forceinline__ mat4x<T, C>
operator /(T c, const mat4x<T, C>& a)
{
    return {c / a.x, c / a.y, c / a.z, c / a.w};
}

template <class T, size_t C> __host__ __device__ __forceinline__ bool
operator ==(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <class T, size_t C> __host__ __device__ __forceinline__ bool
operator !=(const mat4x<T, C>& a, const mat4x<T, C>& b)
{
    return !(a == b);
}

/************************************************************************
*  Multiplication and division by scalar				*
************************************************************************/
template <class VM> __host__ __device__ __forceinline__
std::enable_if_t<(size0<VM>() > 1), VM&>
operator /=(VM& a, element_t<VM> c)
{
    return a *= (element_t<VM>(1)/c);
}

template <class VM> __host__ __device__ __forceinline__
std::enable_if_t<(size0<VM>() > 1), VM>
operator *(element_t<VM> c, const VM& a)
{
    return a * c;
}

template <class VM> __host__ __device__ __forceinline__
std::enable_if_t<(size0<VM>() > 1), VM>
operator /(const VM& a, element_t<VM> c)
{
    return a * (element_t<VM>(1)/c);
}

/************************************************************************
*  I/O functions							*
************************************************************************/
template <class T, size_t C> std::ostream&
operator <<(std::ostream& out, const mat2x<T, C>& a)
{
    return out << '[' << a.x << ',' << a.y << ']';
}

template <class T, size_t C> std::ostream&
operator <<(std::ostream& out, const mat3x<T, C>& a)
{
    return out << '[' << a.x << ',' << a.y << ',' << a.z << ']';
}

template <class T, size_t C> std::ostream&
operator <<(std::ostream& out, const mat4x<T, C>& a)
{
    return out << '[' << a.x << ',' << a.y << ',' << a.z << ',' << a.w << ']';
}

template <class T, size_t C> std::istream&
operator >>(std::istream& in, mat2x<T, C>& a)
{
    return in >> a.x >> a.y;
}

template <class T, size_t C> std::istream&
operator >>(std::istream& in, mat3x<T, C>& a)
{
    return in >> a.x >> a.y >> a.z;
}

template <class T, size_t C> std::istream&
operator >>(std::istream& in, mat4x<T, C>& a)
{
    return in >> a.x >> a.y >> a.z >> a.w;
}

/************************************************************************
*  homogeneous()							*
************************************************************************/
template <class T> __host__ __device__ __forceinline__ mat3x<T, 1>
homogeneous(const mat2x<T, 1>& a)
{
    return {a.x, a.y, T(1)};
}

template <class T> __host__ __device__ __forceinline__ mat4x<T, 1>
homogeneous(const mat3x<T, 1>& a)
{
    return {a.x, a.y, a.z, T(1)};
}

/************************************************************************
*  inhomogeneous()							*
************************************************************************/
template <class T> __host__ __device__ __forceinline__ mat2x<T, 1>
inhomogeneous(const mat3x<T, 1>& a)
{
    return {a.x / a.z, a.y / a.z};
}

template <class T> __host__ __device__ __forceinline__ mat3x<T, 1>
inhomogeneous(const mat4x<T, 1>& a)
{
    return {a.x / a.w, a.y / a.w, a.z / a.w};
}

/************************************************************************
*  dot()								*
************************************************************************/
template <class T, size_t C> __host__ __device__ __forceinline__ vec<T, C>
dot(const mat2x<T, 1>& a, const mat2x<T, C>& b)
{
    return a.x * b.x + a.y * b.y;
}

template <class T, size_t C> __host__ __device__ __forceinline__ vec<T, C>
dot(const mat3x<T, 1>& a, const mat3x<T, C>& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <class T, size_t C> __host__ __device__ __forceinline__ vec<T, C>
dot(const mat4x<T, 1>& a, const mat4x<T, C>& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <class T, size_t C, class VM> __host__ __device__ __forceinline__
std::enable_if_t<size0<VM>() == C, mat<T, 2, size1<VM>()> >
dot(const mat2x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a)};
}

template <class T, size_t C, class VM> __host__ __device__ __forceinline__
std::enable_if_t<size0<VM>() == C, mat<T, 3, size1<VM>()> >
dot(const mat3x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a), dot(m.z, a)};
}

template <class T, size_t C, class VM> __host__ __device__ __forceinline__
std::enable_if_t<size0<VM>() == C, mat<T, 4, size1<VM>()> >
dot(const mat4x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a), dot(m.z, a), dot(m.w, a)};
}

/************************************************************************
*  square()								*
************************************************************************/
template <class VEC> __host__ __device__ __forceinline__
std::enable_if_t<size1<VEC>() == 1, element_t<VEC> >
square(const VEC& a)
{
    return dot(a, a);
}

/************************************************************************
*  norm()								*
************************************************************************/
template <class VEC> __host__ __device__ __forceinline__ element_t<VEC>
norm(const VEC& a)
{
    using	device::square;

    return sqrt(square(a));
}

/************************************************************************
*  normalized()								*
************************************************************************/
template <class VEC> __host__ __device__ __forceinline__ VEC
normalized(const VEC& a)
{
    return a / norm(a);
}

/************************************************************************
*  cross()								*
************************************************************************/
template <class T> __host__ __device__ __forceinline__ mat3x<T, 1>
cross(const mat3x<T, 1>& a, const mat3x<T, 1>& b)
{
    return {a.y * b.z - a.z * b.y,
	    a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/************************************************************************
*  ext()								*
************************************************************************/
template <class T, class VEC> __host__ __device__ __forceinline__
std::enable_if_t<size1<VEC>() == 1, mat2x<T, size0<VEC>()> >
ext(const mat2x<T, 1>& a, const VEC& b)
{
    return {a.x * b, a.y * b};
}

template <class T, class VEC> __host__ __device__ __forceinline__
std::enable_if_t<size1<VEC>() == 1, mat3x<T, size0<VEC>()> >
ext(const mat3x<T, 1>& a, const VEC& b)
{
    return {a.x * b, a.y * b, a.z * b};
}

template <class T, class VEC> __host__ __device__ __forceinline__
std::enable_if_t<size1<VEC>() == 1, mat4x<T, size0<VEC>()> >
ext(const mat4x<T, 1>& a, const VEC& b)
{
    return {a.x * b, a.y * b, a.z * b, a.w * b};
}

/************************************************************************
*  rotation()								*
************************************************************************/
template <class T> __host__ __device__ __forceinline__ mat2x<T, 2>
rotation(T theta)
{
    return {{cos(theta), -sin(theta)}, {sin(theta), cos(theta)}};
}

template <class T> __host__ __device__ __forceinline__ mat3x<T, 3>
rotation(const mat3x<T, 1>& axis)
{
    const auto	theta  = sqrt(square(axis));
    const auto	normal = (theta > 0 ? axis / theta : mat3x<T, 1>{1, 0, 0});
    const auto	c = cos(theta);
    const auto	s = sin(theta);
    auto	R = ext(normal, normal)*(T(1) - c);
    R.x.x += c;
    R.y.y += c;
    R.z.z += c;
    R.x.y -= normal.z * s;
    R.x.z += normal.y * s;
    R.y.x += normal.z * s;
    R.y.z -= normal.x * s;
    R.z.x -= normal.y * s;
    R.z.y += normal.x * s;

    return R;
}

namespace device
{
/************************************************************************
*  atomic operations							*
************************************************************************/
template <class T> __device__ __forceinline__
std::enable_if_t<std::is_arithmetic<T>::value, T>
atomicOp(T* p, T val, std::nullptr_t)
{
    return ::atomicExch(p, val);
}

template <class T> __device__ __forceinline__
std::enable_if_t<std::is_arithmetic<T>::value, T>
atomicOp(T* p, T val, std::plus<>)
{
    return ::atomicAdd(p, val);
}

template <class T> __device__ __forceinline__
std::enable_if_t<std::is_arithmetic<T>::value, T>
atomicOp(T* p, T val, std::minus<>)
{
    return ::atomicSub(p, val);
}

template <class T> __device__ __forceinline__
std::enable_if_t<std::is_integral<T>::value, T>
atomicOp(T* p, T val, std::bit_and<>)
{
    return ::atomicAnd(p, val);
}

template <class T> __device__ __forceinline__
std::enable_if_t<std::is_integral<T>::value, T>
atomicOp(T* p, T val, std::bit_or<>)
{
    return ::atomicOr(p, val);
}

template <class T> __device__ __forceinline__
std::enable_if_t<std::is_integral<T>::value, T>
atomicOp(T* p, T val, std::bit_xor<>)
{
    return ::atomicXor(p, val);
}

template <class T, size_t C, class OP> __device__ __forceinline__ mat2x<T, C>
atomicOp(mat2x<T, C>* p, const mat2x<T, C>& val, OP op)
{
    return {atomicOp(&p->x, val.x, op), atomicOp(&p->y, val.y, op)};
}

template <class T, size_t C, class OP> __device__ __forceinline__ mat3x<T, C>
atomicOp(mat3x<T, C>* p, const mat3x<T, C>& val, OP op)
{
    return {atomicOp(&p->x, val.x, op), atomicOp(&p->y, val.y, op),
	    atomicOp(&p->z, val.z, op)};
}

template <class T, size_t C, class OP> __device__ __forceinline__ mat4x<T, C>
atomicOp(mat4x<T, C>* p, const mat4x<T, C>& val, OP op)
{
    return {atomicOp(&p->x, val.x, op), atomicOp(&p->y, val.y, op),
	    atomicOp(&p->z, val.z, op), atomicOp(&p->w, val.w, op)};
}

}	// namespace device

/************************************************************************
*  class Projectivity<T, DO, DI>					*
************************************************************************/
template <class T, size_t DO, size_t DI>
class Projectivity
{
  public:
    constexpr static size_t	DO1 = DO + 1;
    constexpr static size_t	DI1 = DI + 1;
    constexpr static size_t	DOF = DO1 * DI1 - 1;

    using element_type	= T;
    using matrix_type	= mat<element_type, DO1, DI1>;
    using point_type	= vec<element_type, DO>;
    using ppoint_type	= vec<element_type, DO1>;
    using param_type	= array<element_type, DOF>;

    constexpr static size_t	inDim()		{ return DI; }
    constexpr static size_t	outDim()	{ return DO; }
    constexpr static size_t	dof()		{ return DOF; }

		Projectivity()					= default;
    __host__ __device__
    explicit	Projectivity(const matrix_type& m)	:_m(m)	{}

    __host__ __device__
    void	initialize(const matrix_type& m=matrix_type::identity())
		{
		    _m = m;
		}

    __host__ __device__
    point_type	operator ()(const vec<T, DI>& p) const
		{
		    return inhomogeneous(mapP(p));
		}
    __host__ __device__
    point_type	operator ()(const vec<T, DI1>& p) const
		{
		    return inhomogeneous(mapP(p));
		}
    __host__ __device__
    point_type	operator ()(T u, T v) const
    		{
    		    return inhomogeneous(mapP(u, v));
    		}

    __host__ __device__
    ppoint_type	mapP(const vec<T, DI>& p) const
		{
		    return mapP(homogeneous(p));
		}
    __host__ __device__
    ppoint_type	mapP(const vec<T, DI1>& p) const
		{
		    return dot(_m, p);
		}
    __host__ __device__
    ppoint_type	mapP(T u, T v) const
    		{
    		    return dot(_m, vec<T, 3>{u, v, T(1)});
    		}

    template <size_t DO_=DO, size_t DI_=DI> __host__ __device__
    std::enable_if_t<DO_ == DI_, Projectivity>
		operator *(const Projectivity& projectivity) const
		{
		    return Projectivity(dot(_m, projectivity._m));
		}

    template <class ITER, size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 2 && DI_ == 2, Projectivity>
		exp(ITER delta)
		{
		    Projectivity	projectivity;
		    projectivity._m.x.x = *delta + 1;
		    projectivity._m.x.y = *++delta;
		    projectivity._m.x.z  = *++delta;
		    projectivity._m.y.x = *++delta;
		    projectivity._m.y.y = *++delta + 1;
		    projectivity._m.y.z = *++delta;
		    projectivity._m.z.x = *++delta;
		    projectivity._m.z.y = *++delta;
		    projectivity._m.z.z = 1;

		    return projectivity;
		}

    template <class ITER, size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 3 && DI_ == 3, Projectivity>
		exp(ITER delta)
		{
		    Projectivity	projectivity;
		    projectivity._m.x.x = *delta + 1;
		    projectivity._m.x.y = *++delta;
		    projectivity._m.x.z = *++delta;
		    projectivity._m.x.w = *++delta;
		    projectivity._m.y.x = *++delta;
		    projectivity._m.y.y = *++delta + 1;
		    projectivity._m.y.z = *++delta;
		    projectivity._m.y.w = *++delta;
		    projectivity._m.z.x = *++delta;
		    projectivity._m.z.y = *++delta;
		    projectivity._m.z.z = *++delta + 1;
		    projectivity._m.z.w = *++delta;
		    projectivity._m.w.x = *++delta;
		    projectivity._m.w.y = *++delta;
		    projectivity._m.w.z = *++delta;
		    projectivity._m.w.w = 1;

		    return projectivity;
		}

    template <size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 2 && DI_ == 2, param_type>
		image_derivative0(T u, T v, T eH, T eV)
		{
		    return {eH*u, eH*v, eH, eV*u, eV*v, eV,
			    -(eH*u + eV*v)*u, -(eH*u + eV*v)*v};
		}

    template <class ITER, size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 2 && DI_ == 2, void>
		unnormalize_updates(ITER delta, T scale)
		{
		    *delta   *= scale;
		    *++delta *= scale;
		    ++delta;
		    *++delta *= scale;
		    *++delta *= scale;
		    ++delta;
		    *++delta *= (scale * scale);
		    *++delta *= (scale * scale);
		}

    friend std::ostream&
		operator <<(std::ostream& out, const Projectivity& proj)
		{
		    return out << proj._m;
		}

    __device__
    void	print() const
		{
		    _m.print();
		}

  private:
    matrix_type	_m;
};

/************************************************************************
*  class Affinity<T, DO, DI>						*
************************************************************************/
template <class T, size_t DO, size_t DI>
class Affinity
{
  public:
    constexpr static size_t	DO1 = DO + 1;
    constexpr static size_t	DI1 = DI + 1;
    constexpr static size_t	DOF = DO * DI1;

    using element_type	= T;
    using matrix_type	= mat<element_type, DO, DI>;
    using point_type	= vec<element_type, DO>;
    using param_type	= array<element_type, DOF>;

    constexpr static size_t	inDim()		{ return DI; }
    constexpr static size_t	outDim()	{ return DO; }
    constexpr static size_t	dof()		{ return DOF; }

		Affinity()			= default;
    __host__ __device__
		Affinity(const matrix_type& A, const point_type& b)
		    :_A(A), _b(b)
		{
		}

    __host__ __device__
    void	initialize(const matrix_type& A=matrix_type::identity(),
			   const point_type&  b=point_type::zero())
		{
		    _A = A;
		    _b = b;
		}

    __host__ __device__
    const auto&	A()			const	{ return _A; }
    __host__ __device__
    const auto&	b()			const	{ return _b; }

    __host__ __device__
    point_type	operator ()(const vec<T, DI>& p) const
		{
		    return dot(_A, p) + _b;
		}
    template <size_t DO_=DO, size_t DI_=DI> __host__ __device__
    std::enable_if_t<DO_ == 2 && DI_ == 2, point_type>
		operator ()(T u, T v) const
		{
		    return (*this)(vec<T, 2>({u, v}));
		}

    template <size_t DO_=DO, size_t DI_=DI> __host__ __device__
    std::enable_if_t<DO_ == DI_, Affinity>
		operator *(const Affinity& affinity) const
		{
		    return {dot(_A, affinity._A), _b + dot(_A, affinity._b)};
		}

    template <class ITER, size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 2 && DI_ == 2, Affinity>
		exp(ITER delta)
		{
		    Affinity	affinity;
		    affinity._A.x.x = *delta + 1;
		    affinity._A.x.y = *++delta;
		    affinity._b.x   = *++delta;
		    affinity._A.y.x = *++delta;
		    affinity._A.y.y = *++delta + 1;
		    affinity._b.y   = *++delta;

		    return affinity;
		}

    template <class ITER, size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 3 && DI_ == 3, Affinity>
		exp(ITER delta)
		{
		    Affinity	affinity;
		    affinity._A.x.x = *delta + 1;
		    affinity._A.x.y = *++delta;
		    affinity._A.x.z = *++delta;
		    affinity._b.x   = *++delta;
		    affinity._A.y.x = *++delta;
		    affinity._A.y.y = *++delta + 1;
		    affinity._A.y.z = *++delta;
		    affinity._b.y   = *++delta;
		    affinity._A.z.x = *++delta;
		    affinity._A.z.y = *++delta;
		    affinity._A.z.z = *++delta + 1;
		    affinity._b.y   = *++delta;

		    return affinity;
		}

    template <size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 2 && DI_ == 2, param_type>
		image_derivative0(T u, T v, T eH, T eV)
		{
		    return {eH*u, eH*v, eH,  eV*u, eV*v, eV};
		}

    template <class ITER, size_t DO_=DO, size_t DI_=DI> __host__ __device__
    static std::enable_if_t<DO_ == 2 && DI_ == 2, void>
		unnormalize_updates(ITER delta, T scale)
		{
		    *delta   *= scale;
		    *++delta *= scale;
		    ++delta;
		    *++delta *= scale;
		    *++delta *= scale;
		}

    friend std::ostream&
		operator <<(std::ostream& out, const Affinity& affinity)
		{
		    return out << affinity._A << ',' << affinity._b;
		}

    __device__
    void	print() const
		{
		    _A.print();
		    printf(",");
		    _b.print();
		}

  protected:
    matrix_type	_A;
    point_type	_b;
};

/************************************************************************
*  class Rigidity<T, D>							*
************************************************************************/
template <class T, size_t D>
class Rigidity : public Affinity<T, D, D>
{
  public:
    constexpr static size_t	DOF = D*(D+1)/2;

    using base_type		= Affinity<T, D, D>;
    using			typename base_type::element_type;
    using			typename base_type::matrix_type;
    using			typename base_type::point_type;
    using direction_type	= vec<element_type, D>;
    using param_type		= array<element_type, DOF>;

    constexpr static size_t	dof()		{ return DOF; }
    constexpr static size_t	dim()		{ return D; }

  public:
		Rigidity()			= default;
    __host__ __device__
		Rigidity(const matrix_type& R, const point_type& t)
		    :base_type(R, t)
		{
		}

    __host__ __device__
    const auto&	R()			const	{ return base_type::A(); }
    __host__ __device__
    const auto&	t()			const	{ return base_type::b(); }
    __host__ __device__
    Rigidity	inv() const
		{
		    return {R().transpose(), -dot(t(), R())};
		}

    using	base_type::operator ();

    __host__ __device__
    direction_type
		direction(const direction_type& d) const
		{
		    return dot(R(), d);
		}
    __host__ __device__
    point_type	invert(const point_type& p) const
		{
		    return dot(p - t(), R());
		}
    __host__ __device__
    direction_type
		invert_direction(const direction_type& d) const
		{
		    return dot(d, R());
		}

    __host__ __device__
    Rigidity	operator *(const Rigidity& rigidity) const
		{
		    return {dot(R(), rigidity.R()),
			    dot(R(), rigidity.t()) + t()};
		}

    template <class ITER, size_t D_=D> __host__ __device__
    static std::enable_if_t<D_ == 2, Rigidity>
		exp(ITER delta)
		{
		    point_type	dt;
		    dt.x = *delta;
		    dt.y = *++delta;
		    const element_type	dtheta = *++delta;

		    return {rotation(dtheta), dt};
		}

    template <class ITER, size_t D_=D> __host__ __device__
    static std::enable_if_t<D_ == 3, Rigidity>
		exp(ITER delta)
		{
		    point_type	dt;
		    dt.x = *delta;
		    dt.y = *++delta;
		    dt.z = *++delta;
		    point_type	dtheta;
		    dtheta.x = *++delta;
		    dtheta.y = *++delta;
		    dtheta.z = *++delta;

		    return {rotation(dtheta), dt};
		}

    template <size_t D_=D> __host__ __device__
    static std::enable_if_t<D_ == 2, param_type>
		image_derivative0(T u, T v, T eH, T eV)
		{
		    return {eH, eV, eV*u - eH*v};
		}

    template <class ITER, size_t D_=D> __host__ __device__
    static std::enable_if_t<D_ == 2, void>
		unnormalize_updates(ITER delta, T scale)
		{
		    ++delta;
		    *++delta *= scale;
		}

    using	base_type::initialize;
    using	base_type::print;
};

/************************************************************************
*  class Intrinsics<T>							*
************************************************************************/
template <class T>
class Intrinsics
{
  public:
    using element_type		= T;
    using point2_type		= vec<element_type, 2>;
    using point3_type		= vec<element_type, 3>;
    using derivative_type	= vec<element_type, 3>;

  public:
    Intrinsics()						= default;
    template <class ITER_K, class ITER_D> __host__ __device__
    Intrinsics(ITER_K K, ITER_D d, ITER_D de, element_type scale=1)
    {
	initialize(K, d, de, scale);
    }

    template <class ITER_K, class ITER_D> __host__ __device__ void
    initialize(ITER_K K, ITER_D d, ITER_D de, element_type scale=1)
    {
	_flen.x = scale * *K;
	std::advance(K, 2);
	_uv0.x = scale * *K;
	std::advance(K, 2);
	_flen.y = scale * *K;
	++K;
	_uv0.y = scale * *K;
	for (auto&& distortion : _d)
	    distortion = (d != de ? *d++ : 0);
    }

  //! 画素座標における2D点を正規化画像座標における2D点に変換
    __host__ __device__ point2_type
    operator ()(element_type u, element_type v) const
    {
	constexpr static element_type	MAX_ERR  = 0.001*0.001;
	constexpr static int		MAX_ITER = 5;

	const point2_type	uv{u, v};
	auto			xy  = (uv - _uv0)/_flen;
	const auto		xy0 = xy;

      // compensate distortion iteratively
	for (int n = 0; n < MAX_ITER; ++n)
	{
	    const auto	r2 = cu::square(xy);
	    const auto	k  = element_type(1) + (_d[0] + _d[1]*r2)*r2;
	    if (k < element_type(0))
		break;

	    const auto		 a = element_type(2)*xy.x*xy.y;
	    vec<element_type, 2> delta{_d[2]*a + _d[3]*(r2 + 2*xy.x*xy.x),
				       _d[2]*(r2 + 2*xy.y*xy.y) + _d[3]*a};
	    const auto	uv_proj = _flen*(k*xy + delta) + _uv0;

	    if (cu::square(uv_proj - uv) < MAX_ERR)
		break;

	    xy = (xy0 - delta)/k;	// compensate lens distortion
	}

	return xy;
    }

  //! 画素座標における2D点を正規化画像座標における2D点に変換
    __host__ __device__ point2_type
    operator ()(const point2_type& uv) const
    {
	return (*this)(uv.x, uv.y);
    }

  //! depth画像における画素座標とdepthからカメラ座標における3D点を計算
    __host__ __device__ point3_type
    operator ()(element_type u, element_type v, element_type d) const
    {
	const auto	xy = (*this)(u, v);
	return {d*xy.x, d*xy.y, d};
    }

  //! カメラ座標における3D点からそれが投影される画像点の画素座標を計算
    __host__ __device__ point2_type
    operator ()(const point3_type& p) const
    {
	const point2_type	xy(p.x/p.z, p.y/p.z);
	const auto		r2 = cu::square(xy);
	const auto		k  = 1 + (_d[0] + _d[1]*r2)*r2;
	const auto		a  = element_type(2)*xy.x*xy.y;
	vec<element_type, 2>	delta{_d[2]*a + _d[3]*(r2 + 2*xy.x*xy.x),
				      _d[2]*(r2 + 2*xy.y*xy.y) + _d[3]*a};

	return _flen*(k*xy + delta) + _uv0;
    }

    __host__ __device__ derivative_type
    image_derivative0(const point3_type& p,
		      element_type eH, element_type eV) const
    {
	point2_type		xy(p.x/p.z, p.y/p.z);
	const auto		r2 = cu::square(xy);
	const auto		k  = 1 + (_d[0] + _d[1]*r2)*r2;
	const auto		a  = element_type(2)*xy.x*xy.y;
	vec<element_type, 2>	delta{_d[2]*a + _d[3]*(r2 + 2*xy.x*xy.x),
				      _d[2]*(r2 + 2*xy.y*xy.y) + _d[3]*a};
	xy = _flen*(k*xy + delta);

	return {eH/p.z, eV/p.z, -(eH*xy.x + eV*xy.y)/p.z};
    }

  private:
    vec<element_type, 2>	_flen;
    point2_type			_uv0;
    element_type		_d[4];
};

/************************************************************************
*  class Camera<T>							*
************************************************************************/
template <class T>
class Camera
{
  public:
    using rigidity_type		= Rigidity<T, 3>;
    using intrinsics_type	= Intrinsics<T>;
    using element_type		= typename intrinsics_type::element_type;
    using point2_type		= typename intrinsics_type::point2_type;
    using point3_type		= typename intrinsics_type::point3_type;
    using direction_type	= typename rigidity_type::direction_type;

  public:
    __host__ __device__		Camera(const rigidity_type& rigidity,
				       const intrinsics_type& intrinsics)
				    :_rigidity(rigidity),
				     _intrinsics(intrinsics)
				{}

    __host__ __device__
    const rigidity_type&	rigidity()	const	{ return _rigidity; }
    __host__ __device__
    const intrinsics_type&	intrinsics()	const	{ return _intrinsics; }
    __host__ __device__
    point2_type			operator ()(const point3_type& p) const
				{
				    return _intrinsics(_rigidity(p));
				}

  private:
    rigidity_type	_rigidity;
    intrinsics_type	_intrinsics;
};

/************************************************************************
*  class Triange<T>							*
************************************************************************/
template <class T>
class Triangle
{
  public:
    using element_type	= T;
    using vector_type	= vec<element_type, 3>;

  public:
    __host__ __device__ __forceinline__
		Triangle()						{}
    __host__ __device__ __forceinline__
		Triangle(const vector_type& v0,
			 const vector_type& v1,
			 const vector_type& v2)	:_v{v0, v1, v2}		{}

    __host__ __device__ __forceinline__
    vector_type normal() const
		{
		    return normalized(cross(_v.y - _v.x, _v.z - _v.x));
		}
    __host__ __device__ __forceinline__
    vector_type	foot(const vector_type& p) const
		{
		    return dot(barycentric_coord(p), _v);
		}
    __host__ __device__ __forceinline__
    element_type
		sqdist(const vector_type& p) const
		{
		    return square(p - foot(p));
		}
    __host__ __device__ __forceinline__
    vector_type	closest_point(const vector_type& p) const
		{
		    const auto	coord = barycentric_coord(p);

		    if (coord.x > 1)
			return _v.x;
		    else if (coord.y > 1)
			return _v.y;
		    else if (coord.z > 1)
			return _v.z;
		    else if (coord.x < 0)
			return (coord.y*_v.y + coord.z*_v.z) / (1 - coord.x);
		    else if (coord.y < 0)
			return (coord.z*_v.z + coord.x*_v.x) / (1 - coord.y);
		    else if (coord.z < 0)
			return (coord.x*_v.x + coord.y*_v.y) / (1 - coord.z);
		    else
			return dot(coord, _v);
		}
    __host__ __device__ __forceinline__
    vector_type	barycentric_coord(const vector_type& p) const
		{
		    const auto	b  = _v.y - _v.x;
		    const auto	c  = _v.z - _v.x;
		    const auto	bc = cross(b, c);
		    const auto	x  = p - _v.x;
		    const auto	k  = element_type(1) / square(bc);
		    vector_type	coord;
		    coord.y = dot(bc, cross(x, c)) * k;
		    coord.z = dot(bc, cross(b, x)) * k;
		    coord.x = element_type(1) - coord.y - coord.z;

		    return coord;
		}

    std::istream&
		read(std::istream& in)
		{
		    float	buf[9];
		    if (!in.read((char*)buf, sizeof(buf)))
			throw
			    std::runtime_error("Triangle<T>::reead() failed!");

		    _v.x.x = buf[0];
		    _v.x.y = buf[1];
		    _v.x.z = buf[2];
		    _v.y.x = buf[3];
		    _v.y.y = buf[4];
		    _v.y.z = buf[5];
		    _v.z.x = buf[6];
		    _v.z.y = buf[7];
		    _v.z.z = buf[8];

		    return in;
		}

    friend std::istream&
		operator >>(std::istream& in, Triangle<T>& triangle)
		{
		    return in >> triangle._v;
		}

    friend std::ostream&
		operator <<(std::ostream& out, const Triangle<T>& triangle)
		{
		    return out << triangle._v;
		}

  private:
    mat3x<T, 3>	_v;
};

//! 入力ストリームからSTL形式のメッシュを読み込む．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T> std::istream&
restoreSTL(std::istream& in, std::vector<Triangle<T> >& faces)
{
  // 先頭の5 byteを読む．
    char	magic[6];
    in.read(magic, 5);
    magic[5] = '\0';

    if (std::string(magic) != "solid")
    {
      // ヘッダ(80文字)の残りを読み捨てる.
	char	header[80 - 5];
	in.read(header, sizeof(header));

      // 面数を読み込む.
	uint32_t	nfaces;
	in.read((char*)&nfaces, sizeof(nfaces));
	faces.resize(nfaces);

	for (auto& face : faces)
	{
	    float	buf[3];
	    in.read((char*)buf, sizeof(buf));	// 法線ベクトルを読み捨てる
	    face.read(in);			// 頂点の3D座標を読み込む

	    uint16_t	flags;
	    in.read((char*)&flags, sizeof(flags));	// フラグを読み込む
	}
    }
    else
    {
	for (std::string s; in >> s && s != "endsolid"; )
	{
	    if (s != "facet")
		continue;

	    float	buf;
	    Triangle<T>	triangle;
	    in >> s >> buf >> buf >> buf	// "normal" nx ny nz
	       >> s >> s			// "outer" "loop"
	       >> s >> triangle			// "vertex" x y z
	       >> s >> s;			// "endloop" "endfacet"
	    if (!in)
		throw std::runtime_error("restoreSTL() failed.");

	    faces.push_back(triangle);
	}
    }

    return in;
}

/************************************************************************
*  operators for fitting plane to 3D points 				*
************************************************************************/
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
      Qt = mat3x<T, 3>::identity();

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
	const auto	x = get<I+1>(e);
	const auto	y = s*get<I>(e);
	const auto	z = c*get<I>(e);
	if (abs(x) > abs(y))
	{
	    const auto	t = y/x;
	    const auto	r = sqrt(square(t) + T(1));
	    get<I+1>(e) = x*r;
	    c 	      = T(1)/r;
	    s	      = c*t;
	}
	else
	{
	    const auto	t = x/y;
	    const auto	r = sqrt(square(t) + T(1));
	    get<I+1>(e) = y*r;
	    s	      = T(1)/r;
	    c	      = s*t;
	}

	const auto	v = s*(get<I>(w) - get<I+1>(w)) + T(2)*c*z;
	const auto	p = s*v;
	get<I  >(w) -= p;
	get<I+1>(w) += p;
	get<I  >(e)  = c*v - z;

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

    // Enforce positive determinant value
      if (dot(Qt.x, cross(Qt.y, Qt.z)) < 0)
	  Qt.x *= T(-1);

      return true;
  }

  template <class T> __device__  __forceinline__ bool
  eigen33(const mat3x<T, 3>& A, mat3x<T, 3>& Qt, vec<T, 3>& w)
  {
      w = cardano(A);		// Calculate eigenvalues

    // t: minimum absolute eigenvalue
      const auto	t = min(min(abs(w.x), abs(w.y)), abs(w.z));
      const auto	u = (t < T(1) ? t : square(t));
    // error should be positive even if u == 0
      const auto	error = T(256)*epsilon<T>*square(u)
			      + T(2)  *epsilon<T>;

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

/************************************************************************
*  class Moment<T>							*
************************************************************************/
template <class T>
class Moment : public mat4x<T, 3>
{
  /*
   * [[  x,   y,   z],
   *  [x*x, x*y, x*z],
   *  [y*y, y*z, z*z],
   *  [  u,   v,   w]]		# w = (z > 0 ? 1 : 0)
   */
  public:
    using super		= mat4x<T, 3>;
    using element_type	= typename super::element_type;
    using value_type	= typename super::value_type;
    using vector_type	= vec<element_type, 3>;
    using matrix_type	= mat3x<element_type, 3>;
    using transform_type= Rigidity<element_type, 3>;

  public:
			Moment()				= default;
    __host__ __device__ __forceinline__
    explicit		Moment(element_type c)	:super(c)	{}
    __host__ __device__ __forceinline__
    explicit		Moment(const super& m)	:super(m)	{}
    __host__ __device__ __forceinline__
			Moment(const vector_type& p, int u=0, int v=0)
			    :super(isnan(p.z) ?
				   super{0} :
				   super{p, p.x*p, {p.y*p.y, p.y*p.z, p.z*p.z},
					 {u, v, 1}})
			{
			}

    __host__ __device__ __forceinline__
    Moment&		operator +=(const Moment& m)
			{
			    static_cast<super&>(*this) += m;
			    return *this;
			}

    __host__ __device__ __forceinline__
    Moment&		operator -=(const Moment& m)
			{
			    static_cast<super&>(*this) -= m;
			    return *this;
			}

    using		super::operator =;
    using		super::x;
    using		super::y;
    using		super::z;
    using		super::w;

    __host__ __device__ __forceinline__
    static Moment	invalid_moment()		{ return super{0}; }
    __host__ __device__ __forceinline__
    bool		is_valid()		const	{ return w.z != T(0); }
    __host__ __device__ __forceinline__
    element_type	u()			const	{ return w.x; }
    __host__ __device__ __forceinline__
    element_type	v()			const	{ return w.y; }
    __host__ __device__ __forceinline__
    element_type	npoints()		const	{ return w.z; }
    __host__ __device__ __forceinline__
    vector_type		mean()			const	{ return x / w.z; }
    __host__ __device__ __forceinline__
    matrix_type		covariance() const
			{
			    const auto	m = mean();

			    return {y - x.x*m,
				    {T(0), z.x - x.y*m.y, z.y - x.y*m.z},
				    {T(0), T(0),	  z.z - x.z*m.z}};
			}

    __device__ __forceinline__
    matrix_type		PCA(vector_type& evals) const
			{
			    const auto	C = covariance();
			    matrix_type	evecs;
			    device::eigen33(C, evecs, evals);

			    if (evals.x < evals.y)
			    {
			    	swap(evals.x, evals.y);
			    	swap(evecs.x, evecs.y);
			    }

			    if (evals.x < evals.z)
			    {
			    	swap(evals.x, evals.z);
			    	swap(evecs.x, evecs.z);
			    }

			    if (evals.y < evals.z)
			    {
			    	swap(evals.y, evals.z);
			    	swap(evecs.y, evecs.z);
			    }

			    return evecs;
			}

    __device__ __forceinline__
    transform_type	get_transform() const
			{
			    vector_type	evals;
			    return {PCA(evals).transpose(), mean()};
			}

  private:
    __device__ __forceinline__
    static void		swap(T& a, T& b)
			{
			    const auto t = a;
			    a = b;
			    b = t;
			}

  // Negate second vector in order to preserve sign of determinant value.
    __device__ __forceinline__
    static void		swap(vector_type& a, vector_type& b)
			{
			    const auto t = a;
			    a = -b;
			    b = t;
			}
};

template <class T> __host__ __device__ __forceinline__ Moment<T>
operator +(const Moment<T>& a, const Moment<T>& b)
{
    Moment<T>	val(a);
    return val += b;
}

template <class T> __host__ __device__ __forceinline__ Moment<T>
operator -(const Moment<T>& a, const Moment<T>& b)
{
    Moment<T>	val(a);
    return val -= b;
}

/************************************************************************
*  struct moment_creator<T, POINT_ARG>					*
************************************************************************/
template <class T, bool POINT_ARG=false>
struct moment_creator
{
    using result_type	= Moment<T>;

    template <bool POINT_ARG_=POINT_ARG> __host__ __device__ __forceinline__
    std::enable_if_t<!POINT_ARG_, result_type>
    operator ()(const vec<T, 3>& point) const
    {
	return {point, 0, 0};
    }

    template <bool POINT_ARG_=POINT_ARG> __host__ __device__ __forceinline__
    std::enable_if_t<POINT_ARG_, result_type>
    operator ()(int u, int v, const vec<T, 3>& point) const
    {
	return {point, u, v};
    }
};

/************************************************************************
*  class Plane<T>							*
************************************************************************/
template <class T>
class Plane
{
  /*
   * [[cx, cy, cz],		# center of sampled points
   *  [nx, ny, nz],		# plane normal
   *  [u,  v, mse]]		# mean of 2D sample points, mean-square error
   */
  public:
    using element_type	= T;
    using matrix_type	= mat3x<element_type, 3>;
    using vector_type	= vec<element_type, 3>;

  public:
			Plane()					= default;
    __host__ __device__ __forceinline__
			Plane(const matrix_type& m) :_m(m)	{}
    __host__ __device__ __forceinline__
			Plane(const Moment<T>& moment)
			    :_m()
			{
			    if (moment.npoints() < T(3))
			    {
				_m = invalid_plane()._m;
				return;
			    }

			    _m.x = moment.mean();

			    vector_type	evals;
			    const auto	evecs = moment.PCA(evals);
			    _m.y = evecs.z;
			    if (dot(_m.x, _m.y) > T(0))
				_m.y *= T(-1);	// should point toward camera

			    const auto	sc = 1/moment.npoints();
			    _m.z = {moment.u()*sc, moment.v()*sc, evals.z*sc};
			}

    __host__ __device__ __forceinline__
    static Plane	invalid_plane()
			{
			    return {{{0, 0, 0}, {0, 0, 0},
				     {0, 0, device::maxval<T>}}};
			}
    __host__ __device__ __forceinline__
    bool		is_valid() const
			{
			    return mse() != device::maxval<T>;
			}
    __host__ __device__ __forceinline__
    const vector_type&	center()		const	{ return _m.x; }
    __host__ __device__ __forceinline__
    const vector_type&	normal()		const	{ return _m.y; }
    __host__ __device__ __forceinline__
    element_type	u()			const	{ return _m.z.x; }
    __host__ __device__ __forceinline__
    element_type	v()			const	{ return _m.z.y; }
    __host__ __device__ __forceinline__
    element_type	mse()			const	{ return _m.z.z; }
    __host__ __device__ __forceinline__
    element_type	distance(const vector_type& point) const
			{
			    return abs(signed_distance(point));
			}
    __host__ __device__ __forceinline__
    element_type	signed_distance(const vector_type& point) const
			{
			    return dot(normal(), point - center());
			}

    friend std::ostream&
			operator <<(std::ostream& out, const Plane& plane)
			{
			    return out << plane._m;
			}

  private:
    matrix_type	_m;
};

/************************************************************************
*  struct plane_estimator<T>						*
************************************************************************/
template <class T>
struct plane_estimator
{
    using result_type = Plane<T>;

    __host__ __device__ __forceinline__
    result_type	operator ()(const Moment<T>& moment) const
		{
		    return {moment};
		}
};

/************************************************************************
*  struct normal_estimator<T>						*
************************************************************************/
template <class T>
struct normal_estimator : public plane_estimator<T>
{
  /*
   * [nx, ny, nz]		# plane normal
   */
    using result_type	= vec<T, 3>;
    using super		= plane_estimator<T>;

    __host__ __device__ __forceinline__
    result_type	operator ()(const Moment<T>& moment) const
		{
		    return super::operator ()(moment).normal();
		}
};

/************************************************************************
*  struct colored_normal<T>						*
************************************************************************/
template <class T=vec<uint8_t, 3> >
struct colored_normal
{
    using result_type	= T;

    __host__ __device__ __forceinline__
    result_type	operator ()(const vec<float, 3>& normal) const
		{
		    return {uint8_t(128 + 127*normal.x),
			    uint8_t(128 + 127*normal.y),
			    uint8_t(128 + 127*normal.z)};
		}

    template <class T_> __host__ __device__ __forceinline__
    result_type	operator ()(const Plane<T_>& plane) const
		{
		    return (*this)(plane.normal());
		}
};

/************************************************************************
*  TU::cu::get_element_ptr()						*
************************************************************************/
template <class T> inline element_t<T>*
get_element_ptr(thrust::device_ptr<T> p)
{
    return reinterpret_cast<element_t<T>*>(p.get());
}

template <class T> inline const element_t<T>*
get_element_ptr(thrust::device_ptr<const T> p)
{
    return reinterpret_cast<const element_t<T>*>(p.get());
}

/************************************************************************
*  struct to_vec<T>							*
************************************************************************/
//! カラー画素をCUDAベクトルへ変換する関数オブジェクト
/*!
  \param T	変換先のCUDAベクトルの型
*/
template <class T>
struct to_vec
{
    using result_type	= T;

    template <class S_>
    result_type	operator ()(const S_& val) const
		{
		    return T(val);
		}
    template <class T_>
    result_type	operator ()(const mat2x<T_, 1>& val) const
		{
		    return T(val.x);
		}
    template <class T_>
    result_type	operator ()(const mat3x<T_, 1>& val) const
		{
		    return T(val.x);
		}
    template <class T_>
    result_type	operator ()(const mat4x<T_, 1>& val) const
		{
		    return T(val.x);
		}
};

template <class T>
struct to_vec<mat3x<T, 1> >
{
    using result_type	= mat3x<T, 1>;

    template <class S_>
    result_type	operator ()(const S_& val) const
		{
		    return {T(val), T(val), T(val)};
		}
    template <class E_>
    result_type	operator ()(const RGB_<E_>& rgb) const
		{
		    return {T(rgb.r), T(rgb.g), T(rgb.b)};
		}
    template <class T_>
    result_type	operator ()(const mat3x<T_, 1>& val) const
		{
		    return {T(val.x), T(val.y), T(val.z)};
		}
    template <class T_>
    result_type	operator ()(const mat4x<T_, 1>& val) const
		{
		    return {T(val.x), T(val.y), T(val.z)};
		}
};

template <class T>
struct to_vec<mat4x<T, 1> >
{
    using result_type	= mat4x<T, 1>;

    template <class S_>
    result_type	operator ()(const S_& val) const
		{
		    return {T(val), T(val), T(val), T(255)};
		}
    template <class E_>
    result_type	operator ()(const RGB_<E_>& rgb) const
		{
		    return {T(rgb.r), T(rgb.g), T(rgb.b), T(rgb.a)};
		}
    template <class T_>
    result_type	operator ()(const mat3x<T_, 1>& val) const
		{
		    return {T(val.x), T(val.y), T(val.z), T(255)};
		}
    template <class T_>
    result_type	operator ()(const mat4x<T_, 1>& val) const
		{
		    return {T(val.x), T(val.y), T(val.z), T(val.w)};
		}
};

/************************************************************************
*  struct from_vec<T>							*
************************************************************************/
template <class T>
struct from_vec
{
    using result_type	= T;

    template <class T_>
    result_type	operator ()(const T_& val) const
		{
		    return T(val);
		}
    template <class T_>
    result_type	operator ()(const mat2x<T_, 1>& yuv422) const
	    	{
		    return T(yuv422.y);
		}
    template <class T_>
    result_type	operator ()(const mat3x<T_, 1>& rgb) const
		{
		    return T(0.229f*rgb.x + 0.587f*rgb.y +0.114f*rgb.z);
		}
    template <class T_>
    result_type	operator ()(const mat4x<T_, 1>& rgba) const
		{
		    return T(0.229f*rgba.x + 0.587f*rgba.y +0.114f*rgba.z);
		}
};

template <class E>
struct from_vec<RGB_<E> >
{
    using result_type	= RGB_<E>;
    using elm_t		= typename E::element_type;

    template <class T_>
    result_type	operator ()(const T_& val) const
		{
		    return {elm_t(val), elm_t(val), elm_t(val)};
		}
    template <class T_>
    result_type	operator ()(const mat3x<T_, 1>& rgb) const
	    	{
		    return {elm_t(rgb.x), elm_t(rgb.y), elm_t(rgb.z)};
		}
    template <class T_>
    result_type	operator ()(const mat4x<T_, 1>& rgba) const
		{
		    return {elm_t(rgba.x), elm_t(rgba.y),
			    elm_t(rgba.z), elm_t(rgba.w)};
		}
};

template <>
struct from_vec<YUV422>
{
    using result_type	= YUV422;
    using elm_t		= YUV422::element_type;

    template <class T_>
    result_type	operator ()(const T_& val) const
		{
		    return {elm_t(val)};
		}
    template <class T_>
    result_type	operator ()(const mat2x<T_, 1>& yuv422) const
		{
		    return {elm_t(yuv422.y), elm_t(yuv422.x)};
		}
};
}	// namespace cu
}	// namespace TU
