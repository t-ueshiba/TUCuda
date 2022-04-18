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
#include <thrust/device_ptr.h>
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
  __device__ __forceinline__ float  sin(float x)	{ return sinf(x); }
  __device__ __forceinline__ float  cos(float x)	{ return cosf(x); }
  __device__ __forceinline__ float  atan2(float y,
					  float x)	{ return atan2f(y, x); }
  __device__ __forceinline__ float  floor(float x)	{ return floorf(x); }
  __device__ __forceinline__ float  ceil(float x)	{ return ceilf(x); }

  __device__ __forceinline__ double min(double x,
					double y)	{ return fmin(x, y); }
  __device__ __forceinline__ double max(double x,
					double y)	{ return fmax(x, y); }
  __device__ __forceinline__ double square(double x)	{ return x*x; }
  __device__ __forceinline__ double abs(double x)	{ return fabs(x); }
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
    static size_t	rank()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    __host__ __device__	mat2x()			:super()		{}
    __host__ __device__ constexpr
			mat2x(T c)		:super{c, c}		{}
    __host__ __device__ constexpr
			mat2x(T x, T y)		:super{x, y}		{}

    __host__ __device__
    mat2x&		operator =(T c)
			{
			    x = c;
			    y = c;
			    return *this;
			}

    __host__ __device__ constexpr
    static mat2x	zero()			{ return {T(0), T(0)}; }
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
    static size_t	rank()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 3; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    __host__ __device__	mat3x()		     :super()			{}
    __host__ __device__ constexpr
			mat3x(T c)	     :super{c, c, c}		{}
    __host__ __device__ constexpr
			mat3x(T x, T y, T z) :super{x, y, z}		{}

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
    static size_t	rank()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 4; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return 1; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    __host__ __device__	mat4x()			  :super()		{}
    __host__ __device__ constexpr
			mat4x(T c)		  :super{c, c, c, c}	{}
    __host__ __device__ constexpr
			mat4x(T x, T y, T z, T w) :super{x, y, z, w}	{}

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
    static size_t	rank()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return C; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    __host__ __device__	mat2x()						{}
    __host__ __device__	constexpr
			mat2x(const value_type& xx, const value_type& yy)
			    :x(xx), y(yy)				{}
    __host__ __device__	constexpr
			mat2x(T c) :x(c), y(c)				{}

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

    value_type	x, y;
};

template <class T, size_t C>
struct mat3x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

    __host__ __device__ constexpr
    static size_t	rank()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 3; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return C; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    __host__ __device__	mat3x()						{}
    __host__ __device__	constexpr
			mat3x(const value_type& xx, const value_type& yy,
			      const value_type& zz)
			    :x(xx), y(yy), z(zz)			{}
    __host__ __device__	constexpr
			mat3x(T c) :x(c), y(c), z(c)			{}

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

    value_type	x, y, z;
};

template <class T, size_t C>
struct mat4x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

    __host__ __device__ constexpr
    static size_t	rank()			{ return 2; }
    __host__ __device__ constexpr
    static size_t	size0()			{ return 4; }
    __host__ __device__ constexpr
    static size_t	size1()			{ return C; }
    __host__ __device__ constexpr
    static size_t	size()			{ return size0(); }

    __host__ __device__	mat4x()						{}
    __host__ __device__	constexpr
			mat4x(const value_type& xx, const value_type& yy,
			      const value_type& zz, const value_type& ww)
			    :x(xx), y(yy), z(zz), w(ww)			{}
    __host__ __device__	constexpr
			mat4x(T c) :x(c), y(c), z(c), w(c)		{}

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
__host__ __device__ __forceinline__ auto	val(const VM& a){ return a.x; }
template <size_t I, class VM, std::enable_if_t<I == 1>* = nullptr>
__host__ __device__ __forceinline__ auto	val(const VM& a){ return a.y; }
template <size_t I, class VM, std::enable_if_t<I == 2>* = nullptr>
__host__ __device__ __forceinline__ auto	val(const VM& a){ return a.z; }
template <size_t I, class VM, std::enable_if_t<I == 3>* = nullptr>
__host__ __device__ __forceinline__ auto	val(const VM& a){ return a.w; }

template <size_t I, class VM, std::enable_if_t<I == 0>* = nullptr>
__host__ __device__ __forceinline__ auto&	val(VM& a)	{ return a.x; }
template <size_t I, class VM, std::enable_if_t<I == 1>* = nullptr>
__host__ __device__ __forceinline__ auto&	val(VM& a)	{ return a.y; }
template <size_t I, class VM, std::enable_if_t<I == 2>* = nullptr>
__host__ __device__ __forceinline__ auto&	val(VM& a)	{ return a.z; }
template <size_t I, class VM, std::enable_if_t<I == 3>* = nullptr>
__host__ __device__ __forceinline__ auto&	val(VM& a)	{ return a.w; }

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
*  Output functions							*
************************************************************************/
template <class T, size_t C> std::ostream&
operator <<(std::ostream& out, const mat2x<T, C>& a)
{
    return out << '[' << a.x << ' ' << a.y << ']';
}

template <class T, size_t C> std::ostream&
operator <<(std::ostream& out, const mat3x<T, C>& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ']';
}

template <class T, size_t C> std::ostream&
operator <<(std::ostream& out, const mat4x<T, C>& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ' ' << a.w << ']';
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
std::enable_if_t<size1<VEC>() == 1, mat2x<T, size1<VEC>()> >
ext(const mat2x<T, 1>& a, const VEC& b)
{
    return {a.x * b, a.y * b};
}

template <class T, class VEC> __host__ __device__ __forceinline__
std::enable_if_t<size1<VEC>() == 1, mat3x<T, size1<VEC>()> >
ext(const mat3x<T, 1>& a, const VEC& b)
{
    return {a.x * b, a.y * b, a.z * b};
}

template <class T, class VEC> __host__ __device__ __forceinline__
std::enable_if_t<size1<VEC>() == 1, mat4x<T, size1<VEC>()> >
ext(const mat4x<T, 1>& a, const VEC& b)
{
    return {a.x * b, a.y * b, a.z * b, a.w * b};
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
class Projectivity : public mat<T, DO + 1, DI + 1>
{
  public:
    constexpr static size_t	DO1	= DO + 1;
    constexpr static size_t	DI1	= DI + 1;
    constexpr static size_t	NPARAMS = DO1 * DI1;
    constexpr static size_t	DOF	= NPARAMS - 1;

    using element_type	= T;
    using matrix_type	= mat<element_type, DO1, DI1>;
    using point_type	= vec<element_type, DO>;
    using ppoint_type	= vec<element_type, DO1>;

    constexpr static size_t	inDim()		{ return DI; }
    constexpr static size_t	outDim()	{ return DO; }
    constexpr static size_t	nparams()	{ return NPARAMS; }

    __host__ __device__
		Projectivity()	:_m()				{}
    __host__ __device__
		Projectivity(const matrix_type& m)	:_m(m)	{}

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
    static std::enable_if_t<DO_ == 2&& DI_ == 2, mat<T, 2, 4> >
		image_derivative0(T eH, T eV, T u, T v)
		{
		    return {{eH*u, eH*v, eH, eV*u},
			    {eV*v, eV, -(eH*u + eV*v)*u, -(eH*u + eV*v)*v}};
		}

    template <size_t DO_=DO, size_t DI_=DI> __host__
    std::enable_if_t<DO_ == 2&& DI_ == 2>
		compose(const TU::Array<T, DOF>& dt)
		{
		    auto	t0 = _m.x.x;
		    auto	t1 = _m.x.y;
		    auto	t2 = _m.x.z;
		    _m.x.x -= (t0*dt[0] + t1*dt[3] + t2*dt[6]);
		    _m.x.y -= (t0*dt[1] + t1*dt[4] + t2*dt[7]);
		    _m.x.z -= (t0*dt[2] + t1*dt[5]);

		    t0 = _m.y.x;
		    t1 = _m.y.y;
		    t2 = _m.y.z;
		    _m.y.x -= (t0*dt[0] + t1*dt[3] + t2*dt[6]);
		    _m.y.y -= (t0*dt[1] + t1*dt[4] + t2*dt[7]);
		    _m.y.z -= (t0*dt[2] + t1*dt[5]);

		    t0 = _m.z.x;
		    t1 = _m.z.y;
		    t2 = _m.z.z;
		    _m.z.x -= (t0*dt[0] + t1*dt[3] + t2*dt[6]);
		    _m.z.y -= (t0*dt[1] + t1*dt[4] + t2*dt[7]);
		    _m.z.z -= (t0*dt[2] + t1*dt[5]);
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
    constexpr static size_t	DO1	= DO + 1;
    constexpr static size_t	DI1	= DI + 1;
    constexpr static size_t	NPARAMS = DO * DI1;
    constexpr static size_t	DOF	= NPARAMS;

    using element_type	= T;
    using matrix_type	= mat<element_type, DO, DI>;
    using point_type	= vec<element_type, DO>;

    constexpr static size_t	inDim()		{ return DI; }
    constexpr static size_t	outDim()	{ return DO; }
    constexpr static size_t	nparams()	{ return NPARAMS; }

    __host__ __device__
		Affinity() :_A(), _b()		{}
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
    static std::enable_if_t<DO_ == 2 && DI_ == 2, matrix_type>
		image_derivative0(T eH, T eV, T u, T v)
		{
		    return {{eH*u, eH*v}, {eV*u, eV*v}};
		}

    template <size_t DO_=DO, size_t DI_=DI> __host__
    std::enable_if_t<DO_ == 2 && DI_ == 2>
		compose(const TU::Array<T, DOF>& dt)
		{
		    auto	t0 = _A.x.x;
		    auto	t1 = _A.x.y;
		    _A.x.x -= (t0*dt[0] + t1*dt[3]);
		    _A.x.y -= (t0*dt[1] + t1*dt[4]);
		    _b.x   -= (t0*dt[2] + t1*dt[5]);

		    t0 = _A.y.x;
		    t1 = _A.y.y;
		    _A.y.x -= (t0*dt[0] + t1*dt[3]);
		    _A.y.y -= (t0*dt[1] + t1*dt[4]);
		    _b.y   -= (t0*dt[2] + t1*dt[5]);
		}

  private:
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
    constexpr static size_t	NPARAMS = D*(D+1)/2;
    constexpr static size_t	DOF	= NPARAMS;

    using base_type	= Affinity<T, D, D>;
    using		typename base_type::element_type;
    using		typename base_type::matrix_type;
    using		typename base_type::point_type;
    using normal_type	= vec<T, D>;

    constexpr static size_t	dim()		{ return D; }

  public:
    __host__ __device__
		Rigidity() :base_type()		{}
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
    normal_type	map_normal(const normal_type& n) const
		{
		    return dot(n, R());
		}

    template <size_t D_=D> __host__ __device__
    static std::enable_if_t<D_ == 2, vec<T, 3> >
		image_derivative0(T eH, T eV, T u, T v)
		{
		    return {eH, eV, eV*u - eH*v};
		}

    template <size_t D_=D> __host__
    std::enable_if_t<D_ == 2>
		compose(const TU::Array<T, DOF>& dt)
		{
		    const auto	Rt = rotation(dt[2]);

		    auto	r0 = this->x.x;
		    auto	r1 = this->x.y;
		    this->x.x  = r0*Rt[0][0] + r1*Rt[1][0];
		    this->x.y  = r0*Rt[0][1] + r1*Rt[1][1];
		    this->x.z -= (this->x.x*dt[0] + this->x.y*dt[1]);

		    r0 = this->y.x;
		    r1 = this->y.y;
		    this->y.x  = r0*Rt[0][0] + r1*Rt[1][0];
		    this->y.y  = r0*Rt[0][1] + r1*Rt[1][1];
		    this->y.z -= (this->y.x*dt[0] + this->y.y*dt[1]);
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
