/*!
  \file		vec.h
  \author	Toshio UESHIBA
  \brief	cudaベクトルクラスとその演算子の定義
*/
#ifndef TU_CUDA_VEC_H
#define TU_CUDA_VEC_H

#include <cstdint>
#include <thrust/device_ptr.h>
#include "TU/Image++.h"		// for TU::RGB_<E>

namespace TU
{
namespace cuda
{
namespace detail
{
  template <class T>
  constexpr std::integral_constant<size_t, 1>	size(T)			;
  template <class E>
  constexpr std::integral_constant<size_t, E::size>
						size(RGB_<E>)		;
  constexpr std::integral_constant<size_t, 3>	size(YUV444)		;
  constexpr std::integral_constant<size_t, 2>	size(YUV422)		;
  constexpr std::integral_constant<size_t, 2>	size(YUYV422)		;
    
  constexpr std::integral_constant<size_t, 1>	size(char1)		;
  constexpr std::integral_constant<size_t, 2>	size(char2)		;
  constexpr std::integral_constant<size_t, 3>	size(char3)		;
  constexpr std::integral_constant<size_t, 4>	size(char4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(uchar1)		;
  constexpr std::integral_constant<size_t, 2>	size(uchar2)		;
  constexpr std::integral_constant<size_t, 3>	size(uchar3)		;
  constexpr std::integral_constant<size_t, 4>	size(uchar4)		;

  constexpr std::integral_constant<size_t, 1>	size(short1)		;
  constexpr std::integral_constant<size_t, 2>	size(short2)		;
  constexpr std::integral_constant<size_t, 3>	size(short3)		;
  constexpr std::integral_constant<size_t, 4>	size(short4)		;

  constexpr std::integral_constant<size_t, 1>	size(ushort1)		;
  constexpr std::integral_constant<size_t, 2>	size(ushort2)		;
  constexpr std::integral_constant<size_t, 3>	size(ushort3)		;
  constexpr std::integral_constant<size_t, 4>	size(ushort4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(int1)		;
  constexpr std::integral_constant<size_t, 2>	size(int2)		;
  constexpr std::integral_constant<size_t, 3>	size(int3)		;
  constexpr std::integral_constant<size_t, 4>	size(int4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(uint1)		;
  constexpr std::integral_constant<size_t, 2>	size(uint2)		;
  constexpr std::integral_constant<size_t, 3>	size(uint3)		;
  constexpr std::integral_constant<size_t, 4>	size(uint4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(float1)		;
  constexpr std::integral_constant<size_t, 2>	size(float2)		;
  constexpr std::integral_constant<size_t, 3>	size(float3)		;
  constexpr std::integral_constant<size_t, 4>	size(float4)		;
    
  constexpr std::integral_constant<size_t, 1>	size(longlong1)		;
  constexpr std::integral_constant<size_t, 2>	size(longlong2)		;

  constexpr std::integral_constant<size_t, 1>	size(double1)		;
  constexpr std::integral_constant<size_t, 2>	size(double2)		;

  template <class T>
  constexpr T					element_t(T)		;
  template <class E>
  constexpr typename E::element_type		element_t(RGB_<E>)	;
  constexpr YUV444::element_type		element_t(YUV444)	;
  constexpr YUV422::element_type		element_t(YUV422)	;
    
  constexpr int8_t				element_t(char1)	;
  constexpr int8_t				element_t(char2)	;
  constexpr int8_t				element_t(char3)	;
  constexpr int8_t				element_t(char4)	;

  constexpr uint8_t				element_t(uchar1)	;
  constexpr uint8_t				element_t(uchar2)	;
  constexpr uint8_t				element_t(uchar3)	;
  constexpr uint8_t				element_t(uchar4)	;

  constexpr int16_t				element_t(short1)	;
  constexpr int16_t				element_t(short2)	;
  constexpr int16_t				element_t(short3)	;
  constexpr int16_t				element_t(short4)	;

  constexpr uint16_t				element_t(ushort1)	;
  constexpr uint16_t				element_t(ushort2)	;
  constexpr uint16_t				element_t(ushort3)	;
  constexpr uint16_t				element_t(ushort4)	;

  constexpr int32_t				element_t(int1)		;
  constexpr int32_t				element_t(int2)		;
  constexpr int32_t				element_t(int3)		;
  constexpr int32_t				element_t(int4)		;

  constexpr uint32_t				element_t(uint1)	;
  constexpr uint32_t				element_t(uint2)	;
  constexpr uint32_t				element_t(uint3)	;
  constexpr uint32_t				element_t(uint4)	;

  constexpr float				element_t(float1)	;
  constexpr float				element_t(float2)	;
  constexpr float				element_t(float3)	;
  constexpr float				element_t(float4)	;
    
  constexpr int64_t				element_t(longlong1)	;
  constexpr int64_t				element_t(longlong2)	;

  constexpr double				element_t(double1)	;
  constexpr double				element_t(double2)	;

  template <class T, size_t N>	struct vec;

  template <>	struct vec<int8_t,   1>		{ using type = char1;	};
  template <>	struct vec<int8_t,   2>		{ using type = char2;	};
  template <>	struct vec<int8_t,   3>		{ using type = char3;	};
  template <>	struct vec<int8_t,   4>		{ using type = char4;	};
    
  template <>	struct vec<uint8_t,  1>		{ using type = uchar1;	};
  template <>	struct vec<uint8_t,  2>		{ using type = uchar2;	};
  template <>	struct vec<uint8_t,  3>		{ using type = uchar3;	};
  template <>	struct vec<uint8_t,  4>		{ using type = uchar4;	};

  template <>	struct vec<int16_t,  1>		{ using type = short1;	};
  template <>	struct vec<int16_t,  2>		{ using type = short2;	};
  template <>	struct vec<int16_t,  3>		{ using type = short3;	};
  template <>	struct vec<int16_t,  4>		{ using type = short4;	};

  template <>	struct vec<uint16_t, 1>		{ using type = ushort1;	};
  template <>	struct vec<uint16_t, 2>		{ using type = ushort2;	};
  template <>	struct vec<uint16_t, 3>		{ using type = ushort3;	};
  template <>	struct vec<uint16_t, 4>		{ using type = ushort4;	};

  template <>	struct vec<int32_t,  1>		{ using type = int1;	};
  template <>	struct vec<int32_t,  2>		{ using type = int2;	};
  template <>	struct vec<int32_t,  3>		{ using type = int3;	};
  template <>	struct vec<int32_t,  4>		{ using type = int4;	};

  template <>	struct vec<uint32_t, 1>		{ using type = uint1;	};
  template <>	struct vec<uint32_t, 2>		{ using type = uint2;	};
  template <>	struct vec<uint32_t, 3>		{ using type = uint3;	};
  template <>	struct vec<uint32_t, 4>		{ using type = uint4;	};

  template <>	struct vec<float,    1>		{ using type = float1;	};
  template <>	struct vec<float,    2>		{ using type = float2;	};
  template <>	struct vec<float,    3>		{ using type = float3;	};
  template <>	struct vec<float,    4>		{ using type = float4;	};

  template <>	struct vec<int64_t,  1>		{ using type = longlong1; };
  template <>	struct vec<int64_t,  2>		{ using type = longlong2; };

  template <>	struct vec<double,   1>		{ using type = double1;	};
  template <>	struct vec<double,   2>		{ using type = double2;	};
}	// namespace derail
    
template <class T, size_t N>
using vec = typename detail::vec<T, N>::type;

template <class T, size_t C>	struct mat3x;
template <class T, size_t C>	struct mat4x;
    
template <class T, size_t C>
struct mat2x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

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

    value_type	x, y;
};

template <class T, size_t C>
struct mat3x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

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

    value_type	x, y, z;
};

template <class T, size_t C>
struct mat4x
{
    using element_type	= T;
    using value_type	= vec<T, C>;

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

    value_type	x, y, z, w;
};

namespace detail
{
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, 2>	size(mat2x<T, C>)	;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, 3>	size(mat3x<T, C>)	;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, 4>	size(mat4x<T, C>)	;
  template <class T>
  constexpr std::integral_constant<size_t, 1>	ncol(T)			;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, C>	ncol(mat2x<T, C>)	;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, C>	ncol(mat3x<T, C>)	;
  template <class T, size_t C>
  constexpr std::integral_constant<size_t, C>	ncol(mat4x<T, C>)	;
  template <class MAT>
  constexpr typename MAT::element_type		element_t(MAT)		;
}	// namespace detail
    
template <class VM> constexpr static size_t
size()
{
    return decltype(detail::size(std::declval<VM>()))::value;
}

template <class VM> constexpr static size_t
ncol()
{
    return decltype(detail::ncol(std::declval<VM>()))::value;
}

template <class VM>
using element_t	= decltype(detail::element_t(std::declval<VM>()));
    
template <class T, size_t R, size_t C>
using mat = std::conditional_t<R == 2, mat2x<T, C>,
			       std::conditional_t<R == 3,
						  mat3x<T, C>, mat4x<T, C> > >;
}	// namespace cuda

//  vec<T, N> はCUDA組み込みのベクトル型の別名であり global namespace
//  で定義されている．これに関係する演算子は，ADLに頼らずに namespace TU
//  から呼ぶために，namespace TU::cuda ではなく，namespace TU の中で定義する．
/************************************************************************
*  2-dimensional vectors or 2-by-C matrices				*
************************************************************************/
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM>
operator -(const VM& a)
{
    return {-a.x, -a.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM&>
operator +=(VM& a, const VM& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM&>
operator -=(VM& a, const VM& b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM&>
operator *=(VM& a, cuda::element_t<VM> c)
{
    a.x *= c;
    a.y *= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM&>
operator /=(VM& a, cuda::element_t<VM> c)
{
    a.x /= c;
    a.y /= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM>
operator +(const VM& a, const VM& b)
{
    return {a.x + b.x, a.y + b.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM>
operator -(const VM& a, const VM& b)
{
    return {a.x - b.x, a.y - b.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM>
operator *(const VM& a, const VM& b)
{
    return {a.x * b.x, a.y * b.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM>
operator /(const VM& a, const VM& b)
{
    return {a.x / b.x, a.y / b.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM>
operator *(const VM& a, cuda::element_t<VM> c)
{
    return {a.x * c, a.y * c};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM>
operator *(cuda::element_t<VM> c, const VM& a)
{
    return {c * a.x, c * a.y};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 2, VM>
operator /(const VM& a, cuda::element_t<VM> c)
{
    return {a.x / c, a.y / c};
}

/************************************************************************
*  3-dimensional vectors or 3-by-C matrices				*
************************************************************************/
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM>
operator -(const VM& a)
{
    return {-a.x, -a.y, -a.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM&>
operator +=(VM& a, const VM& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM&>
operator -=(VM& a, const VM& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM&>
operator *=(VM& a, cuda::element_t<VM> c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM&>
operator /=(VM& a, cuda::element_t<VM> c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM>
operator +(const VM& a, const VM& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM>
operator -(const VM& a, const VM& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM>
operator *(const VM& a, const VM& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM>
operator /(const VM& a, const VM& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM>
operator *(const VM& a, cuda::element_t<VM> c)
{
    return {a.x * c, a.y * c, a.z * c};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM>
operator *(cuda::element_t<VM> c, const VM& a)
{
    return {c * a.x, c * a.y, c * a.z};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 3, VM>
operator /(const VM& a, cuda::element_t<VM> c)
{
    return {a.x / c, a.y / c, a.z / c};
}

/************************************************************************
*  4-dimensional vectors or 4-by-C matrices				*
************************************************************************/
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM>
operator -(const VM& a)
{
    return {-a.x, -a.y, -a.z, -a.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM&>
operator +=(VM& a, const VM& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM&>
operator -=(VM& a, const VM& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM&>
operator *=(VM& a, cuda::element_t<VM> c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    a.w *= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM&>
operator /=(VM& a, cuda::element_t<VM> c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
    a.w /= c;
    return a;
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM>
operator +(const VM& a, const VM& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM>
operator -(const VM& a, const VM& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM>
operator *(const VM& a, const VM& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM>
operator /(const VM& a, const VM& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM>
operator *(const VM& a, cuda::element_t<VM> c)
{
    return {a.x * c, a.y * c, a.z * c, a.w * c};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM>
operator *(cuda::element_t<VM> c, const VM& a)
{
    return {c * a.x, c * a.y, c * a.z, c * a.w};
}
    
template <class VM> __host__ __device__ inline
std::enable_if_t<cuda::size<VM>() == 4, VM>
operator /(const VM& a, cuda::element_t<VM> c)
{
    return {a.x / c, a.y / c, a.z / c, a.w / c};
}
    
/************************************************************************
*  Output functions							*
************************************************************************/
template <class VM>
std::enable_if_t<cuda::size<VM>() == 2, std::ostream&>
operator <<(std::ostream& out, const VM& a)
{
    return out << '[' << a.x << ' ' << a.y << ']';
}

template <class VM>
std::enable_if_t<cuda::size<VM>() == 3, std::ostream&>
operator <<(std::ostream& out, const VM& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ']';
}

template <class VM>
std::enable_if_t<cuda::size<VM>() == 4, std::ostream&>
operator <<(std::ostream& out, const VM& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ' ' << a.w << ']';
}

namespace cuda
{
/************************************************************************
*  set_zero()								*
************************************************************************/
template <class T> __host__ __device__ inline
std::enable_if_t<std::is_arithmetic<T>::value>
set_zero(T& x)
{
    x = 0;
}

template <class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == 2>
set_zero(VM& a)
{
    set_zero(a.x);
    set_zero(a.y);
}

template <class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == 3>
set_zero(VM& a)
{
    set_zero(a.x);
    set_zero(a.y);
    set_zero(a.z);
}

template <class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == 4>
set_zero(VM& a)
{
    set_zero(a.x);
    set_zero(a.y);
    set_zero(a.z);
    set_zero(a.w);
}

/************************************************************************
*  homogeneous()							*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 2, vec<element_t<VEC>, 3> >
homogeneous(const VEC& a)
{
    return {a.x, a.y, element_t<VEC>(1)};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 3, vec<element_t<VEC>, 4> >
homogeneous(const VEC& a)
{
    return {a.x, a.y, a.z, element_t<VEC>(1)};
}
    
/************************************************************************
*  inhomogeneous()							*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 3, vec<element_t<VEC>, 2> >
inhomogeneous(const VEC& a)
{
    return {a.x / a.z, a.y / a.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 4, vec<element_t<VEC>, 3> >
inhomogeneous(const VEC& a)
{
    return {a.x / a.w, a.y / a.w, a.z / a.w};
}
    
/************************************************************************
*  dot()								*
************************************************************************/
template <class VEC, class VM, std::enable_if_t<ncol<VEC>() == 1 &&
						size<VEC>() == 2 &&
						size<VM>()  == 2>* = nullptr>
__host__ __device__ inline auto
dot(const VEC& a, const VM& b)
{
    return a.x * b.x + a.y * b.y;
}
    
template <class VEC, class VM, std::enable_if_t<ncol<VEC>() == 1 &&
						size<VEC>() == 3 &&
						size<VM>()  == 3>* = nullptr>
__host__ __device__ inline auto
dot(const VEC& a, const VM& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
    
template <class VEC, class VM, std::enable_if_t<ncol<VEC>() == 1 &&
						size<VEC>() == 4 &&
						size<VM>()  == 4>* = nullptr>
__host__ __device__ inline auto
dot(const VEC& a, const VM& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
    
template <class T, size_t C, class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == C,
		 std::conditional_t<ncol<VM>() == 1,
				    vec<T, 2>, mat<T, 2, ncol<VM>()> > >
dot(const mat2x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a)};
}
    
template <class T, size_t C, class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == C,
		 std::conditional_t<ncol<VM>() == 1,
				    vec<T, 3>, mat<T, 3, ncol<VM>()> > >
dot(const mat3x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a), dot(m.z, a)};
}

template <class T, size_t C, class VM> __host__ __device__ inline
std::enable_if_t<size<VM>() == C,
		 std::conditional_t<ncol<VM>() == 1,
				    vec<T, 4>, mat<T, 4, ncol<VM>()> > >
dot(const mat4x<T, C>& m, const VM& a)
{
    return {dot(m.x, a), dot(m.y, a), dot(m.z, a), dot(m.w, a)};
}

/************************************************************************
*  cross()								*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<ncol<VEC>() == 1 && size<VEC>() == 3, VEC>
cross(const VEC& a, const VEC& b)
{
    return {a.y * b.z - a.z * b.y,
	    a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
    
/************************************************************************
*  ext()								*
************************************************************************/
template <class VEC0, class VEC1> __host__ __device__ inline
std::enable_if_t<ncol<VEC0>() == 1 && size<VEC0>() == 2 && ncol<VEC1>() == 1,
		 mat<element_t<VEC0>, 2, size<VEC1>()> >
ext(const VEC0& a, const VEC1& b)
{
    return {a.x * b, a.y * b};
}
    
template <class VEC0, class VEC1> __host__ __device__ inline
std::enable_if_t<ncol<VEC0>() == 1 && size<VEC0>() == 3 && ncol<VEC1>() == 1,
		 mat<element_t<VEC0>, 3, size<VEC1>()> >
ext(const VEC0& a, const VEC1& b)
{
    return {a.x * b, a.y * b, a.z * b};
}
    
template <class VEC0, class VEC1> __host__ __device__ inline
std::enable_if_t<ncol<VEC0>() == 1 && size<VEC0>() == 4 && ncol<VEC1>() == 1,
		 mat<element_t<VEC0>, 4, size<VEC1>()> >
ext(const VEC0& a, const VEC1& b)
{
    return {a.x * b, a.y * b, a.z * b, a.w * b};
}
    
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
    
    using matrix_type	= mat<T, DO1, DI1>;
    using		typename matrix_type::element_type;
    using point_type	= vec<element_type, DO>;
    using ppoint_type	= vec<element_type, DO1>;
    
    constexpr static size_t	inDim()		{ return DI; }
    constexpr static size_t	outDim()	{ return DO; }
    constexpr static size_t	nparams()	{ return NPARAMS; }

    __host__ __device__
		Projectivity(const matrix_type& m)	:matrix_type(m)	{}

    __host__ __device__
    point_type	operator ()(const vec<T, DI>& p) const
		{
		    return (*this)(p);
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
		    return dot(*this, p);
		}
    __host__ __device__
    ppoint_type	mapP(T u, T v) const
    		{
    		    return dot(*this, vec<T, 3>{u, v, T(1)});
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
		    auto	t0 = this->x.x;
		    auto	t1 = this->x.y;
		    auto	t2 = this->x.z;
		    this->x.x -= (t0*dt[0] + t1*dt[3] + t2*dt[6]);
		    this->x.y -= (t0*dt[1] + t1*dt[4] + t2*dt[7]);
		    this->x.z -= (t0*dt[2] + t1*dt[5]);
		    
		    t0 = this->y.x;
		    t1 = this->y.y;
		    t2 = this->y.z;
		    this->y.x -= (t0*dt[0] + t1*dt[3] + t2*dt[6]);
		    this->y.y -= (t0*dt[1] + t1*dt[4] + t2*dt[7]);
		    this->y.z -= (t0*dt[2] + t1*dt[5]);
		    
		    t0 = this->z.x;
		    t1 = this->z.y;
		    t2 = this->z.z;
		    this->z.x -= (t0*dt[0] + t1*dt[3] + t2*dt[6]);
		    this->z.y -= (t0*dt[1] + t1*dt[4] + t2*dt[7]);
		    this->z.z -= (t0*dt[2] + t1*dt[5]);
		}
};

/************************************************************************
*  class Affinity<T, DO, DI>						*
************************************************************************/
template <class T, size_t DO, size_t DI>
class Affinity : public mat<T, DO, DI + 1>
{
  public:
    constexpr static size_t	DO1	= DO + 1;
    constexpr static size_t	DI1	= DI + 1;
    constexpr static size_t	NPARAMS = DO * DI1;
    constexpr static size_t	DOF	= NPARAMS;
    
    using matrix_type	= mat<T, DO, DI1>;
    using		typename matrix_type::element_type;
    using point_type	= vec<element_type, DO>;
    using ppoint_type	= vec<element_type, DO1>;
    using param_type	= matrix_type;
    
    constexpr static size_t	inDim()		{ return DI; }
    constexpr static size_t	outDim()	{ return DO; }
    constexpr static size_t	nparams()	{ return NPARAMS; }

    __host__ __device__
		Affinity(const matrix_type& m)	:matrix_type(m)	{}

    __host__ __device__
    point_type	operator ()(const vec<T, DI>& p) const
		{
		    return (*this)(homogeneous(p));
		}
    __host__ __device__
    point_type	operator ()(const vec<T, DI1>& p) const
		{
		    return dot(*this, p);
		}
    __host__ __device__
    point_type	operator ()(T u, T v) const
    		{
    		    return (*this)(vec<T, 3>({u, v, T(1)}));
    		}

    __host__ __device__
    ppoint_type	mapP(const vec<T, DI>& p) const
		{
		    return homogeneous((*this)(p));
		}
    __host__ __device__
    ppoint_type	mapP(const vec<T, DI1>& p) const
		{
		    return homogeneous((*this)(p));
		}
    __host__ __device__
    ppoint_type	mapP(T u, T v) const
    		{
    		    return homogeneous((*this)(u, v));
    		}

    __host__ __device__
    void	update(const param_type& dt)
		{
		    *this -= dt;
		}

    template <size_t DO_=DO, size_t DI_=DI>
    __host__ __device__ static std::enable_if_t<DO_ == 2&& DI_ == 2, param_type>
		image_derivative0(T eH, T eV, T u, T v)
		{
		    return {{eH*u, eH*v}, {eV*u, eV*v}};
		}

    template <size_t DO_=DO, size_t DI_=DI> __host__
    std::enable_if_t<DO_ == 2&& DI_ == 2>
		compose(const TU::Array<T, DOF>& dt)
		{
		    auto	t0 = this->x.x;
		    auto	t1 = this->x.y;
		    this->x.x -= (t0*dt[0] + t1*dt[3]);
		    this->x.y -= (t0*dt[1] + t1*dt[4]);
		    this->x.z -= (t0*dt[2] + t1*dt[5]);
		    
		    t0 = this->y.x;
		    t1 = this->y.y;
		    this->y.x -= (t0*dt[0] + t1*dt[3]);
		    this->y.y -= (t0*dt[1] + t1*dt[4]);
		    this->y.z -= (t0*dt[2] + t1*dt[5]);
		}
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
    using		typename base_type::matrix_type;
    
    constexpr static size_t	dim()		{ return D; }

  public:
    __host__ __device__
		Rigidity(const matrix_type& m)	:base_type(m)	{}

    template <size_t D_=D>
    __host__ __device__ static std::enable_if_t<D_ == 2, vec<T, 3> >
		image_derivative0(T eH, T eV, T u, T v)
		{
		    return {eH, eV, eV*u - eH*v};
		}

    template <size_t D_=D> __host__ std::enable_if_t<D_ == 2>
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
*  TU::cuda::get_element_ptr()						*
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

}	// namespace cuda

/************************************************************************
*  class to_vec<T>							*
************************************************************************/
//! カラー画素をCUDAベクトルへ変換する関数オブジェクト
/*!
  \param T	変換先のCUDAベクトルの型
*/
template <class T>
class to_vec
{
  public:
    template <class S_>
    T	operator ()(const S_& val) const
	{
	    return vec(val, std::integral_constant<size_t, cuda::size<T>()>());
	}

  private:
    template <class S_>
    T	vec(const S_& val, std::integral_constant<size_t, 1>) const
	{
	    return T(val);
	}
    template <class S_>
    T	vec(const S_& val, std::integral_constant<size_t, 3>) const
	{
	    using elm_t	= cuda::element_t<T>;
	    
	    return {elm_t(val), elm_t(val), elm_t(val)};
	}
    template <class E_>
    T	vec(const RGB_<E_>& rgb, std::integral_constant<size_t, 3>) const
	{
	    using elm_t	= cuda::element_t<T>;
	    
	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b)};
	}
    template <class S_>
    T	vec(const S_& val, std::integral_constant<size_t, 4>) const
	{
	    using elm_t	= cuda::element_t<T>;
	    
	    return {elm_t(val), elm_t(val), elm_t(val), elm_t(255)};
	}
    template <class E_>
    T	vec(const RGB_<E_>& rgb, std::integral_constant<size_t, 4>) const
	{
	    using elm_t	= cuda::element_t<T>;
	    
	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b), elm_t(rgb.a)};
	}
};

/************************************************************************
*  struct from_vec<T>							*
************************************************************************/
template <class T>
struct from_vec
{
    template <class S_>
    std::enable_if_t<cuda::size<S_>() == 1, T>
    	operator ()(const S_& val) const
    	{
    	    return T(val);
    	}
    template <class S_>
    std::enable_if_t<cuda::size<S_>() == 2, T>
    	operator ()(const S_& yuv422) const
    	{
    	    return T(yuv422.y);
    	}
    template <class S_>
    std::enable_if_t<cuda::size<S_>() == 3 || cuda::size<S_>() == 4, T>
    	operator ()(const S_& rgb) const
    	{
    	    return T(0.229f*rgb.x + 0.587f*rgb.y +0.114f*rgb.z);
    	}
};

template <class E>
struct from_vec<RGB_<E> >
{
    using elm_t	 = typename E::element_type;
    
    template <class S_>
    std::enable_if_t<cuda::size<S_>() == 1, RGB_<E> >
    	operator ()(const S_& val) const
    	{
    	    return {elm_t(val), elm_t(val), elm_t(val)};
    	}
    template <class S_>
    std::enable_if_t<cuda::size<S_>() == 3, RGB_<E> >
    	operator ()(const S_& rgb) const
    	{
    	    return {elm_t(rgb.x), elm_t(rgb.y), elm_t(rgb.z)};
    	}
    template <class S_>
    std::enable_if_t<cuda::size<S_>() == 4, RGB_<E> >
	operator ()(const S_& rgba) const
	{
	    return {elm_t(rgba.x), elm_t(rgba.y),
		    elm_t(rgba.z), elm_t(rgba.w)};
	}
};

template <>
struct from_vec<YUV422>
{
    using elm_t	 = YUV422::element_type;
    
    template <class S_>
    std::enable_if_t<cuda::size<S_>() == 1, YUV422>
	operator ()(const S_& val) const
	{
	    return {elm_t(val)};
	}
    template <class S_>
    std::enable_if_t<cuda::size<S_>() == 2, YUV422>
	operator ()(const S_& yuv422) const
	{
	    return {elm_t(yuv422.y), elm_t(yuv422.x)};
	}
};

}	// namespace TU
#endif	// !TU_CUDA_VEC_H
