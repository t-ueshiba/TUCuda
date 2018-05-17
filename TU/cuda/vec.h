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
  template <class VEC>	constexpr size_t	size()		{ return 1; }

  template <>		constexpr size_t	size<char1>()	{ return 1; }
  template <>		constexpr size_t	size<char2>()	{ return 2; }
  template <>		constexpr size_t	size<char3>()	{ return 3; }
  template <>		constexpr size_t	size<char4>()	{ return 4; }
    
  template <>		constexpr size_t	size<uchar1>()	{ return 1; }
  template <>		constexpr size_t	size<uchar2>()	{ return 2; }
  template <>		constexpr size_t	size<uchar3>()	{ return 3; }
  template <>		constexpr size_t	size<uchar4>()	{ return 4; }
    
  template <>		constexpr size_t	size<short1>()	{ return 1; }
  template <>		constexpr size_t	size<short2>()	{ return 2; }
  template <>		constexpr size_t	size<short3>()	{ return 3; }
  template <>		constexpr size_t	size<short4>()	{ return 4; }
    
  template <>		constexpr size_t	size<ushort1>()	{ return 1; }
  template <>		constexpr size_t	size<ushort2>()	{ return 2; }
  template <>		constexpr size_t	size<ushort3>()	{ return 3; }
  template <>		constexpr size_t	size<ushort4>()	{ return 4; }
    
  template <>		constexpr size_t	size<int1>()	{ return 1; }
  template <>		constexpr size_t	size<int2>()	{ return 2; }
  template <>		constexpr size_t	size<int3>()	{ return 3; }
  template <>		constexpr size_t	size<int4>()	{ return 4; }
    
  template <>		constexpr size_t	size<uint1>()	{ return 1; }
  template <>		constexpr size_t	size<uint2>()	{ return 2; }
  template <>		constexpr size_t	size<uint3>()	{ return 3; }
  template <>		constexpr size_t	size<uint4>()	{ return 4; }
    
  template <>		constexpr size_t	size<float1>()	{ return 1; }
  template <>		constexpr size_t	size<float2>()	{ return 2; }
  template <>		constexpr size_t	size<float3>()	{ return 3; }
  template <>		constexpr size_t	size<float4>()	{ return 4; }
    
  template <>		constexpr size_t	size<longlong1>() { return 1; }
  template <>		constexpr size_t	size<longlong2>() { return 2; }
    
  template <>		constexpr size_t	size<double1>()	{ return 1; }
  template <>		constexpr size_t	size<double2>()	{ return 2; }

  template <class VEC>	VEC		element_t(VEC)		;

			int8_t		element_t(char1)	;
			int8_t		element_t(char2)	;
			int8_t		element_t(char3)	;
			int8_t		element_t(char4)	;

			uint8_t		element_t(uchar1)	;
			uint8_t		element_t(uchar2)	;
			uint8_t		element_t(uchar3)	;
			uint8_t		element_t(uchar4)	;

			int16_t		element_t(short1)	;
			int16_t		element_t(short2)	;
			int16_t		element_t(short3)	;
			int16_t		element_t(short4)	;

			uint16_t	element_t(ushort1)	;
			uint16_t	element_t(ushort2)	;
			uint16_t	element_t(ushort3)	;
			uint16_t	element_t(ushort4)	;

			int32_t		element_t(int1)		;
			int32_t		element_t(int2)		;
			int32_t		element_t(int3)		;
			int32_t		element_t(int4)		;

			uint32_t	element_t(uint1)	;
			uint32_t	element_t(uint2)	;
			uint32_t	element_t(uint3)	;
			uint32_t	element_t(uint4)	;

			float		element_t(float1)	;
			float		element_t(float2)	;
			float		element_t(float3)	;
			float		element_t(float4)	;
    
			int64_t		element_t(longlong1)	;
			int64_t		element_t(longlong2)	;

			double		element_t(double1)	;
			double		element_t(double2)	;

  template <class T, size_t N>	struct vec;

  template <>		struct vec<int8_t,   1>	{ using type = char1;	};
  template <>		struct vec<int8_t,   2>	{ using type = char2;	};
  template <>		struct vec<int8_t,   3>	{ using type = char3;	};
  template <>		struct vec<int8_t,   4>	{ using type = char4;	};
    
  template <>		struct vec<uint8_t,  1>	{ using type = uchar1;	};
  template <>		struct vec<uint8_t,  2>	{ using type = uchar2;	};
  template <>		struct vec<uint8_t,  3>	{ using type = uchar3;	};
  template <>		struct vec<uint8_t,  4>	{ using type = uchar4;	};

  template <>		struct vec<int16_t,  1>	{ using type = short1;	};
  template <>		struct vec<int16_t,  2>	{ using type = short2;	};
  template <>		struct vec<int16_t,  3>	{ using type = short3;	};
  template <>		struct vec<int16_t,  4>	{ using type = short4;	};

  template <>		struct vec<uint16_t, 1>	{ using type = ushort1;	};
  template <>		struct vec<uint16_t, 2>	{ using type = ushort2;	};
  template <>		struct vec<uint16_t, 3>	{ using type = ushort3;	};
  template <>		struct vec<uint16_t, 4>	{ using type = ushort4;	};

  template <>		struct vec<int32_t,  1>	{ using type = int1;	};
  template <>		struct vec<int32_t,  2>	{ using type = int2;	};
  template <>		struct vec<int32_t,  3>	{ using type = int3;	};
  template <>		struct vec<int32_t,  4>	{ using type = int4;	};

  template <>		struct vec<uint32_t, 1>	{ using type = uint1;	};
  template <>		struct vec<uint32_t, 2>	{ using type = uint2;	};
  template <>		struct vec<uint32_t, 3>	{ using type = uint3;	};
  template <>		struct vec<uint32_t, 4>	{ using type = uint4;	};

  template <>		struct vec<float,    1>	{ using type = float1;	};
  template <>		struct vec<float,    2>	{ using type = float2;	};
  template <>		struct vec<float,    3>	{ using type = float3;	};
  template <>		struct vec<float,    4>	{ using type = float4;	};

  template <>		struct vec<int64_t,  1>	{ using type = longlong1; };
  template <>		struct vec<int64_t,  2>	{ using type = longlong2; };

  template <>		struct vec<double,   1>	{ using type = double1;	};
  template <>		struct vec<double,   2>	{ using type = double2;	};
}	// namespace detail
    
template <class VEC>
using element_t	= decltype(detail::element_t(std::declval<VEC>()));
    
template <class VEC>
constexpr static size_t	size()	{ return detail::size<std::decay_t<VEC> >(); }

template <class T, size_t N>
using vec	= typename detail::vec<T, N>::type;
}	// namespace cuda
}	// namespace TU

#if defined(__NVCC__)
/************************************************************************
*  2-dimensional vectors						*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC&>
operator +=(VEC& a, const VEC& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC&>
operator -=(VEC& a, const VEC& b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC&>
operator *=(VEC& a, TU::cuda::element_t<VEC> c)
{
    a.x *= c;
    a.y *= c;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC&>
operator /=(VEC& a, TU::cuda::element_t<VEC> c)
{
    a.x /= c;
    a.y /= c;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC>
operator +(const VEC& a, const VEC& b)
{
    return {a.x + b.x, a.y + b.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC>
operator -(const VEC& a, const VEC& b)
{
    return {a.x - b.x, a.y - b.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC>
operator *(const VEC& a, const VEC& b)
{
    return {a.x * b.x, a.y * b.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC>
operator /(const VEC& a, const VEC& b)
{
    return {a.x / b.x, a.y / b.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC>
operator *(const VEC& a, TU::cuda::element_t<VEC> c)
{
    return {a.x * c, a.y * c};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC>
operator *(TU::cuda::element_t<VEC> c, const VEC& a)
{
    return {c * a.x, c * a.y};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 2, VEC>
operator /(const VEC& a, TU::cuda::element_t<VEC> c)
{
    return {a.x / c, a.y / c};
}
    
/************************************************************************
*  3-dimensional vectors						*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC&>
operator +=(VEC& a, const VEC& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC&>
operator -=(VEC& a, const VEC& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC&>
operator *=(VEC& a, TU::cuda::element_t<VEC> c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC&>
operator /=(VEC& a, TU::cuda::element_t<VEC> c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC>
operator +(const VEC& a, const VEC& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC>
operator -(const VEC& a, const VEC& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC>
operator *(const VEC& a, const VEC& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC>
operator /(const VEC& a, const VEC& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC>
operator *(const VEC& a, TU::cuda::element_t<VEC> c)
{
    return {a.x * c, a.y * c, a.z * c};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC>
operator *(TU::cuda::element_t<VEC> c, const VEC& a)
{
    return {c * a.x, c * a.y, c * a.z};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 3, VEC>
operator /(const VEC& a, TU::cuda::element_t<VEC> c)
{
    return {a.x / c, a.y / c, a.z / c};
}
    
/************************************************************************
*  4-dimensional vectors						*
************************************************************************/
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC&>
operator +=(VEC& a, const VEC& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC&>
operator -=(VEC& a, const VEC& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC&>
operator *=(VEC& a, TU::cuda::element_t<VEC> c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    a.w *= c;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC&>
operator /=(VEC& a, TU::cuda::element_t<VEC> c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
    a.w /= c;
    return a;
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC>
operator +(const VEC& a, const VEC& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC>
operator -(const VEC& a, const VEC& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC>
operator *(const VEC& a, const VEC& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC>
operator /(const VEC& a, const VEC& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC>
operator *(const VEC& a, TU::cuda::element_t<VEC> c)
{
    return {a.x * c, a.y * c, a.z * c, a.w * c};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC>
operator *(TU::cuda::element_t<VEC> c, const VEC& a)
{
    return {c * a.x, c * a.y, c * a.z, c * a.w};
}
    
template <class VEC> __host__ __device__ inline
std::enable_if_t<TU::cuda::size<VEC>() == 4, VEC>
operator /(const VEC& a, TU::cuda::element_t<VEC> c)
{
    return {a.x / c, a.y / c, a.z / c, a.w / c};
}
    
/************************************************************************
*  Output functions							*
************************************************************************/
template <class VEC>
std::enable_if_t<TU::cuda::size<VEC>() == 2, std::ostream&>
operator <<(std::ostream& out, const VEC& a)
{
    return out << '[' << a.x << ' ' << a.y << ']';
}

template <class VEC>
std::enable_if_t<TU::cuda::size<VEC>() == 3, std::ostream&>
operator <<(std::ostream& out, const VEC& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ']';
}

template <class VEC>
std::enable_if_t<TU::cuda::size<VEC>() == 4, std::ostream&>
operator <<(std::ostream& out, const VEC& a)
{
    return out << '[' << a.x << ' ' << a.y << ' ' << a.z << ' ' << a.w << ']';
}

namespace TU
{
/************************************************************************
*  struct color_to_vec<VEC>						*
************************************************************************/
//! カラー画素をCUDAベクトルへ変換する関数オブジェクト
/*!
  \param VEC	変換先のCUDAベクトルの型
*/
template <class VEC>
struct color_to_vec
{
    template <class E_>
    std::enable_if_t<E_::size == 3, VEC>
    	operator ()(const RGB_<E_>& rgb) const
	{
	    using elm_t	= cuda::element_t<VEC>;
	    
	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b)};
	}
    template <class E_>
    std::enable_if_t<E_::size == 4, VEC>
    	operator ()(const RGB_<E_>& rgb) const
	{
	    using elm_t	= cuda::element_t<VEC>;
	    
	    return {elm_t(rgb.r), elm_t(rgb.g), elm_t(rgb.b), elm_t(rgb.a)};
	}
};

/************************************************************************
*  struct vec_to_color<COLOR>						*
************************************************************************/
//! CUDAベクトルをカラー画素へ変換する関数オブジェクト
/*!
  \param COLOR	変換先のカラー画素の型
*/
template <class COLOR>
struct vec_to_color
{
    template <class VEC_>
    std::enable_if_t<cuda::size<VEC_>() == 3, COLOR>
	operator ()(const VEC_& v) const
	{
	    using elm_t	= typename COLOR::element_type;
	    
	    return {elm_t(v.x), elm_t(v.y), elm_t(v.z)};
	}
    template <class VEC_>
    std::enable_if_t<cuda::size<VEC_>() == 4, COLOR>
    	operator ()(const VEC_& v) const
	{
	    using elm_t	= typename COLOR::element_type;
	    
	    return {elm_t(v.x), elm_t(v.y), elm_t(v.z), elm_t(v.w)};
	}
};

}	// namespace TU

namespace thrust
{
/************************************************************************
*  algorithms overloaded for thrust::device_ptr<__half>			*
************************************************************************/
template <size_t N, class E, class VEC> inline void
copy(const TU::RGB_<E>* p, size_t n, device_ptr<VEC> q)
{
    copy_n(TU::make_map_iterator(TU::color_to_vec<VEC>(), p), (N ? N : n), q);
}

template <size_t N, class VEC, class E> inline void
copy(device_ptr<const VEC> p, size_t n, TU::RGB_<E>* q)
{
#if 0
    copy_n(p, (N ? N : n),
	   TU::make_assignment_iterator(q, TU::vec_to_color<TU::RGB_<E> >()));
#else
    TU::Array<VEC, N>	tmp(n);
    copy_n(p, (N ? N : n), tmp.begin());
    std::copy_n(tmp.cbegin(), (N ? N : n),
		TU::make_assignment_iterator(
		    TU::vec_to_color<TU::RGB_<E> >(), q));
#endif
}

}	// namespace thrust
#endif	// __NVCC__    
#endif	// !TU_CUDA_VEC_H
