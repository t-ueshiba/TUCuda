/*!
  Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
   * Neither the name of National Institute of Advanced Industrial
     Science and Technology (AIST) nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.

  Author: Toshio Ueshiba

  \file		array.h
  \brief	CUDAデバイス上の配列に関連するクラスの定義と実装
*/
#pragma once

#include "TU/cu/Array++.h"

namespace TU
{
namespace cu
{
/************************************************************************
*  TU::cu::device::array<T, D>						*
************************************************************************/
template <class T, size_t D>
class array
{
  public:
    using value_type		= T;
    using reference		= T&;
    using const_reference	= const T&;
    using pointer		= T*;
    using const_pointer		= const T*;

    constexpr static size_t	D_SQOP = (D*(D+1))/2;
    
  private:
    template <size_t I=0, class DUMMY=void>
    struct for_each
    {
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(array& a, OP_ op)
			{
			    op(a[I]);
			    for_each<I + 1>::apply(a, op);
			}
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(array& a, const array& b, OP_ op)
			{
			    op(a[I], b[I]);
			    for_each<I + 1>::apply(a, b, op);
			}
	__host__ __device__ __forceinline__
	static T	dot(const array& a, const array& b)
			{
			    return a[I]*b[I] + for_each<I + 1>::dot(a, b);
			}
    };
    template <class DUMMY>
    struct for_each<D, DUMMY>
    {
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(array& a, OP_ op)			{}
	template <class OP_> __host__ __device__ __forceinline__
	static void	apply(array& a, const array& b, OP_ op)	{}
	__host__ __device__ __forceinline__
	static T	dot(const array& a, const array& b)	{ return T(0); }
    };

    template <size_t I=0, size_t J=0, class DUMMY=void>
    struct outpro
    {
	__host__ __device__ __forceinline__
	static void	apply(const array& in, array<T, D_SQOP>& out)
			{
			    out[D*I + J - (I*(I+1))/2] = in[I] * in[J];
			    outpro<I, J+1>::apply(in, out);
			}
	
    };
    template <size_t I, class DUMMY>
    struct outpro<I, D, DUMMY>
    {
	__host__ __device__ __forceinline__
	static void	apply(const array& in, array<T, D_SQOP>& out)
			{
			    outpro<I+1, I+1>::apply(in, out);
			}
    };
    template <class DUMMY>
    struct outpro<D, D, DUMMY>
    {
	__host__ __device__ __forceinline__
	static void	apply(const array& in, array<T, D_SQOP>& out)
			{
			}
    };

  public:
    // __host__ __device__
    // explicit		array(size_t) :array()	{}
    // 			array()			= default;

    __host__ __device__ constexpr
    static int		size()			{ return D; }

    __host__ __device__
    const_reference	operator [](int i)const	{ return _data[i]; }
    __host__ __device__
    reference		operator [](int i)	{ return _data[i]; }
    __host__ __device__
    const_pointer	data()		const	{ return _data; }
    __host__ __host__ __device__
    pointer		data()			{ return _data; }
    
    __host__ __device__
    array&		operator +=(const array& b)
			{
			    for_each<>::apply(*this, b,
					      [] __host__ __device__
					      (auto&& x, const auto& y)
					      { x += y; });
			    return *this;
			}
    __host__ __device__
    array&		operator -=(const array& b)
			{
			    for_each<>::apply(*this, b,
					      [] __host__ __device__
					      (auto&& x, const auto& y)
					      { x -= y; });
			    return *this;
			}
    __host__ __device__
    array&		operator *=(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (auto&& x){ x *= c; });
			    return *this;
			}
    __host__ __device__
    array&		operator /=(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (auto&& x){ x /= c; });
			    return *this;
			}
    __host__ __device__
    void		fill(const value_type& c)
			{
			    for_each<>::apply(*this,
					      [c] __host__ __device__
					      (auto&& x){ x = c; });
			}
    __host__ __device__
    value_type		dot(const array& b) const
			{
			    return for_each<>::dot(*this, b);
			}
    __host__ __device__
    array<T, D_SQOP>	outer_product() const
			{
			    array<T, D_SQOP>	val;
			    outpro<>::apply(*this, val);
			    return val;
			}

  private:
    value_type	_data[D];
};

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator +(const array<T, D>& a, const array<T, D>& b)
{
    auto	val(a);
    return val += b;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator -(const array<T, D>& a, const array<T, D>& b)
{
    auto	val(a);
    return val -= b;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator *(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val *= c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator *(const T& c, const array<T, D>& a)
{
    return a*c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ array<T, D>
operator /(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val /= c;
}

template <class T, size_t D> __host__ __device__ __forceinline__ T
dot(const array<T, D>& a, const array<T, D>& b)
{
    return a.dot(b);
}

template <class T, size_t D> std::ostream&
operator <<(std::ostream& out, const array<T, D>& a)
{
    for (size_t i = 0; i < a.size(); ++i)
	out << ' ' << a[i];
    return out;
}

}	// namespace cu
}	// namespace TU
