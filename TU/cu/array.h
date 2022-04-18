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
*  TU::cu::array<T, D>							*
************************************************************************/
template <class T, size_t D>
class array
{
  public:
    using value_type		= T;
    using reference		= T&;
    using const_reference	= const T&;
    using pointer		= thrust::device_ptr<T>;
    using const_pointer		= thrust::device_ptr<const T>;

  private:
    template <size_t I_, class DUMMY=void>
    struct for_each
    {
	template <class OP_> __device__
	void	apply(array& a, OP_ op) const
		{
		    a[I_] = op(a[I_]);
		    for_each<I_ + 1>().apply(a, op);
		}
	template <class OP_> __device__
	void	apply(array& a, const array& b, OP_ op) const
		{
		    a[I_] = op(a[I_], b[I_]);
		    for_each<I_ + 1>().apply(a, b, op);
		}
    };
    template <class DUMMY>
    struct for_each<D, DUMMY>
    {
	template <class OP_> __device__
	void	apply(array& a, OP_ op)			const	{}
	template <class OP_> __device__
	void	apply(array& a, const array& b, OP_ op)	const	{}
    };

  public:
    __device__ explicit	array(size_t) :array()	{}

    __device__		array()			= default;
    __device__		array(const array&)	= default;
    __device__ array&	operator =(const array&)= default;

    __host__ __device__ constexpr
    static size_t	size()			{ return D; }
    __device__
    const_reference	operator [](int i)const	{ return _data[i]; }
    __device__
    reference		operator [](int i)	{ return _data[i]; }
    __host__ __device__
    const_pointer	data()		  const	{ return const_pointer(_data); }
    __host__ __device__
    pointer		data()			{ return pointer(_data); }

    __device__ array&	operator +=(const array& b)
			{
			    for_each<0>().apply(*this, b,
						[] __device__
						(const auto& x, const auto& y)
						{ return x + y; });
			    return *this;
			}
    __device__ array&	operator -=(const array& b)
			{
			    for_each<0>().apply(*this, b,
						[] __device__
						(const auto& x, const auto& y)
						{ return x - y; });
			    return *this;
			}
    __device__ array&	operator *=(const value_type& c)
			{
			    for_each<0>().apply(*this,
						[c] __device__ (const auto& x)
						{ return x * c; });
			    return *this;
			}
    __device__ array&	operator /=(const value_type& c)
			{
			    for_each<0>().apply(*this,
						[c] __device__ (const auto& x)
						{ return x / c; });
			    return *this;
			}
    __device__ void	fill(const value_type& c)
			{
			    for_each<0>().apply(*this,
						[c] __device__ (auto&& x)
						{ x = c; });
			}

  private:
    value_type	_data[D];
};

template <class T, size_t D> __device__ __forceinline__ array<T, D>
operator +(const array<T, D>& a, const array<T, D>& b)
{
    auto	val(a);
    return val += b;
}

template <class T, size_t D> __device__ __forceinline__ array<T, D>
operator -(const array<T, D>& a, const array<T, D>& b)
{
    auto	val(a);
    return val -= b;
}

template <class T, size_t D> __device__ __forceinline__ array<T, D>
operator *(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val *= c;
}

template <class T, size_t D> __device__ __forceinline__ array<T, D>
operator *(const T& c, const array<T, D>& a)
{
    return a*c;
}

template <class T, size_t D> __device__ __forceinline__ array<T, D>
operator /(const array<T, D>& a, const T& c)
{
    auto	val(a);
    return val /= c;
}

}	// namespace cu
}	// namespace TU
