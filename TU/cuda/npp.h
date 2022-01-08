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
  \file		npp.h
  \author	Toshio UESHIBA
*/
#pragma once

#include <cstdint>
#include <npp.h>

#define NPP_TYPE(type)			NPP_TYPE_##type
#define NPP_NCHANNELS(nchannels)	NPP_NCHANNELS_##nchannels
#define NPP_INPLACE(inplace)		NPP_INPLACE_##inplace
#define NPP_ROI(roi)			NPP_ROI_##roi
#define NPP_SCALING(scaling)		NPP_SCALING_##scaling
#define NPP_CTX(ctx)			NPP_CTX_##ctx

#define NPP_TYPE_uint8_t		_8u
#define NPP_TYPE_int8_t			_8s
#define NPP_TYPE_uint16_t		_16u
#define NPP_TYPE_int16_t		_16s
#define NPP_TYPE_uint32_t		_32u
#define NPP_TYPE_int32_t		_32s
#define NPP_TYPE_uint64_t		_64u
#define NPP_TYPE_int64_t		_64s
#define NPP_TYPE_float			_32f
#define NPP_TYPE_double			_64f

#define NPP_NCHANNELS_1			_C1
#define NPP_NCHANNELS_2			_C2
#define NPP_NCHANNELS_3			_C3
#define NPP_NCHANNELS_4			_C4

#define NPP_INPLACE_true		I
#define NPP_INPLACE_false

#define NPP_ROI_true			R
#define NPP_ROI_false

#define NPP_SCALING_true		_Sfs
#define NPP_SCALING_false

#define NPP_CTX_true			_Ctx
#define NPP_CTX_false

#define NPP_CAT(op, type, nchannels, inplace, roi, scaling, ctx)	\
    op##type##nchannels##inplace##roi##scaling##ctx

#define NPP_MNEMONIC(op, type, nchannels, inplace, roi, scaling, ctx)	\
    NPP_CAT(op, type, nchannels, inplace, roi, scaling, ctx)

#define NPP_FUNC(signature, op, args, type, nchannels,			\
		 inplace, roi, scaling, ctx)				\
    template <> inline NppStatus signature				\
    {									\
	return NPP_MNEMONIC(op, NPP_TYPE(type),				\
			    NPP_NCHANNELS(nchannels),			\
			    NPP_INPLACE(inplace), NPP_ROI(roi),		\
			    NPP_SCALING(scaling), NPP_CTX(ctx))args;	\
    }

namespace TU
{
namespace cuda
{
template <size_t N, class T> NppStatus
nppiFilterGauss(const T* src, int src_type, T* dst, int dst_step,
		NppiSize roi_size, NppiMaskSize mask_size)		;
template <size_t N, class T> NppStatus
nppiFilterGauss(const T* src, int src_type, T* dst, int dst_step,
		NppiSize roi_size, NppiMaskSize mask_size,
		NppStreamContext ctx)					;

template <class IN, class OUT> NppStatus
nppiFilterGauss(IN in, IN ie, OUT out, NppiMaskSize mask_size)
{
    using value_t = value_t<iterator_value<IN> >;

    return nppiFilterGauss<size<value_t>()>(
		get_element_ptr(std::cbegin(*in)),
		stride(in)*sizeof(value_t),
		get_element_ptr(std::begin(*out)),
		stride(out)*sizeof(value_t),
		{int(std::distance(std::cbegin(*in), std::cend(*in))),
		 int(std::distance(in, ie))},
		mask_size);
}

template <class IN, class OUT> NppStatus
nppiFilterGauss(IN in, IN ie, OUT out,
		NppiMaskSize mask_size, NppStreamContext ctx)
{
    using value_t = value_t<iterator_value<IN> >;

    return nppiFilterGauss<size<value_t>()>(
		get_element_ptr(std::cbegin(*in)),
		stride(in)*sizeof(value_t),
		get_element_ptr(std::begin(*out)),
		stride(out)*sizeof(value_t),
		{int(std::distance(std::cbegin(*in), std::cend(*in))),
		 int(std::distance(in, ie))},
		mask_size, ctx);
}

#define NPP_FILTER(filter, type, nchannels)				\
    NPP_FUNC(filter<nchannels>(const type* src, int src_step,		\
			       type* dst, int dst_step,			\
			       NppiSize roi_size,			\
			       NppiMaskSize mask_size),			\
	     filter,							\
	     (src, src_step, dst, dst_step, roi_size, mask_size),	\
	     type, nchannels, false, true, false, false)		\
    NPP_FUNC(filter<nchannels>(const type* src, int src_step,		\
			       type* dst, int dst_step,			\
			       NppiSize roi_size,			\
			       NppiMaskSize mask_size,			\
			       NppStreamContext ctx),			\
	     filter,							\
	     (src, src_step, dst, dst_step, roi_size, mask_size, ctx),	\
	     type, nchannels, false, true, false, true)

NPP_FILTER(nppiFilterGauss, uint8_t, 1)
NPP_FILTER(nppiFilterGauss, uint8_t, 3)
NPP_FILTER(nppiFilterGauss, uint8_t, 4)
NPP_FILTER(nppiFilterGauss, uint16_t, 1)
NPP_FILTER(nppiFilterGauss, uint16_t, 3)
NPP_FILTER(nppiFilterGauss, uint16_t, 4)
NPP_FILTER(nppiFilterGauss, int16_t, 1)
NPP_FILTER(nppiFilterGauss, int16_t, 3)
NPP_FILTER(nppiFilterGauss, int16_t, 4)
NPP_FILTER(nppiFilterGauss, float, 1)
NPP_FILTER(nppiFilterGauss, float, 3)
NPP_FILTER(nppiFilterGauss, float, 4)

#define NPP_WATERSHED(type)						\
    NPP_FUNC(watershed(type* image, int image_step,			\
		       uint32_t* label, int label_step,			\
		       NppiNorm norm,					\
		       NppiWatershedSegmentationBoundaryType boundary_type, \
		       NppiSize roi_size, uint8_t* buffer),		\
	     watershed,							\
	     (image, image_step, label, label_step, norm, boundary_type, \
	      roi_size, buffer),					\
	     type, 1, false, true, false, false)		\
    NPP_FUNC(filter<nchannels>(const type* src, int src_step,		\
			       type* dst, int dst_step,			\
			       NppiSize roi_size,			\
			       NppiMaskSize mask_size,			\
			       NppStreamContext ctx),			\
	     filter,							\
	     (src, src_step, dst, dst_step, roi_size, mask_size, ctx),	\
	     type, nchannels, false, true, false, true)


}	// namespace cuda
}	// namespace TU
