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
  \file		chrono.h
  \brief	GPUクロックの定義と実装
*/
#pragma once

#include <chrono>
#include <cuda_runtime.h>

namespace TU
{
namespace cu
{
//! GPUデバイスのクロックを表すクラス
class clock
{
  public:
    typedef float				rep;		//!< 表現
    typedef std::milli				period;		//!< 解像度
    typedef std::chrono::duration<rep, period>	duration;	//!< 時間
    typedef std::chrono::time_point<clock>	time_point;	//!< 時刻

  private:
    class Event
    {
      public:
			Event()
			{
			    cudaEventCreate(&_epoch);
			    cudaEventCreate(&_now);
			    cudaEventRecord(_epoch, 0);
			}
			~Event()
			{
			    cudaEventDestroy(_now);
			    cudaEventDestroy(_epoch);
			}
	rep		now() const
			{
			    cudaEventRecord(_now, 0);
			    cudaEventSynchronize(_now);
			    rep	time;
			    cudaEventElapsedTime(&time, _epoch, _now);
			    return time;
			}

      private:
	cudaEvent_t	_epoch;
	cudaEvent_t	_now;
    };

  public:
  //! 現在の時刻を返す.
    static time_point	now() noexcept
			{
			    return time_point(duration(_event.now()));
			}

  private:
    static Event	_event;
};

}	// namespace cu
}	// namespace TU
