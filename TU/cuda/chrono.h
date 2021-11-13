/*
 *  $Id$
 */
/*!
  \file		chrono.h
  \brief	GPUクロックの定義と実装
*/
#pragma once

#include <chrono>
#include <cuda_runtime.h>

namespace TU
{
namespace cuda
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

}	// namespace cuda
}	// namespace TU
