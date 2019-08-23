/*
 *  $Id$
 */
#include "TU/GuidedFilter.h"
#include "TU/Profiler.h"

namespace TU
{
template <class S, class T, class U> void
cpuJob(const Array2<S>& in, const Array2<S>& guide,
       Array2<T>& out, size_t winSize, U epsilon)
{
    GuidedFilter2<U>	guidedFilter(winSize, winSize, epsilon);
    Profiler<>		profiler(1);
    constexpr size_t	NITER = 100;
    for (size_t n = 0; n < NITER; ++n)
    {
	profiler.start(0);
	guidedFilter.convolve(in.cbegin(), in.cend(),
			      guide.cbegin(), guide.cend(), out.begin());
	profiler.nextFrame();
    }
    profiler.print(std::cerr);
}

template void
cpuJob(const Array2<u_char>& in, const Array2<u_char>& guide,
       Array2<float>& out, size_t winSize, float epsilon)		;
  /*
template void
cudaJob<float4>(const Array2<RGBA>& in, Array2<RGBA>& out, size_t winSize);
  */
}	// namespace TU
