/*
 *  \file cpuJob.cc
 */
#include "TU/BoxFilter.h"
#include "TU/Image++.h"
#include "TU/Profiler.h"

namespace TU
{
template <class S, class T> void
cpuJob(const Array2<S>& in, Array2<T>& out, size_t winSize, bool shift)
{
    BoxFilter2<T>	box(winSize, winSize);
    Profiler<>		profiler(1);
    constexpr size_t	NITER = 100;
    for (size_t n = 0; n < NITER; ++n)
    {
	profiler.start(0);
	box.convolve(in.cbegin(), in.cend(), out.begin(), shift);
	profiler.nextFrame();
    }

    profiler.print(std::cerr);

    out *= float(1)/float(winSize*winSize);
}

template void
cpuJob(const Array2<u_char>& in, Array2<float>& out,
       size_t winSize, bool shift);
template void
cpuJob(const Array2<RGBA>& in, Array2<RGBA>& out, size_t winSize, bool shift);
    
}
