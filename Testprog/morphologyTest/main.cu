/*
 *  $Id: main.cu,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/BoxFilter.h"
#include "TU/cu/Array++.h"
#include "TU/cu/algorithm.h"
#include "TU/cu/functional.h"
#include "TU/cu/chrono.h"
#include "TU/cu/Morphology.h"
#include <thrust/functional.h>

namespace TU
{
template <class T> void
doJob(const Image<T>& in, size_t winRadius)
{
    cu::Array2<T>	in_d(in);
    cu::Array2<T>	out_d(in_d.nrow(), in_d.ncol());

    const cu::Morphology<T, cu::BlockTraits<128, 1> >	filter(winRadius, winRadius);
    filter.apply_debug(in_d.cbegin(), in_d.cend(), out_d.begin(),
		       thrust::maximum<T>(), 0);
    cudaDeviceSynchronize();
#if 0
    Profiler<cu::clock>	cuProfiler(1);
    constexpr size_t	NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)
    {
	cuProfiler.start(0);
	filter.apply(in_d.cbegin(), in_d.cend(), out_d.begin(),
		     thrust::maximum<T>(), 0);
	cuProfiler.nextFrame();
    }
    cuProfiler.print(std::cerr);
#endif
    const Image<T>	out(out_d);
    out.save(std::cout);
}

}	// namespace TU

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    size_t		winRadius = 1;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "w:")) != -1; )
	switch (c)
	{
	  case 'w':
	    winRadius = atoi(optarg);
	    break;
	}

    try
    {
	using value_type = float;

	TU::Image<value_type>	in;
	in.restore(std::cin);
	in.save(std::cout);

	TU::doJob(in, winRadius);
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
