/*
 *  $Id: main.cu,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"
#include "TU/cuda/functional.h"
#include "TU/cuda/chrono.h"
#include "TU/cuda/ExtremaFilter.h"
#include <thrust/functional.h>
#include <cstdlib>

namespace TU
{
template <class T> void
doJob(const Array2<T>& in, size_t winSize)
{
    const cuda::ExtremaFilter2<T>	extrema(winSize, winSize);
    const cuda::Array2<T>		in_d(in);
    cuda::Array2<T>			out_d(in_d.nrow(), in_d.ncol());
    cuda::Array2<short2>		pos_d(in_d.nrow(), in_d.ncol());
	
    extrema.extrema(in_d.cbegin(), in_d.cend(), out_d.begin(), pos_d.begin(),
		    thrust::greater<T>(), false);
    // extrema.convolve(in_d.cbegin(), in_d.cend(), out_d.begin(),
    // 		     thrust::greater<T>(), false);

    std::cerr << "--- out ---" << std::endl;
    Array2<T>	out(out_d);
    for (size_t v = 0; v < out.nrow(); ++v)
    {
	for (size_t u = 0; u < out.ncol(); ++u)
	    std::cerr << std::setw(3) << out[v][u];
	std::cerr << std::endl;
    }
    std::cerr << std::endl;

    Array2<short2>	pos(pos_d);
    for (size_t v = 0; v < out.nrow(); ++v)
    {
	std::cerr << "v=" << v << ':';
	for (size_t u = 0; u < out.ncol(); ++u)
	{
	    const auto	p = pos[v][u];
	    std::cerr << p;
	    out[v][u] = in[p.y][p.x];
	}
	std::cerr << std::endl;
    }
    std::cerr << std::endl;
    
    std::cerr << "--- out(peak) ---" << std::endl;
    for (size_t v = 0; v < out.nrow(); ++v)
    {
	for (size_t u = 0; u < out.ncol(); ++u)
	    std::cerr << std::setw(3) << out[v][u];
	std::cerr << std::endl;
    }
    std::cerr << std::endl;
}
}

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    size_t		winSize = 3;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "w:")) != -1; )
	switch (c)
	{
	  case 'w':
	    winSize = atoi(optarg);
	    break;
	}
    
    try
    {
	using value_type = float;
	
	TU::Array2<value_type>	in(8, 8);
	for (size_t i = 0; i < in.nrow(); ++i)
	    for (size_t j = 0; j < in.ncol(); ++j)
		in[i][j] = rand() % 20;

	std::cerr << "--- in ---" << std::endl;
	for (size_t v = 0; v < in.nrow(); ++v)
	{
	    for (size_t u = 0; u < in.ncol(); ++u)
		std::cerr << std::setw(3) << in[v][u];
	    std::cerr << std::endl;
	}
	std::cerr << std::endl;
	
	doJob(in, winSize);
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
