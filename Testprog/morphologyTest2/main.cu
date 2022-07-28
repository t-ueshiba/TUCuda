/*
 *  $Id: main.cu,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include <limits>
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cu/Array++.h"
#include "TU/cu/algorithm.h"
#include "TU/cu/functional.h"
#include "TU/cu/chrono.h"
#include "TU/cu/Morphology.h"
#include <thrust/functional.h>

namespace TU
{
template <class T> void
cudaJob(const Array2<T>& in, size_t winRadius)
{
    cu::Array2<T>	in_d(in);
    cu::Array2<T>	out_d(in_d.nrow(), in_d.ncol());

    const cu::Morphology<T, cu::BlockTraits<8, 2> >	filter(winRadius,
							       winRadius);
    filter.apply_debug(in_d.cbegin(), in_d.cend(), out_d.begin(),
		       thrust::maximum<T>(), 0);

    cu::Array2<T>	out(out_d);
    std::cerr << "--- out ---\n" << out;
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

	std::cerr << "Input array2 >> ";
	for (TU::Array2<value_type> in; std::cin >> in; )
	{
	    std::cerr << "--- array2 ---\n" << in << std::endl;

	    cudaJob(in, winRadius);

	    std::cerr << "Input array2 >> ";
	}
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
