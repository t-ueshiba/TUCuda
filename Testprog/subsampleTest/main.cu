/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cu/Array++.h"
#include "TU/cu/algorithm.h"
#include "TU/cu/functional.h"
#include "TU/cu/chrono.h"

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;

    typedef float	pixel_t;

    try
    {
	Image<pixel_t>	image;
	image.restore(cin);				// 原画像を読み込む
	image.save(cout);

	cu::Array2<pixel_t>	in_d(image),
				out_d(in_d.nrow()/2, in_d.ncol()/2);
	cu::subsample(in_d.cbegin(), in_d.cend(), out_d.begin(),
		      cu::identity1x1());
	cudaDeviceSynchronize();

	Profiler<cu::clock>	cuProfiler(1);
	constexpr size_t	NITER = 1000;
	for (size_t n = 0; n < NITER; ++n)
	{
	    cuProfiler.start(0);
	    cu::subsample(in_d.cbegin(), in_d.cend(), out_d.begin(),
			  cu::identity1x1());
	    cuProfiler.nextFrame();
	}
	cuProfiler.print(cerr);

	image = out_d;
	image.save(cout);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
