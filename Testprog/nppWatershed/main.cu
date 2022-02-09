/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cu/Array++.h"
#include "TU/cu/algorithm.h"
#include "TU/cu/chrono.h"
#include "TU/cu/vec.h"
#include "TU/cu/npp.h"

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	TU;

    using pixel_t	= uint8_t;
  //using pixel_t	= RGB;
    
    try
    {
	Image<pixel_t>	image;
	image.restore(std::cin);		// 原画像を読み込む

	cu::Array2<pixel_t>	in_d(image), out_d(in_d.nrow(), in_d.ncol());
	cu::nppiFilterGauss(in_d.cbegin(), in_d.cend(), out_d.begin(),
			    NPP_MASK_SIZE_15_X_15);

	Profiler<cu::clock>	cuProfiler(1);
	constexpr size_t	NITER = 1000;

	for (size_t n = 0; n < NITER; ++n)
	{
	    cuProfiler.start(0);
	    cu::nppiFilterGauss(in_d.cbegin(), in_d.cend(), out_d.begin(),
				NPP_MASK_SIZE_15_X_15);
	    cuProfiler.nextFrame();
	}
	cuProfiler.print(std::cerr);

	image = out_d;
	image.save(std::cout);			// 結果画像をセーブ
    }
    catch (const std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
