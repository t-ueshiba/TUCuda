/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"
#include "TU/cuda/chrono.h"
#include "TU/cuda/vec.h"
#include "TU/cuda/npp.h"

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	TU;

  //using pixel_t	= uint8_t;
    using pixel_t	= RGB;
    
    try
    {
	Image<pixel_t>	image;
	image.restore(std::cin);			// 原画像を読み込む

	cuda::Array2<pixel_t>	in_d(image), out_d(in_d.nrow(), in_d.ncol());
      //cuda::gauss_filter(in_d.cbegin(), in_d.cend(), out_d.begin());
	auto			roi = in_d(200, 300, 50, 250);
	TU::cuda::gauss_filter(roi.cbegin(), roi.cend(), out_d.begin(),
			       NPP_MASK_SIZE_15_X_15);
	image = out_d;
	image.save(std::cout);
    }
    catch (const std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
