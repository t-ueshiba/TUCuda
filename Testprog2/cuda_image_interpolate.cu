/*
 * $Id: cuda_image_interpolate.cu,v 1.3 2011-04-11 08:06:06 ueshiba Exp $
 */
#include "TU/CudaArray++.h"
#include "TU/Image++.h"
#include "cuda_image_interpolate_kernel.h"
#include <cutil.h>

namespace TU
{
void
interpolate(const Image<RGBA>& image0,
	    const Image<RGBA>& image1,
		  Image<RGBA>& image2)
{
    using namespace	std;

#ifdef PROFILE
  // timer
    u_int	timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
#endif
  // allocate device memory and copy host memory to them
    static CudaArray2<RGBA>	d_image0, d_image1,
				d_image2(image0.height(), image0.width());
    d_image0 = image0;
    d_image1 = image1;
    
  // setup execution parameters
    dim3  threads(16, 16, 1);
    dim3  blocks(image0.ncol()/threads.x, image0.nrow()/threads.y, 1);
    
  // execute the kernel
    interpolate_kernel<<<blocks, threads>>>((const RGBA*)d_image0,
					    (const RGBA*)d_image1,
					    (      RGBA*)d_image2,
					    d_image2.stride(), 0.5f);
    
  // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

  // copy result from device to host
    d_image2.write(image2);

#ifdef PROFILE
  // time
    CUT_SAFE_CALL(cutStopTimer(timer));
    cerr << "Processing time: " << cutGetTimerValue(timer) << " (ms)" << endl;
    CUT_SAFE_CALL(cutDeleteTimer(timer));
#endif
}

}
