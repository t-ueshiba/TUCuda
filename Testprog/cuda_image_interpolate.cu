/*
 * $Id: cuda_image_interpolate.cu,v 1.3 2009-04-20 01:16:37 ueshiba Exp $
 */
#include "TU/CudaDeviceMemory.h"
#include "TU/Image++.h"

#include "cuda_image_interpolate_kernel.h"

namespace TU
{
/*
 *  本当は引数の型を (const) Image<RGBA>& としたいところであるが，
 *  CUDA-2.1の nvcc でコンパイルしたC++関数は仮想メンバ関数を持つ
 *  クラスのオブジェクトを正しく扱えないようである．
 */ 
void
interpolate(const Array2<ImageLine<RGBA> >& image0,
	    const Array2<ImageLine<RGBA> >& image1,
		  Array2<ImageLine<RGBA> >& image2)
{
    using namespace	std;

  // timer
    u_int	timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));

  // allocate device memory and copy host memory to them
    CudaDeviceMemory2<uchar4>	d_image0, d_image1, d_image2;
    d_image0.readFrom(image0);
    d_image1.readFrom(image1);
    d_image2.resize(image0.nrow(), image0.ncol());
    
  // setup execution parameters
    dim3  threads(16, 16, 1);
    dim3  blocks(image0.ncol()/threads.x, image0.nrow()/threads.y, 1);
    cerr << blocks.x << 'x' << blocks.y << " blocks..." << endl;
    
  // execute the kernel
    cerr << "Let's go!" << endl;
    for (int i = 0; i < 1000; ++i)
	interpolate_kernel<<<blocks, threads>>>((const uchar4*)d_image0,
						(const uchar4*)d_image1,
						(      uchar4*)d_image2,
						d_image2.ncol(),
						d_image2.nrow(), 0.5f);
    cerr << "Returned!" << endl;
    
  // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

  // copy result from device to host
    d_image2.writeTo(image2);

  // time
    CUT_SAFE_CALL(cutStopTimer(timer));
    cerr << "Processing time: " << cutGetTimerValue(timer) << " (ms)" << endl;
    CUT_SAFE_CALL(cutDeleteTimer(timer));

#if 0	
  // compute reference solution
    float* reference = (float*) malloc(mem_size);
    computeGold(reference, h_idata, num_threads);

  // check result
    if (cutCheckCmdLineFlag(argc, ( const char** ) argv, "regression" ))
    {
      // write file for regression test
        CUT_SAFE_CALL(cutWriteFilef( "./data/regression.dat",
				     h_odata, num_threads, 0.0));
    }
    else
    {
        CUTBoolean res = cutComparef(reference, h_odata, num_threads);
        printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    }
#endif
}

}
