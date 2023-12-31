/*
 *  $Id: main.cu,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/cu/Texture.h"
#include "TU/cu/vec.h"
#include "TU/cu/chrono.h"

//#define OP	cu::det3x3
//#define OP	cu::laplacian3x3
//#define OP	cu::sobelAbs3x3
#define OP	cu::maximal3x3
//#define OP	cu::minimal3x3

namespace TU
{
namespace cu
{
template <class T> Rigidity<T, 2>
createRigidity(T u0, T v0, T theta)
{
    const auto	c = std::cos(theta);
    const auto	s = std::sin(theta);

    return {{{c, -s}, {s,  c}}, {(1 - c)*u0 + s*v0, (1 - c)*v0 - s*u0}};
}

}
}

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;
#if 1
    using pixel_t	= float;
    using mid_t		= float;
#else
    using pixel_t	= RGBA;
    using mid_t		= float4;
#endif
    using value_t	= float;

    value_t		u0 = -1;
    value_t		v0 = -1;
    value_t		theta = M_PI/6;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "u:v:t:")) != -1; )
	switch (c)
	{
	  case 'u':
	    u0 = atof(optarg);
	    break;
	  case 'v':
	    v0 = atof(optarg);
	    break;
	  case 't':
	    theta = atof(optarg)*M_PI/180;
	    break;
	}

    try
    {
	Image<pixel_t>	in;
	in.restore(cin);				// 原画像を読み込む
	in.save(cout);					// 原画像をセーブ

	if (u0 < 0)
	    u0 = in.width()/2;
	if (v0 < 0)
	    v0 = in.height()/2;

      // GPUによって計算する．
	cu::Array2<mid_t>	in_d(in);
	cu::Array2<mid_t>	out_d(in.nrow(), in.ncol());
	const auto		op = cu::createRigidity(u0, v0, theta);
	cu::warp(in_d, out_d.begin(), op);
	cudaDeviceSynchronize();

	Profiler<cu::clock>	cuProfiler(1);
	constexpr size_t	NITER = 1000;
	for (size_t n = 0; n < NITER; ++n)		// フィルタリング
	{
	    cuProfiler.start(0);
	    cu::warp(in_d, out_d.begin(), op);
	    cuProfiler.nextFrame();
	}
	cuProfiler.print(std::cerr);

	Image<pixel_t>	out(out_d);
	out.save(cout);					// 結果画像をセーブ
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
