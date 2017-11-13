/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"

namespace TU
{
template <class U, class S, class T> void
cudaJob(const Array2<S>& in, Array2<T>& out, size_t winSize)	;

template <class S, class T> void
cpuJob(const Array2<S>& in, Array2<T>& out, size_t winSize)	;
}

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;
#if 0
    using in_t	= u_char;
    using mid_t	= float;
    using out_t	= float;
#else
    using in_t	= RGBA;
    using mid_t = float4;
    using out_t	= RGBA;
#endif    
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
	Image<in_t>	in;
	in.restore(cin);				// 原画像を読み込む

      // GPUによって計算する．
	Image<out_t>	out(in.width(), in.height());
	cudaJob<mid_t>(in, out, winSize);
	out.save(cout);					// 結果画像をセーブ

      // CPUによって計算する．
	cpuJob(in, out, winSize);
	out.save(cout);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
