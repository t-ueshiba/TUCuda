/*
 *  $Id: main.cc,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"

namespace TU
{
template <class S, class T, class U> void
cudaJob(const Array2<S>& in, const Array2<S>& guide,
	Array2<T>& out, size_t winSize, U epsilon)			;
}

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;

    using in_t	= u_char;
    using mid_t	= float;
    using out_t	= float;

    size_t		winSize = 3;
    mid_t		epsilon = 0.01;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "w:e:")) != -1; )
	switch (c)
	{
	  case 'w':
	    winSize = atoi(optarg);
	    break;
	  case 'e':
	    epsilon = atof(optarg);
	    break;
	}
    
    try
    {
	Image<in_t>	in, guide;
	in.restore(cin);				// 原画像を読み込む
	guide.restore(cin);
	
      // GPUによって計算する．
	Image<out_t>	out(in.width(), in.height());
	cudaJob(in, guide, out, winSize, epsilon);
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
