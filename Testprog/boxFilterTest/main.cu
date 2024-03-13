/*
 *  $Id: main.cu,v 1.1 2012-08-30 00:13:51 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Profiler.h"
#include "TU/BoxFilter.h"
#include "TU/cu/Array++.h"
#include "TU/cu/algorithm.h"
#include "TU/cu/functional.h"
#include "TU/cu/chrono.h"
#include "TU/cu/BoxFilter.h"
#include <thrust/functional.h>

namespace TU
{
template <class CONVOLVER, class IN, class OUT> void
cudaJob(IN in, IN ie, OUT out, size_t winSize, bool shift)
{
    const cu::BoxFilter2<CONVOLVER, cu::BlockTraits<32, 16> >
			filter(winSize, winSize);
    filter.convolve(in, ie, out, shift);
    cudaDeviceSynchronize();

    Profiler<cu::clock>	cuProfiler(1);
    constexpr size_t	NITER = 1000;
    for (size_t n = 0; n < NITER; ++n)
    {
	cuProfiler.start(0);
	filter.convolve(in, ie, out, shift);
	cuProfiler.nextFrame();
    }
    cuProfiler.print(std::cerr);
}

template <class S, class T> void
box_convolver_test(const Image<T>& in, size_t winSize)
{
    using convolver_t = cu::device::box_convolver<S>;

  // GPUによって計算する．
    std::cerr << "=== box_convolver_test ===\n[GPU] ";
    const cu::Array2<T>	in_d(in);
    cu::Array2<T>	out_d(in_d.nrow(), in_d.ncol());
    cudaJob<convolver_t>(in_d.cbegin(), in_d.cend(), out_d.begin(),
			 winSize, false);
    Image<T>		out(out_d);
    out *= float(1)/float(winSize*winSize);
    out.save(std::cout);				// 結果画像をセーブ

  // CPUによって計算する．
    std::cerr << "[CPU] ";
    BoxFilter2<T>	box(winSize, winSize);
    Profiler<>		profiler(1);
    constexpr size_t	NITER = 100;
    for (size_t n = 0; n < NITER; ++n)
    {
	profiler.start(0);
	box.convolve(in.cbegin(), in.cend(), out.begin(), false);
	profiler.nextFrame();
    }
    profiler.print(std::cerr);

    out *= float(1)/float(winSize*winSize);
    out.save(std::cout);				// 結果画像をセーブ
}

template <class T> void
extrema_value_test(const Image<T>& in, size_t winSize)
{
    using convolver_t = cu::device::extrema_finder<
			    cu::device::extrema_value<thrust::less<T> > >;

  // GPUによって計算する．
    std::cerr << "=== extrema_value_test ===\n[GPU] ";
    const cu::Array2<T>	in_d(in);
    cu::Array2<T>	out_d(in_d.nrow(), in_d.ncol());
    cudaJob<convolver_t>(in_d.cbegin(), in_d.cend(), out_d.begin(),
			 winSize, false);
    const Image<T>	out(out_d);
    out.save(std::cout);				// 結果画像をセーブ

  // CPUによって計算する．
    Image<T>	outGold(in.width(), in.height());
    for (size_t v = 0; v < in.height() - winSize + 1; ++v)
	for (size_t u = 0; u < in.width() - winSize + 1; ++u)
	{
	    T	minval = 1000;

	    for (size_t vv = v ; vv < v + winSize; ++vv)
		for (size_t uu = u ; uu < u + winSize; ++uu)
		    if (in[vv][uu] <= minval)
			minval = in[vv][uu];
	    outGold[v][u] = minval;
	}
    outGold.save(std::cout);				// 結果画像をセーブ

  // 結果を比較する．
    const int	V = 160;
    for (u_int u = 1; u < out.width() - 1; ++u)
	if (out[V][u] != outGold[V][u])
	{
	    std::cerr << ' ' << u << ":(" << out[V][u] << ',' << outGold[V][u]
		      << ')' << std::endl;
	    std::cerr << slice<3, 3>(in, u-1, V-1);
	}
}

template <class T> void
extrema_position_test(const Image<T>& in, size_t winSize)
{
    using convolver_t = cu::device::extrema_finder<
			    cu::device::extrema_position<thrust::less<T> > >;

  // GPUによって計算する．
    std::cerr << "=== extrema_position_test ===\n[GPU] ";
    const cu::Array2<T>			in_d(in);
    cu::Array2<cu::vec<int, 2> >	pos_d(in_d.nrow(), in_d.ncol());
    cudaJob<convolver_t>(in_d.cbegin(), in_d.cend(), pos_d.begin(),
			 winSize, true);
    const Array2<cu::vec<int, 2> >	pos(pos_d);

  // 結果をcheckする．
    const auto	offsetL = winSize/2;
    const auto	offsetR = winSize - offsetL;
    for (size_t v = offsetL; v < pos.nrow() - offsetR; ++v)
    	for (size_t u = offsetL; u < pos.ncol() - offsetR; ++u)
    	{
    	    const auto	p = pos[v][u];
	    const auto	minval = in[p.y][p.x];

	    for (size_t vv = v - offsetL; vv < v + offsetR; ++vv)
		for (size_t uu = u - offsetL; uu < u + offsetR; ++uu)
		    if (in[vv][uu] < minval)
		    {
			std::cerr << "minval: " << minval
				  << "@(" << p.x << ' ' << p.y << ')'
				  << std::endl;
			for (size_t vvv = v - offsetL; vvv < v + offsetR;
			     ++vvv)
			{
			    for (size_t uuu = u - offsetL; uuu < u + offsetR;
				 ++uuu)
				std::cerr << ' ' << in[vvv][uuu];
			    std::cerr << std::endl;
			}

			uu = u + offsetR - 1;
			vv = v + offsetR - 1;
		    }
    	}
}

template <class T> void
extrema_value_position_test(const Image<T>& in, size_t winSize)
{
    using convolver_t = cu::device::extrema_finder<
			    cu::device::extrema_value_position<
				thrust::less<T> > >;

  // GPUによって計算する．
    std::cerr << "=== extrema_value_position_test ===\n[GPU] ";
    const cu::Array2<T>			in_d(in);
    cu::Array2<T>			out_d(in_d.nrow(), in_d.ncol());
    cu::Array2<cu::vec<int, 2> >	pos_d(in_d.nrow(), in_d.ncol());

    cudaJob<convolver_t>(in_d.cbegin(), in_d.cend(),
			 cu::make_range_iterator(
			     thrust::make_zip_iterator(out_d.begin()->begin(),
						       pos_d.begin()->begin()),
			     cu::stride(out_d.begin(), pos_d.begin()),
			     out_d.size()),
			 winSize, true);
    Image<T>		out(out_d);
    out.save(std::cout);				// 結果画像をセーブ

    auto	s = cu::stride(out_d.begin(), pos_d.begin());
    std::cerr << demangle<decltype(s)>()
	      << '[' << cuda::std::get<0>(s)
	      << ',' << cuda::std::get<1>(s) << ']' << std::endl;
    

  // CPUによって計算する．
    Array2<cu::vec<int, 2> >	pos(pos_d);
    Image<T>			out2(in.width(), in.height());
    for (size_t v = 0; v < out2.nrow(); ++v)
    {
    	for (size_t u = 0; u < out2.ncol(); ++u)
    	{
    	    const auto	p = pos[v][u];
    	    out2[v][u] = in[p.y][p.x];
    	}
    }
    out2.save(std::cout);				// 結果画像をセーブ

  // 結果を比較する．
    const int	V = 160;
    for (u_int u = 1; u < out.width() - 1; ++u)
	if (out[V][u] != out2[V][u])
	    std::cerr << ' ' << u << ":(" << out[V][u] << ',' << out2[V][u]
		      << ')' << std::endl;
}

}	// namespace TU

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    size_t		winSize = 7;
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

	TU::Image<value_type>	in;
	in.restore(std::cin);
	in.save(std::cout);

	TU::box_convolver_test<float>(in, winSize);
	TU::extrema_value_test(in, winSize);
	TU::extrema_position_test(in, winSize);
	TU::extrema_value_position_test(in, winSize);
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
