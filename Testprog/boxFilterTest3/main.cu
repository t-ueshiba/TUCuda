#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "TU/cu/BoxFilter.h"
#include <thrust/functional.h>

namespace TU
{
namespace cu
{
namespace device
{
  template <class T>
  struct bit_greater
  {
      using first_argument_type	 = T;
      using second_argument_type = T;
      using result_type		 = T;

      __device__ __forceinline__
      result_type	operator ()(const first_argument_type&  a,
				    const second_argument_type& b) const
			{
			    return a & ~b;
			}
  };
}	// namespace device

template <class T> void
doJob(const cv::Mat& in, cv::Mat& out, size_t winSize)
{
    using convolver_t	= device::extrema_finder<
			      device::extrema_value<device::bit_greater<T> > >;
    using filter_t	= BoxFilter2<convolver_t, BlockTraits<32, 16> >;

    Array2<T>		in_d(TU::Array2<T>(const_cast<T*>(in.ptr<T>()),
					   in.rows, in.cols));
    Array2<T>		out_d(in_d.nrow(), in_d.ncol());
    const filter_t	filter(winSize, winSize);

    filter.convolve(in_d.cbegin(), in_d.cend(), out_d.begin(), true);

    TU::Array2<T>(out.ptr<T>(), out.rows, out.cols) = out_d;
}

}	// namespace cuda
}	// namespace TU

int
main(int argc, char* argv[])
{
    using pixel_type = uchar;
    using label_type = int;

    size_t	winSize = 5;

    for (int c; (c = getopt(argc, argv, "w:")) != -1; )
	switch (c)
	{
	  case 'w':
	    winSize = atoi(optarg);
	    break;
	}

    if (optind == argc)
    {
	std::cerr << "Specify input image file!" << std::endl;
	return -1;
    }

    cv::Mat	in;
    in = cv::imread(argv[optind], 1);
    cv::cvtColor(in, in, cv::COLOR_RGB2GRAY);

    cv::Mat	bin;
    cv::threshold(in, bin, 0, 128, cv::THRESH_OTSU);

    cv::Mat	bout(bin.size(), CV_8U);
    TU::cu::doJob<uint8_t>(bin, bout, winSize);

  // Disaplay results.
    cv::imshow("Binarized", bin);
    cv::imshow("Convolved", bout);
    cv::waitKey();

    return 0;
}
