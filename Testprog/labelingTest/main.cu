#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "TU/cu/Labeling.h"
#include "labeling.h"

namespace TU
{
namespace cu
{
template <class T, class L> int
label_image(const cv::Mat& image, cv::Mat& labels, bool set_bg)
{
    Labeling<>	labeling;
    Array2<T>	image_d(TU::Array2<T>(const_cast<T*>(image.ptr<T>()),
				   image.rows, image.cols));
    Array2<L>	labels_d(image_d.nrow(), image_d.ncol());

    labeling.label(image_d.cbegin(), image_d.cend(), labels_d.begin(),
		   thrust::equal_to<T>());
    if (set_bg)
      labeling.set_background(image_d.cbegin(), image_d.cend(),
			      labels_d.begin(),
			      [] __device__ (auto pixel)
			      { return pixel == 0; });
    labeling.print(std::cerr);
    
    TU::Array2<L>(labels.ptr<L>(), labels.rows, labels.cols) = labels_d;

  // Relabel extracted regions sequentially
    L			relabel = 0;
    std::map<L, L>	lookup;
    for (int v = 0; v < labels.rows; ++v)
	for (int u = 0; u < labels.cols; ++u)
	    if (labels.at<L>(v, u) >= 0)
	    {
		const auto	result = lookup.emplace(labels.at<L>(v, u),
							relabel);
		labels.at<L>(v, u) = (result.first)->second;
		if (result.second)
		    ++relabel;
	    }
    
    return lookup.size();
}
}	// namespace cuda
    
template <class T> cv::Mat
color_encode(const cv::Mat& labels, int nlabels)
{
    std::vector<cv::Vec3b>	colors(1 + nlabels);

    colors[0] = cv::Vec3b(0, 0, 0);
    for (int i = 1; i < colors.size(); ++i)
        colors[i] = cv::Vec3b(32 + rand() % 224,
			      32 + rand() % 224,
			      32 + rand() % 224);
    
    cv::Mat	result(labels.size(), CV_8UC3);
    for (int v = 0; v < result.rows; ++v)
	for (int u = 0; u < result.cols; ++u)
	    result.at<cv::Vec3b>(v, u) = colors[1 + labels.at<T>(v, u)];

    return result;
}
}	// namespace TU

int
main(int argc, char* argv[])
{
    using pixel_type = uchar;
    using label_type = int;
    
    int		algorithm = 0;
    bool	set_bg = false;
    
    for (int c; (c = getopt(argc, argv, "cgb")) != -1; )
	switch (c)
	{
	  case 'c':
	    algorithm = 1;
	    break;
	  case 'g':
	    algorithm = 2;
	    break;
	  case 'b':
	    set_bg = true;
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
    cv::threshold(in, bin, 0, 255, cv::THRESH_OTSU);

    cv::Mat	labels(bin.size(), CV_32S);
    int		nlabels;
    switch (algorithm)
    {
      default:
	nlabels = TU::label_image<pixel_type, label_type>(bin, labels, set_bg);
	break;
      case 1:
	nlabels = cv::connectedComponents(bin, labels, 4);
	break;
      case 2:
	nlabels = TU::cu::label_image<pixel_type, label_type>(bin, labels,
							      set_bg);
	break;
    }
    std::cerr << nlabels << " regions found." << std::endl;
    
    const auto	encoded = TU::color_encode<label_type>(labels, nlabels);
    
  // Disaplay results.
    cv::imshow("Binarized", bin);
    cv::imshow("Labels", encoded);
    cv::waitKey();

    return 0;
}
