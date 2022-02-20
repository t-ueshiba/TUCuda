#include <opencv2/core.hpp>
#include <map>

namespace TU
{
namespace detail
{
  template <class L> L
  root(const L* labels, L label)
  {
      while (label > labels[label])
	  label = labels[label];
      return label;
  }

  template <class L> inline void
  merge(L* labels, L label0, L label1)
  {
      const auto root0 = root(labels, label0);
      const auto root1 = root(labels, label1);

      if (root0 < root1)
	  labels[root1] = root0;
      else if (root1 < root0)
	  labels[root0] = root1;
  }
}	// namespace detail

template <class T, class L> int
label_image(const cv::Mat& in, cv::Mat& out, bool set_bg)
{
    const auto	idx    = [in](int v, int u){ return v * in.cols + u; };
    const auto	labels = out.ptr<L>();

  // Pass 1: create equivalence trees
    for (int v = 0; v < in.rows; ++v)
	for (int u = 0; u < in.cols; ++u)
	{
	    const L	label = idx(v, u);
	    labels[label] = label;

	    if (v > 0 && in.at<T>(v-1, u) == in.at<T>(v, u))
		detail::merge(labels, labels[idx(v-1, u)], label);

	    if (u > 0 && in.at<T>(v, u-1) == in.at<T>(v, u))
		detail::merge(labels, labels[idx(v, u-1)], label);
	}

  // Pass 2: flatten equivalence trees
    for (int v = 0; v < out.rows; ++v)
	for (int u = 0; u < out.cols; ++u)
	    out.at<L>(v, u) = detail::root(labels, v * out.cols + u);

  // Pass 3:
    if (set_bg)
    {
	for (int v = 0; v < out.rows; ++v)
	    for (int u = 0; u < out.cols; ++u)
		if (in.at<T>(v, u) == 0)
		    out.at<L>(v, u) = -1;
    }
    
  // Pass 4: relabel extracted regions sequentially
    L			relabel = 0;
    std::map<L, L>	lookup;
    for (int v = 0; v < out.rows; ++v)
	for (int u = 0; u < out.cols; ++u)
	    if (out.at<L>(v, u) >= 0)
	    {
		const auto	result = lookup.emplace(out.at<L>(v, u),
							relabel);
		out.at<L>(v, u) = (result.first)->second;
		if (result.second)
		    ++relabel;
	    }
    
    return lookup.size();
}
}
    
