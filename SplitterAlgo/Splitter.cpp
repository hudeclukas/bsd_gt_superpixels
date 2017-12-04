#include "Splitter.h"

#include <opencv2/ximgproc/lsc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cvxip = cv::ximgproc;

Splitter::Splitter()
{
}

Splitter::~Splitter()
{
}

cv::Mat Splitter::run(const cv::Mat& in, cv::Mat& out)
{
    cv::Mat tmp;
    cv::GaussianBlur(in, tmp, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_REFLECT);
    cv::cvtColor(tmp, tmp, cv::COLOR_BGR2Lab);
    auto lsc = cvxip::createSuperpixelLSC(tmp, lsc_region_size, lsc_ratio);
    lsc->iterate(lsc_num_iterations);
    lsc->enforceLabelConnectivity(lsc_conectivity_min_element);
    cv::Mat mask;
    lsc->getLabelContourMask(mask, false);
    in.copyTo(out);
    out.setTo(cv::Scalar(255, 255, 255), mask);
    lsc->getLabels(superpixelsLabels);
    return superpixelsLabels;
}

void Splitter::setRegionSize(const int value)
{
    lsc_region_size = value;
}

void Splitter::setRatio(const float value)
{
    lsc_ratio = value;
}

void Splitter::setConnectivityMinElement(const int value)
{
    lsc_conectivity_min_element = value;
}

int Splitter::getRegionSize() const
{
    return lsc_region_size;
}

float Splitter::getRatio() const
{
    return lsc_ratio;
}

int Splitter::getConectivityMinElement() const
{
    return lsc_conectivity_min_element;
}

int Splitter::getNumberOfIterations() const
{
    return lsc_num_iterations;
}

cv::Mat Splitter::getLabels()
{
    return superpixelsLabels;
}

void Splitter::setNumberOfIterations(int value)
{
    this->lsc_num_iterations = value;
}
