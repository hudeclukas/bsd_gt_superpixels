#include "Splitter.h"

#include <opencv2/ximgproc/lsc.hpp>
#include <opencv2/ximgproc/seeds.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cvxip = cv::ximgproc;

Splitter::Splitter()
{
}

Splitter::~Splitter()
{
}

void Splitter::superpixelAlgorithm(Algorithm algo)
{
    this->algo = algo;
}

cv::Mat Splitter::run(const cv::Mat& in, cv::Mat& out)
{
    switch (algo)
    {
        case LSC:
        {
            cv::Mat tmp;
            cv::cvtColor(in, tmp, cv::COLOR_BGR2Lab);
            auto lsc = cvxip::createSuperpixelLSC(tmp, lsc_options_.region_size, lsc_options_.ratio);
            lsc->iterate(lsc_options_.num_iterations);
            lsc->enforceLabelConnectivity(lsc_options_.conectivity_min_element);
            cv::Mat mask;
            lsc->getLabelContourMask(mask, false);
            in.copyTo(out);
            out.setTo(cv::Scalar(255, 255, 0), mask);
            lsc->getLabels(superpixelsLabels);
            return superpixelsLabels;
        }
        case SEEDS:
        {
            cv::Mat tmp;
            in.copyTo(tmp);
            auto seeds = cv::ximgproc::createSuperpixelSEEDS(tmp.cols, tmp.rows, tmp.channels(), seeds_options_.number_of_superpixels, seeds_options_.levels, seeds_options_.prior);
            seeds->iterate(tmp, 10);
            cv::Mat mask;
            seeds->getLabelContourMask(mask, false);
            in.copyTo(out);
            out.setTo(cv::Scalar(255, 255, 0), mask);
            seeds->getLabels(superpixelsLabels);
            return superpixelsLabels;
        }
        default :
            return superpixelsLabels;
    }
}

void Splitter::setRegionSize(const int value)
{
    lsc_options_.region_size = value;
}

void Splitter::setRatio(const float value)
{
    lsc_options_.ratio = value;
}

void Splitter::setConnectivityMinElement(const int value)
{
    lsc_options_.conectivity_min_element = value;
}

int Splitter::getRegionSize() const
{
    return lsc_options_.region_size;
}

float Splitter::getRatio() const
{
    return lsc_options_.ratio;
}

int Splitter::getConectivityMinElement() const
{
    return lsc_options_.conectivity_min_element;
}

int Splitter::getNumberOfIterations() const
{
    return lsc_options_.num_iterations;
}

void Splitter::setNumberOfSuperpixels(int value)
{
    seeds_options_.number_of_superpixels = value;
}

void Splitter::setLevels(int value)
{
    seeds_options_.levels = value;
}

void Splitter::setPrior(int value)
{
    seeds_options_.prior = value;
}

int Splitter::getNumberOfSuperpixels()
{
    return seeds_options_.number_of_superpixels;
}

int Splitter::getLevels()
{
    return seeds_options_.levels;
}

int Splitter::getPrior()
{
    return seeds_options_.prior;
}

cv::Mat Splitter::getLabels()
{
    return superpixelsLabels;
}

void Splitter::setNumberOfIterations(int value)
{
    lsc_options_.num_iterations = value;
}
