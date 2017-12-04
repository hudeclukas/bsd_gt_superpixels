#ifndef SPLITTER_H
#define SPLITTER_H

#include <opencv2/core/mat.hpp>

class Splitter
{
public:
    Splitter();
    ~Splitter();

    cv::Mat run(const cv::Mat& input, cv::Mat& output);
    void setRegionSize(const int value);
    void setRatio(const float value);
    void setConnectivityMinElement(const int value);
    void setNumberOfIterations(int value);

    int getRegionSize() const;
    float getRatio() const;
    int getConectivityMinElement() const;
    int getNumberOfIterations() const;

    cv::Mat getLabels();
private:
    int lsc_region_size = 20;
    float lsc_ratio = 0.05f;
    int lsc_conectivity_min_element = 10;
    int lsc_num_iterations = 30;


    cv::Mat superpixelsLabels;


};
#endif // SPLITTER_H
