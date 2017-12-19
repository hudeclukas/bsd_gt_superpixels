#ifndef SPLITTER_H
#define SPLITTER_H

#include <opencv2/core/mat.hpp>

struct SeedsOptions
{
    int number_of_superpixels = 400;
    int levels = 8;
    int prior = 2;
};

struct LscOptions
{
    int region_size = 20;
    float ratio = 0.05f;
    int conectivity_min_element = 10;
    int num_iterations = 30;
};

class Splitter
{
public:
    enum Algorithm
    {
        LSC, SEEDS
    };
    Splitter();
    ~Splitter();

    void superpixelAlgorithm(Algorithm algo);

    cv::Mat run(const cv::Mat& input, cv::Mat& output);
    void setRegionSize(const int value);
    void setRatio(const float value);
    void setConnectivityMinElement(const int value);
    void setNumberOfIterations(int value);

    int getRegionSize() const;
    float getRatio() const;
    int getConectivityMinElement() const;
    int getNumberOfIterations() const;

    void setNumberOfSuperpixels(int value);
    void setLevels(int value);
    void setPrior(int value);

    int getNumberOfSuperpixels();
    int getLevels();
    int getPrior();

    cv::Mat getLabels();
private:
    SeedsOptions seeds_options_;
    LscOptions lsc_options_;

    cv::Mat superpixelsLabels;
    
    Algorithm algo = LSC;
};
#endif // SPLITTER_H
