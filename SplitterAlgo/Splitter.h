#ifndef SPLITTER_H
#define SPLITTER_H

#include <opencv2/core/mat.hpp>
#include <map>

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

struct SquaresOptions
{
    int size = 20;
    int shift = 10;
    double shift_ = 0.5;
    void setShift(const double shift)
    {
        this->shift = size*shift > 1 ? shift * size : 1;
        shift_ = shift;
    }
};

class Splitter
{
public:
    enum Algorithm
    {
        LSC, SEEDS, SQUARES
    };
    Splitter();
    ~Splitter();

    void superpixelAlgorithm(Algorithm algo);
    Algorithm superpixelAlgorithm() const;

    cv::Mat run(const cv::Mat& input, cv::Mat& output, cv::Mat &labelsmask);
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

    void setSquareSize(int size);
    void setSquareShift(double shift);

    int getSquareSize();
    double getSquareShift();

    cv::Mat getLabels();
    std::map<int, std::vector<cv::Mat>> getSquares();
private:
    SeedsOptions seeds_options_;
    LscOptions lsc_options_;
    SquaresOptions squares_options_;


    cv::Mat superpixelsLabels;
    std::map<int, std::vector<cv::Mat>> squares;

    Algorithm algo = LSC;
};
#endif // SPLITTER_H
