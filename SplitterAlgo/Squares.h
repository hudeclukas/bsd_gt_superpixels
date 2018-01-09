#ifndef SQUARES_H
#define SQUARES_H

#include <opencv2/core/mat.hpp>
#include <map>

class SuperpixelSquares
{
public:
    SuperpixelSquares(const cv::Mat &image, const cv::Mat &labelmask, const int size, const int shift);
    ~SuperpixelSquares();
    void iterate();
    void getLabelContourMask(cv::Mat& mask) const;
    std::map<int, std::vector<cv::Mat>> getSquares();

private:
    int sq_size;
    int sq_shift;
    const cv::Mat& image_;
    const cv::Mat& inputlabelmask_;
    
    cv::Mat labelsContourMask;
    std::map<int, std::vector<cv::Mat>> obj_squares;

};

#endif // SQUARES_H
