#include "Squares.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

SuperpixelSquares::SuperpixelSquares(const cv::Mat &image, const cv::Mat &labelmask, const int size, const int shift): image_(image),
                                                                                             inputlabelmask_(labelmask),
                                                                                             sq_size(size),
                                                                                             sq_shift(shift)
{
    labelsContourMask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
}

SuperpixelSquares::~SuperpixelSquares()
{
}

void SuperpixelSquares::iterate()
{
    for (auto row = 0; row < image_.rows; row += sq_shift)
    {
        for (auto col = 0; col < image_.cols; col += sq_shift)
        {
            cv::line(labelsContourMask, { 0,row }, { labelsContourMask.cols,row }, { 255,255,0 }, 1);
            cv::line(labelsContourMask, { col,0 }, { col,labelsContourMask.rows }, { 255,255,0 }, 1);

            const auto rx_f = row + sq_size;
            const auto cy_f = col + sq_size;
            if (rx_f > image_.rows) continue;
            if (cy_f > image_.cols) continue;

            int objLbl_1 = inputlabelmask_.at<int>(row, col);
            bool samelabel = true;
            
            cv::Mat patch(sq_size, sq_size, CV_8UC3);

            for (auto rx = row, r = 0; rx < rx_f; rx++, r++)
            {
                auto ptrI = image_.ptr<cv::Vec3b>(rx);
                auto ptrP = patch.ptr<cv::Vec3b>(r);
                auto ptrL = inputlabelmask_.ptr<int>(rx);
                for (auto cy = col, c = 0; cy < cy_f; cy++, c++)
                {
                    if (objLbl_1 != ptrL[cy])
                    {
                        samelabel = false;
                    }
                    else
                    {
                        ptrP[c] = ptrI[cy];
                    }
                }
                if (!samelabel)
                {
                    break;
                }
            }
            if (samelabel)
            {
                obj_squares[objLbl_1].push_back(patch);
            }
        }
    }
}

void SuperpixelSquares::getLabelContourMask(cv::Mat& mask) const
{
    labelsContourMask.copyTo(mask);
}

std::map<int, std::vector<cv::Mat>> SuperpixelSquares::getSquares()
{
    return obj_squares;
}
