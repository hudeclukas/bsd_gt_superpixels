#ifndef CONVERTER_H
#define CONVERTER_H

#include <QtGui/qimage.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/shape/hist_cost.hpp>


inline QImage Mat2QImage(cv::Mat src)
{ 
    switch (src.type())
    {
        // 8-bit, 4 channel
        case CV_8UC4:
        {
            QImage dest(src.data, src.cols, src.rows, src.step, QImage::Format_RGB32);

            return dest.copy();
        }

        // 8-bit, 3 channel
        case CV_8UC3:
        {
            QImage dest(src.data, src.cols, src.rows, src.step, QImage::Format_RGB888);

            return dest.copy();
        }

        // 16-bit, 1 channel
        case CV_16U:
        {
            QImage dest(src.data, src.cols, src.rows, src.step, QImage::Format_RGB16);

            return dest.copy();
        }

        // 8-bit, 1 channel
        case CV_8U:
        {
            cv::Mat dst;
            QImage dest(dst.data, dst.cols, dst.rows, dst.step, QImage::Format_RGB888);
            return dest.copy();
        }

        default:

            break;
    }
    return QImage(src.cols, src.rows, QImage::Format_Mono);
}
#endif // CONVERTER_H
