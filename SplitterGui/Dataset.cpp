#include "Dataset.h"

#include <set>
#include <fstream>
#include <QMenu>
#include <QDir>

#include <map>
#include <opencv2/imgcodecs.hpp>


Superpixel::Pixel::Pixel(int row_, int col_, uchar r_, uchar g_, uchar b_)
{
    row = row_;
    col = col_;
    r = r_;
    g = g_;
    b = b_;
}

Superpixel::Pixel::Pixel(int row_, int col_, cv::Vec3b color)
{
    row = row_;
    col = col_;
    b = color[0];
    g = color[1];
    r = color[2];
}

void Superpixel::createSuperpixelMat()
{
    if (!pixels.empty())
    {
        int left = INT_MAX, up = INT_MAX, right = 0, down = 0;
        int r = 0, g = 0, b = 0;
        for (auto&& p : pixels)
        {
            if (p.col < left) left = p.col;
            if (p.row < up) up = p.row;
            if (p.col > right) right = p.col;
            if (p.row > down) down = p.row;
            b += p.b;
            g += p.g;
            r += p.r;
        }
        b /= pixels.size();
        g /= pixels.size();
        r /= pixels.size();
        cv::Vec3b color = {static_cast<uchar>(b),static_cast<uchar>(g),static_cast<uchar>(r)};
        superpixelMat = cv::Mat::zeros(down - up + 1 + 2, right - left + 1 + 2, CV_8UC3) + color;
        for (auto&& p : pixels)
        {
            superpixelMat.at<cv::Vec3b>(p.row - up + 1, p.col - left + 1) = {p.b,p.g,p.r};
        }
    }
}


Image::Image()
{
}

Image::Image(std::map<int, std::vector<cv::Mat>> obj_superpixels)
{
    for (auto it = obj_superpixels.begin(); it != obj_superpixels.end(); ++it)
    {
        Image::ImageObject iob;
        for (auto itt = it->second.begin(); itt != it->second.end(); ++itt)
        {
            Superpixel sp;
            sp.superpixelMat = *itt;
            iob.push_back(sp);
        }
        this->objects.push_back(iob);
    }
}

cv::Mat Dataset::readImage(QString path)
{
    lastLoadedImagePath = path;
    lastLoadedImage = cv::imread(path.toStdString(), cv::IMREAD_ANYCOLOR);
    return lastLoadedImage;
}

cv::Mat Dataset::image()
{
    return lastLoadedImage;
}

cv::Mat Dataset::segments()
{
    return lastLoadedSegment;
}

cv::Mat &Dataset::mask()
{
    return lastLabelsMask;
}

void Dataset::setSaveSuperpixelsMask(bool save)
{
    saveSuperpixelsMask = save;
}

void Dataset::writeObjectFile(Image image, QString fileName)
{
    QFileInfo fi(fileName);
    auto dir = fi.dir();
    if (!dir.exists())
    {
        dir.cdUp();
        dir.mkdir("train/");
        dir.mkdir("test/");
    }
    std::ofstream fo(fileName.toStdString(), std::ios_base::out | std::ios_base::binary);
    int objectsCount = image.objects.size();
    fo.write(reinterpret_cast<char*>(&objectsCount), sizeof objectsCount);
    for (auto && objects : image.objects)
    {
        int superpixelsCount = objects.size();
        if (superpixelsCount > 0)
        {
            fo.write(reinterpret_cast<char*>(&superpixelsCount), sizeof superpixelsCount);
            for (auto && superpixel : objects)
            {
                fo.write(reinterpret_cast<char*>(&superpixel.label), sizeof superpixel.label);
                auto rows = superpixel.superpixelMat.rows;
                auto cols = superpixel.superpixelMat.cols;
                fo.write(reinterpret_cast<char*>(&rows), sizeof rows);
                fo.write(reinterpret_cast<char*>(&cols), sizeof cols);
                for (auto row = 0; row < superpixel.superpixelMat.rows; row++)
                {
                    const auto smPtr = superpixel.superpixelMat.ptr<cv::Vec3b>(row);
                    for (auto col = 0; col < superpixel.superpixelMat.cols; col++)
                    {
                        fo << smPtr[col][0];
                        fo << smPtr[col][1];
                        fo << smPtr[col][2];
                    }
                }
            }
        }

    }
    fo.close();
}

void Dataset::buildObjectFileName(QString& fileName)
{
    auto imgData = matchedData[saveOptions.Image];
    fileName = saveOptions.Path + "/";
    if (imgData.type == ImageData::TRAIN)
    {
        fileName += "train/";
    }
    else if (imgData.type == ImageData::TEST)
    {
        fileName += "test/";
    }
    fileName += saveOptions.Prefix + "_" + saveOptions.Image + "_" + QString::number(saveOptions.Counter) + "." + saveOptions.Extension;
 
}

std::map<QString, ImageData> MatchImage2Segmentation(std::vector<QString>& images, std::vector<QString> segmentations, ImageData::Type type)
{
    std::map<QString, ImageData> matches;
    for (auto &imPath : images)
    {
        QFileInfo imFI = QFileInfo(imPath);
        ImageData id;
        id.name = imFI.baseName();
        id.absPath = imPath;
        id.type = type;
        matches[id.name] = id;
        for (auto its = segmentations.begin(); its != segmentations.end();)
        {
            QFileInfo seFI = QFileInfo(*its);
            QString seName = seFI.baseName();

            if (seName == id.name)
            {
                matches[id.name].groundTruths.push_back(*its);
                its = segmentations.erase(its);
            }
            else
            {
                ++its;
            }
        }
    }
    return matches;
}

