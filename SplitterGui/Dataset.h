#ifndef DATASET_H
#define DATASET_H

#include <QObject>
#include <QMenu>
#include <QFileInfo>
#include <QDir>

#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <fstream>

struct ImageData
{
    enum Type
    {
        TRAIN, TEST
    };

    QString name;
    QString absPath;
    std::vector<QString> groundTruths;
    Type type;
};

struct Superpixel
{
    struct Pixel
    {
        Pixel(int row_, int col_, uchar r_, uchar g_, uchar b_)
        {
            row = row_;
            col = col_;
            r = r_;
            g = g_;
            b = b_;
        }
        Pixel(int row_, int col_, cv::Vec3b color)
        {
            row = row_;
            col = col_;
            b = color[0];
            g = color[1];
            r = color[2];
        }
        int row, col;
        uchar r, g, b;
    };

    void createSuperpixelMat()
    {
        if (!pixels.empty())
        {
            int left = INT_MAX, up = INT_MAX, right = 0, down = 0;
            int r=0, g=0, b=0;
            for (auto && p : pixels)
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
            cv::Vec3b color = { static_cast<uchar>(b),static_cast<uchar>(g),static_cast<uchar>(r) };
            superpixelMat = cv::Mat::zeros(down - up + 1 + 2, right - left + 1 + 2, CV_8UC3) + color;
            for (auto && p : pixels)
            {
                superpixelMat.at<cv::Vec3b>(p.row - up + 1, p.col - left + 1) = { p.b,p.g,p.r };
            }
        }
    }

    std::vector<Pixel> pixels;
    uint label;
    int invalidCount;
    bool isValid = true;
    cv::Mat superpixelMat;
};

struct Image
{
    Image();
    Image(std::map<int, std::vector<cv::Mat>> obj_superpixels);
    typedef std::vector<Superpixel> ImageObject;
    
    QString name;
    std::vector<ImageObject> objects;
};

class Dataset : public QObject
{
    Q_OBJECT

public:

    virtual QMenu * getDatasetMenu() = 0;
    virtual std::map<QString, ImageData> getLoadedData() = 0;
    virtual cv::Mat getSegmentedImage(QString image, int segmentation = 0) = 0;
    virtual void resetData() = 0;
    virtual void saveSegment2SuperpixelLabels(cv::Mat image) = 0;
    virtual void saveSegment2SuperpixelLabels(std::map<int, std::vector<cv::Mat>> patches) = 0;

    cv::Mat readImage(QString path);
    cv::Mat image();
    cv::Mat segments();
    cv::Mat &mask();

    public slots:
    virtual void changeSavePattern() = 0;
    virtual void setSaveCounter(int value) = 0;

    signals:
    void saved(QString path);

protected:
    QString lastLoadedImagePath;
    cv::Mat lastLoadedImage;
    cv::Mat lastLoadedSegment;
    cv::Mat lastLabelsMask;
    cv::Mat lastEdgesMask;

    void writeObjectFile(Image image, QString fileName);
};

inline Image::Image()
{
}

inline Image::Image(std::map<int, std::vector<cv::Mat>> obj_superpixels)
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

inline cv::Mat Dataset::readImage(QString path)
{
    lastLoadedImagePath = path;
    lastLoadedImage = cv::imread(path.toStdString(), cv::IMREAD_ANYCOLOR);
    return lastLoadedImage;
}

inline cv::Mat Dataset::image()
{
    return lastLoadedImage;
}

inline cv::Mat Dataset::segments()
{
    return lastLoadedSegment;
}

inline cv::Mat &Dataset::mask()
{
    return lastLabelsMask;
}

inline std::map<QString, ImageData> MatchImage2Segmentation(std::vector<QString>& images, std::vector<QString> segmentations, ImageData::Type type)
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


inline void Dataset::writeObjectFile(Image image, QString fileName)
{
    QFileInfo fi(fileName);
    auto dir = fi.dir();
    if (!dir.exists())
    {
        dir.cdUp();
        dir.mkdir("train/");
        dir.mkdir("test");
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
#endif // DATASET_H
