#ifndef DATASET_H
#define DATASET_H

#include <QObject>
#include <QMenu>
#include <QFileInfo>

#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <set>

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
            int left = 0, up = 0, right = 0, down = 0;
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
            superpixelMat = cv::Mat::zeros(down - up + 1, right - left + 1, CV_8UC3) + color;
            for (auto && p : pixels)
            {
                superpixelMat.at<cv::Vec3b>(p.row, p.col) = { p.b,p.g,p.r };
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

    cv::Mat readImage(QString path);
    cv::Mat image();
    cv::Mat segments();
    cv::Mat mask();

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
};

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

inline cv::Mat Dataset::mask()
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

            if (seName.contains(id.name))
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

#endif // DATASET_H
