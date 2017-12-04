#ifndef DATASET_H
#define DATASET_H

#include <QObject>
#include <QMenu>
#include <QFileInfo>

#include <map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

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
    cv::Mat lastBerkeleyLabelsMask;
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
    return lastBerkeleyLabelsMask;
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
