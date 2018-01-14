#ifndef DATASET_H
#define DATASET_H

#include <QObject>

#include <opencv2/core/mat.hpp>

struct SaveOptions
{
    QString Path = "";
    QString Prefix = "";
    QString Image = "";
    QString Extension = "";
    int Counter = 0;
};

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
        Pixel(int row_, int col_, uchar r_, uchar g_, uchar b_);
        Pixel(int row_, int col_, cv::Vec3b color);
        int row, col;
        uchar r, g, b;
    };

    void createSuperpixelMat();

    std::vector<Pixel> pixels;
    uint label = 0;
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

class QMenu;

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

    void setSaveSuperpixelsMask(bool save);

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

    std::map<QString, ImageData> matchedData;

    SaveOptions saveOptions;
    bool saveSuperpixelsMask = false;

    void buildObjectFileName(QString& fileName);
    static void writeObjectFile(Image image, QString fileName);
};

std::map<QString, ImageData> MatchImage2Segmentation(std::vector<QString>& images, std::vector<QString> segmentations, ImageData::Type type);

#endif // DATASET_H
