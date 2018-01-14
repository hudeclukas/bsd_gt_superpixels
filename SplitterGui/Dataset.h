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

    virtual cv::Mat readImage(QString path);
    cv::Mat image();
    cv::Mat segments();
    cv::Mat &mask();

    void setSaveSuperpixelsMask(bool save);

    public slots:
    virtual QMenu * getDatasetMenu();

    virtual std::map<QString, ImageData> getLoadedData() = 0;
    virtual cv::Mat getSegmentedImage(QString image, int segmentation = 0) = 0;
    virtual void resetData();
    virtual void saveSegment2SuperpixelLabels(cv::Mat image) = 0;
    virtual void saveSegment2SuperpixelLabels(std::map<int, std::vector<cv::Mat>> patches) = 0;

    virtual void changeSavePattern();
    virtual void setSaveCounter(int value);
    void loadTrainData();
    void loadTestData();

    signals:
    void saved(QString path);

protected:
    QMenu *menu = nullptr;

    QString trainDataDirPath = "";
    std::vector<QString> trainFiles;
    QString testDataDirPath = "";
    std::vector<QString> testFiles;

    SaveOptions saveOptions;
    bool saveSuperpixelsMask = false;

    QString lastLoadedImagePath;
    // Last loaded and used RGB image from dataset
    cv::Mat lastLoadedImage;
    // Last loaded and used Ground Truth segmentation
    cv::Mat lastLoadedSegment;
    // Last used Ground Truth segment label-indexes mask
    cv::Mat lastLabelsMask;
    // Last used Ground Truth segment edges mask
    cv::Mat lastEdgesMask;

    std::map<QString, ImageData> matchedData;

    void buildObjectFileName(QString& fileName);
    static void writeObjectFile(Image image, QString fileName);
};

std::map<QString, ImageData> MatchImage2GroundTruths(std::vector<QString>& images, std::vector<QString> segmentations, ImageData::Type type);

#endif // DATASET_H
