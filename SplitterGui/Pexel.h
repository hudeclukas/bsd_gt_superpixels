#ifndef PEXEL_H
#define PEXEL_H
#include "Dataset.h"

class QMenu;

class Pexel : public Dataset
{
    Q_OBJECT
public:
    Pexel();
    ~Pexel();

    cv::Mat readImage(QString path) override;

    public slots:

    std::map<QString, ImageData> getLoadedData() override;
    cv::Mat getSegmentedImage(QString image, int segmentation) override;
    void saveSegment2SuperpixelLabels(cv::Mat image) override;
    void saveSegment2SuperpixelLabels(std::map<int, std::vector<cv::Mat>> obj_patches) override;

private:
    std::map<QString, ImageData> loadImageMetaData(std::vector<QString>& images, ImageData::Type type);

};
#endif // PEXEL_H
