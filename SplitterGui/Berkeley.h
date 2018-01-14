#ifndef BERKELEY_H
#define BERKELEY_H

#include "Dataset.h"
#include <map>


class QMenu;

class Berkeley : public Dataset
{
    Q_OBJECT
public:
    Berkeley();
    ~Berkeley();

    public slots:
    void loadGroundTruth();

    std::map<QString, ImageData> getLoadedData() override;
    void drawSegments(const cv::Mat& input, cv::Mat& output, QString& ground_truth_path);
    cv::Mat getSegmentedImage(QString image, int segmentation) override;
    void saveSegment2SuperpixelLabels(cv::Mat image) override;
    void saveSegment2SuperpixelLabels(std::map<int, std::vector<cv::Mat>> obj_patches) override;

private:
    QString groundTruthDataDirPath = "";
    std::vector<QString> groundTruthFiles;

};
#endif // BERKELEY_H
