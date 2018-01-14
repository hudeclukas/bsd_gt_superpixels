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

    public slots:
    void loadTrainData();
    void loadTestData();

    void changeSavePattern() override;
    void setSaveCounter(int value) override;

    QMenu* getDatasetMenu() override;
    std::map<QString, ImageData> getLoadedData() override;
    void drawSegments(const cv::Mat& input, cv::Mat& output, QString& ground_truth_path);
    cv::Mat getSegmentedImage(QString image, int segmentation) override;
    void resetData() override;
    void saveSegment2SuperpixelLabels(cv::Mat image) override;
    void saveSegment2SuperpixelLabels(std::map<int, std::vector<cv::Mat>> obj_patches) override;

private:
    QMenu *menu = nullptr;
    QString trainDataDirPath = "";
    std::vector<QString> trainFiles;
    QString testDataDirPath = "";
    std::vector<QString> testFiles;
};
#endif // PEXEL_H
