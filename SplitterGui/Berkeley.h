#ifndef BERKELEY_H
#define BERKELEY_H

#include "Dataset.h"
#include <map>


class QMenu;

struct SaveOptions
{
    QString Path = "";
    QString Prefix = "";
    QString Image = "";
    QString Extension = "";
    int Counter = 0;
};

class Berkeley : public Dataset
{
    Q_OBJECT
public:
    Berkeley();
    ~Berkeley();


    public slots:
    void loadTrainData();
    void loadTestData();
    void loadGroundTruth();
    
    void changeSavePattern() override;
    void setSaveCounter(int value) override;

    QMenu* getDatasetMenu() override;
    std::map<QString, ImageData> getLoadedData() override;
    void drawSegments(const cv::Mat& input, cv::Mat& output, QString& ground_truth_path);
    cv::Mat getSegmentedImage(QString image, int segmentation) override;
    void resetData() override;
    void saveSegment2SuperpixelLabels(cv::Mat image) override;


private:
    QMenu *menu;
    QString trainDataDirPath = "";
    std::vector<QString> trainFiles;
    QString testDataDirPath = "";
    std::vector<QString> testFiles;
    QString groundTruthDataDirPath = "";
    std::vector<QString> groundTruthFiles;

    std::map<QString, ImageData> matchedData;

    SaveOptions saveOptions;
};
#endif // BERKELEY_H
