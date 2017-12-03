#ifndef BERKELEY_H
#define BERKELEY_H

#include "Dataset.h"
#include <map>

class QMenu;

struct SaveOptions
{
    QString Path = "";
    QString Name = "";
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

    QMenu* getDatasetMenu() override;
    std::map<QString, ImageData> getLoadedData() override;


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
