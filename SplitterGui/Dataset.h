#ifndef DATASET_H
#define DATASET_H

#include <QObject>
#include <QMenu>
#include <QFileInfo>

#include <map>

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

    public slots:
    virtual void changeSavePattern() = 0;

};

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
