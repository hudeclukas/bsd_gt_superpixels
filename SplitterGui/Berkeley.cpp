#include "Berkeley.h"

#include <QMenu>
#include <QFileDialog>

#include <iostream>

#include "DirIO.h"

Berkeley::Berkeley()
{
    menu = new QMenu("Berkeley data");
    auto trainData = menu->addAction("Train images");
    trainData->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_R));
    auto testData = menu->addAction("Test images");
    testData->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_T));
    auto groundTruth = menu->addAction("Ground truth");
    groundTruth->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_G));

    connect(trainData, SIGNAL(triggered()), this, SLOT(loadTrainData()));
    connect(testData, SIGNAL(triggered()), this, SLOT(loadTestData()));
    connect(groundTruth, SIGNAL(triggered()), this, SLOT(loadGroundTruth()));
}

Berkeley::~Berkeley()
{
}

void Berkeley::changeSavePattern()
{
    auto savePattern = SaveFileTo("Select where to save", trainDataDirPath);
    QFileInfo spFI(savePattern);
    saveOptions.Name = spFI.baseName();
    saveOptions.Path = spFI.absolutePath();
    saveOptions.Extension = spFI.suffix();

}

void Berkeley::loadTrainData()
{
    trainFiles = GetAllFiles("Select Train Data root folder", std::move(trainDataDirPath), std::move(QStringList() << "*.jpg"), std::move(trainDataDirPath));
    {
        std::cout << trainDataDirPath.toStdString() << std::endl;
        std::cout << trainFiles.size() << " train files loaded" << std::endl;
    }
}

void Berkeley::loadTestData()
{
    testFiles = GetAllFiles("Select Test Data root folder", std::move(testDataDirPath), std::move(QStringList() << "*.jpg"), std::move(testDataDirPath));
    {
        std::cout << testDataDirPath.toStdString() << std::endl;
        std::cout << testFiles.size() << " test files loaded" << std::endl;
    }
}

void Berkeley::loadGroundTruth()
{
    groundTruthFiles = GetAllFiles("Select Ground Truth Data root folder", std::move(groundTruthDataDirPath), std::move(QStringList() << "*.seg"), std::move(groundTruthDataDirPath));
    {
        std::cout << groundTruthDataDirPath.toStdString() << std::endl;
        std::cout << groundTruthFiles.size() << " ground truth files loaded" << std::endl;
    }
}

QMenu* Berkeley::getDatasetMenu()
{
    return menu;
}

std::map<QString, ImageData> Berkeley::getLoadedData()
{
    if (matchedData.empty())
    {
        matchedData = MatchImage2Segmentation(trainFiles, groundTruthFiles, ImageData::TRAIN);
        auto testData = MatchImage2Segmentation(testFiles, groundTruthFiles, ImageData::TEST);

        matchedData.insert(testData.begin(), testData.end());
    }
    return matchedData;
}
