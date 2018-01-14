#include "Pexel.h"

#include <iostream>

#include <QMenu>
#include <QFileDialog>

#include "DirIO.h"

Pexel::Pexel()
{
    menu = new QMenu("Pexel photos");
    auto trainData = menu->addAction("Train images");
    trainData->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_R));
    auto testData = menu->addAction("Test images");
    testData->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_T));
    
    connect(trainData, SIGNAL(triggered()), this, SLOT(loadTrainData()));
    connect(testData, SIGNAL(triggered()), this, SLOT(loadTestData()));
}

Pexel::~Pexel()
{
}

void Pexel::loadTrainData()
{
    auto files = GetAllFiles("Select Train Data root folder", std::move(trainDataDirPath), std::move(QStringList() << "*.jpg"), std::move(trainDataDirPath));
    if (!files.empty())
    {
        trainFiles = files;
    }

    std::cout << trainDataDirPath.toStdString() << std::endl;
    std::cout << trainFiles.size() << " train files loaded" << std::endl;
    resetData();
}

void Pexel::loadTestData()
{
    auto files = GetAllFiles("Select Test Data root folder", std::move(testDataDirPath), std::move(QStringList() << "*.jpg"), std::move(testDataDirPath));
    if (!files.empty())
    {
        testFiles = files;
    }

    std::cout << testDataDirPath.toStdString() << std::endl;
    std::cout << testFiles.size() << " test files loaded" << std::endl;
    resetData();
}

void Pexel::changeSavePattern()
{
    auto savePattern = SaveFileTo("Select where to save", trainDataDirPath);
    QFileInfo spFI(savePattern);
    saveOptions.Prefix = spFI.baseName();
    saveOptions.Path = spFI.absolutePath();
    saveOptions.Extension = spFI.suffix();
}

void Pexel::setSaveCounter(int value)
{
    saveOptions.Counter = value;
}

QMenu* Pexel::getDatasetMenu()
{
    return menu;
}

std::map<QString, ImageData> Pexel::getLoadedData()
{
    return std::map<QString, ImageData>();
}

void Pexel::drawSegments(const cv::Mat& input, cv::Mat& output, QString& ground_truth_path)
{
}

cv::Mat Pexel::getSegmentedImage(QString image, int segmentation)
{
    return cv::Mat();
}

void Pexel::resetData()
{
    matchedData.clear();
}

void Pexel::saveSegment2SuperpixelLabels(cv::Mat image)
{
}

void Pexel::saveSegment2SuperpixelLabels(std::map<int, std::vector<cv::Mat>> obj_patches)
{
}
