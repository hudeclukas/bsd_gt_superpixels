#include "Berkeley.h"

#include <QMenu>
#include <QFileDialog>

#include <iostream>

#include "DirIO.h"

#include <opencv2/core/core.hpp>
//#include <fstream>

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
    saveOptions.Prefix = spFI.baseName();
    saveOptions.Path = spFI.absolutePath();
    saveOptions.Extension = spFI.suffix();
}

void Berkeley::setSaveCounter(int value)
{
    saveOptions.Counter = value;
}

void Berkeley::loadTrainData()
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

void Berkeley::loadTestData()
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

void Berkeley::loadGroundTruth()
{
    auto files = GetAllFiles("Select Ground Truth Data root folder", std::move(groundTruthDataDirPath), std::move(QStringList() << "*.seg"), std::move(groundTruthDataDirPath));
    if (!files.empty())
    {
        groundTruthFiles = files;
    }
    std::cout << groundTruthDataDirPath.toStdString() << std::endl;
    std::cout << groundTruthFiles.size() << " ground truth files loaded" << std::endl;
    resetData();
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

void Berkeley::drawSegments(const cv::Mat& input, cv::Mat& output, QString& ground_truth_path)
{
    std::ifstream segFile(ground_truth_path.toStdString());
    std::string line = "";
    while (line.compare("data") != 0)
    {
        segFile >> line;
    }
    
    input.copyTo(output);
    lastLabelsMask = cv::Mat::zeros(input.rows, input.cols, CV_32SC1);
    lastEdgesMask = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
    for (int label, row, col_s, col_e; segFile >> label >> row >> col_s >> col_e; )
    {
        auto inPtr = input.ptr<cv::Vec3b>(row);
        auto oPtr = output.ptr<cv::Vec3b>(row);
        auto mPtr = lastLabelsMask.ptr<int>(row);
        auto ePtr = lastEdgesMask.ptr<uchar>(row);
        if (row > 0)
        {
            auto upMRow = lastLabelsMask.ptr<int>(row-1);
            for (int col = col_s; col <= col_e; ++col)
            {
                auto upVal = upMRow[col];
                if (upVal == label || upVal == -1)
                {
                    mPtr[col] = label;
                    oPtr[col] = inPtr[col];
                } 
                else
                {
                    mPtr[col] = -1;
                    ePtr[col] = 1;
                    auto color = inPtr[col];
                    color[0] = color[0] < 127 ? 255 : 0;
                    color[1] = color[1] < 127 ? 255 : 0;
                    color[2] = color[2] < 127 ? 255 : 0;
                    oPtr[col] = color;
                }
            }
        }
        else
        {
            for (int col = col_s; col < col_e; ++col)
            {
                mPtr[col] = label;
                oPtr[col] = inPtr[col];
            }
        }
        if (col_e + 1 < input.cols)
        {
            mPtr[col_e] = -1;
            ePtr[col_e] = 1;
            auto color = inPtr[col_e];
            color[0] = color[0] < 127 ? 255 : 0;
            color[1] = color[1] < 127 ? 255 : 0;
            color[2] = color[2] < 127 ? 255 : 0;
            oPtr[col_e] = color;
        }
    }
    segFile.close();
}

cv::Mat Berkeley::getSegmentedImage(QString image, int segmentation)
{
    if (matchedData.empty())
    {
        return cv::Mat();
    }
    auto imageData = matchedData[image];
    if (imageData.groundTruths.empty()) return cv::Mat();

    if (imageData.absPath.compare(lastLoadedImagePath) == 0)
    {
        drawSegments(lastLoadedImage, lastLoadedSegment, matchedData[image].groundTruths[segmentation]);
    }
    saveOptions.Image = image;
    return lastLoadedSegment;
}

void Berkeley::resetData()
{
    matchedData.clear();
}

void Berkeley::buildObjectFileName(QString& fileName)
{
    auto imgData = matchedData[saveOptions.Image];
    fileName = saveOptions.Path + "/";
    if (imgData.type == ImageData::TRAIN)
    {
        fileName += "train/";
    }
    else if (imgData.type == ImageData::TEST)
    {
        fileName += "test/";
    }
    fileName += saveOptions.Prefix + "_" + saveOptions.Image + "_" + QString::number(saveOptions.Counter) + "." + saveOptions.Extension;
}

void Berkeley::saveSegment2SuperpixelLabels(cv::Mat superpixelsLabels)
{
    assert(superpixelsLabels.size == lastLabelsMask.size);
    if (saveSuperpixelsMask)
    {
        cv::Mat tmp;
        superpixelsLabels.convertTo(tmp, CV_16UC1);
        tmp += 1;
        auto path = saveOptions.Path + "/";
        QDir dir(path);
        if (!dir.exists("segments/"))
        {
            dir.mkdir("segments/");
        }
        cv::imwrite(path.toStdString() + "segments/" + saveOptions.Prefix.toStdString() + "_" + saveOptions.Image.toStdString() + "_" + std::to_string(saveOptions.Counter) + ".png", tmp);
    }
    double maxSups, maxBerks;
    cv::minMaxIdx(superpixelsLabels, nullptr, &maxSups);
    cv::minMaxIdx(lastLabelsMask, nullptr, &maxBerks);
    maxSups++;
    maxBerks++;

    Image image;
    image.name = saveOptions.Image;
    for (int i = 0; i < maxBerks; ++i)
    {
        image.objects.push_back(Image::ImageObject(maxSups));
    }

    for (auto row = 0; row < superpixelsLabels.rows; ++row)
    {
        auto bPtr = lastLabelsMask.ptr<int>(row);
        auto sPtr = superpixelsLabels.ptr<int>(row);
        auto iPtr = lastLoadedImage.ptr<cv::Vec3b>(row);
        for (auto col = 0; col < superpixelsLabels.cols; ++col)
        {
            if (bPtr[col] != -1)
            {
                image.objects[bPtr[col]][sPtr[col]].pixels.push_back({ row, col, iPtr[col] });
                image.objects[bPtr[col]][sPtr[col]].label = sPtr[col];
            } else
            {
                bool done = false;
                for (int i = 0; i < image.objects.size(); ++i)
                {
                    for (auto && superpixel : image.objects[i])
                    {
                        if (superpixel.label == sPtr[col])
                        {
                            superpixel.invalidCount++;
                            if (superpixel.invalidCount > 10)
                            {
                                superpixel.isValid = false;
                            }
                            done = true;
                            break;
                        }
                    }
                    if (done) break;
                }
            }
        }
    }

    for (auto objIt = image.objects.begin(); objIt < image.objects.end(); )
    {
        for (auto supIt = objIt->begin(); supIt != objIt->end(); )
        {
            if (supIt->pixels.size() > 20 && supIt->isValid)
            {
                supIt->createSuperpixelMat();
                ++supIt;
            }
            else
            {
                objIt->erase(supIt);
            }
        }

        if (objIt->size() < 2)
        {
            image.objects.erase(objIt);
        }
        else
        {
            ++objIt;
        }
    }

    if (saveOptions.Path.isEmpty())
    {
        changeSavePattern();
    }

    QString fileName;
    buildObjectFileName(fileName);
    writeObjectFile(image, fileName);
    
    emit saved(QString("<font color='#045ae5'>" + fileName + "</font>"));
}

void Berkeley::saveSegment2SuperpixelLabels(std::map<int, std::vector<cv::Mat>> obj_patches)
{
    Image image(obj_patches);
    image.name = saveOptions.Image;
    
    for (auto objIt = image.objects.begin(); objIt < image.objects.end(); )
    {
        if (objIt->size() < 2)
        {
            image.objects.erase(objIt);
        }
        else
        {
            ++objIt;
        }
    }

    if (saveOptions.Path.isEmpty())
    {
        changeSavePattern();
    }

    QString fileName;
    buildObjectFileName(fileName);
    writeObjectFile(image, fileName);

    emit saved(QString("<font color='#045ae5'>" + fileName + "</font>"));
}
