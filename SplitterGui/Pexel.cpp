#include "Pexel.h"

#include <QMenu>
#include <QFileDialog>
#include <opencv2/imgcodecs.hpp>


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

cv::Mat Pexel::readImage(QString path)
{
    Dataset::readImage(path);
    lastLoadedImage.copyTo(lastLoadedSegment);
    lastLabelsMask = cv::Mat::zeros(lastLoadedImage.rows, lastLoadedImage.cols, CV_32SC1);
    QFileInfo name(path);
    saveOptions.Image = name.baseName();
    return lastLoadedImage;
}


std::map<QString, ImageData> Pexel::getLoadedData()
{
    if (matchedData.empty())
    {
        matchedData = loadImageMetaData(trainFiles, ImageData::TRAIN);
        auto testData = loadImageMetaData(testFiles, ImageData::TEST);

        matchedData.insert(testData.begin(), testData.end());
    }
    return matchedData;
}

cv::Mat Pexel::getSegmentedImage(QString image, int)
{
    if (matchedData.empty())
    {
        return cv::Mat();
    }
    auto imageData = matchedData[image];
    if (imageData.groundTruths.empty()) return cv::Mat();

    if (imageData.absPath.compare(lastLoadedImagePath) == 0)
    {
        lastLoadedImage.copyTo(lastLoadedSegment);
    }
    saveOptions.Image = image;
    return lastLoadedSegment;
}

void Pexel::saveSegment2SuperpixelLabels(cv::Mat superpixelsLabels)
{
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
    double maxSups;
    cv::minMaxIdx(superpixelsLabels, nullptr, &maxSups);
    maxSups++;

    Image image;
    image.name = saveOptions.Image;
    image.objects.push_back(Image::ImageObject(maxSups));
    
    for (auto row = 0; row < superpixelsLabels.rows; ++row)
    {
        auto sPtr = superpixelsLabels.ptr<int>(row);
        auto iPtr = lastLoadedImage.ptr<cv::Vec3b>(row);
        for (auto col = 0; col < superpixelsLabels.cols; ++col)
        {
            image.objects[0][sPtr[col]].pixels.push_back({ row, col, iPtr[col] });
            image.objects[0][sPtr[col]].label = sPtr[col];
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

void Pexel::saveSegment2SuperpixelLabels(std::map<int, std::vector<cv::Mat>> obj_patches)
{
    Image image(obj_patches);
    image.name = saveOptions.Image;

    if (saveOptions.Path.isEmpty())
    {
        changeSavePattern();
    }

    QString fileName;
    buildObjectFileName(fileName);
    writeObjectFile(image, fileName);

    emit saved(QString("<font color='#045ae5'>" + fileName + "</font>"));
}

std::map<QString, ImageData> Pexel::loadImageMetaData(std::vector<QString>& images, ImageData::Type type)
{
    std::map<QString, ImageData> loadedImageData;
    for (auto &imPath : images)
    {
        QFileInfo imFI = QFileInfo(imPath);
        ImageData id;
        id.name = imFI.baseName();
        id.absPath = imPath;
        id.type = type;
        
        loadedImageData[id.name] = id;
    }
    return loadedImageData;
}
