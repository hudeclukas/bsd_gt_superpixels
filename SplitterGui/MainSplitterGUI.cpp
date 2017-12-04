#include "MainSplitterGUI.h"

#include "ui_SplitterGUI.h"

#include "Berkeley.h"
#include "Converter.h"

#include <iostream>

MainSplitterGUI::MainSplitterGUI() : dataset_(nullptr)
{
    this->ui = new Ui_SplitterGUI;
    this->ui->setupUi(this);

    initCentralSplitter();
    setRunEnabled(false);
    initSuperpixelOptions();

    connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(on_actionExit()));
    connect(ui->actionNext, SIGNAL(triggered()), this, SLOT(on_next()));
    connect(ui->actionBack, SIGNAL(triggered()), this, SLOT(on_previous()));
    connect(ui->actionBerkeley_dataset, SIGNAL(triggered()), this, SLOT(on_actionBerkeley()));
    connect(ui->loadDataButton, SIGNAL(clicked()), this, SLOT(on_LoadAndViewData()));
    connect(ui->imgList, SIGNAL(itemSelectionChanged()), this, SLOT(change_ImageListSelected()));
    connect(ui->segList, SIGNAL(itemSelectionChanged()), this, SLOT(change_SegListSelected()));
    connect(ui->runButton, SIGNAL(clicked()), this, SLOT(runSuperpixel()));
    connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(saveSuperpixels()));

    connect(ui->autoRunSaveButton, SIGNAL(clicked()), this, SLOT(autoRun()));
}

MainSplitterGUI::~MainSplitterGUI()
{
    qApp->exit();
}

void MainSplitterGUI::on_next()
{
    auto list = ui->imgList;
    list->setCurrentRow(list->currentRow() == list->count() - 1 ? list->currentRow() : list->currentRow() + 1);
}

void MainSplitterGUI::on_previous()
{
    auto list = ui->imgList;
    list->setCurrentRow(list->currentRow() == 0 ? 0 : list->currentRow() - 1);
}

void MainSplitterGUI::on_actionExit()
{
    qApp->exit();
}

void MainSplitterGUI::on_actionBerkeley()
{
    if (!dataset_)
    {
        dataset_ = new Berkeley();
    }
    else
    {
        try
        {
            dynamic_cast<Berkeley *>(dataset_);
            return;
        }
        catch (std::bad_cast)
        {
            delete dataset_;
            dataset_ = new Berkeley();
        }
    }
    ui->menuBar->addMenu(dataset_->getDatasetMenu());
    connect(ui->actionSave_Folder, SIGNAL(triggered()), dataset_, SLOT(changeSavePattern()));
    connect(dataset_, SIGNAL(saved(QString)), ui->saveLbl, SLOT(setText(QString)));
    connect(ui->saveCount, SIGNAL(valueChanged(int)), dataset_, SLOT(setSaveCounter(int)));
    setRunEnabled(false);
}

void MainSplitterGUI::on_LoadAndViewData()
{
    auto loadedData = dataset_->getLoadedData();
    auto imageList = ui->imgList;
    for (auto it = loadedData.begin(); it != loadedData.end(); ++it)
    {
        imageList->addItem(it->first);
        auto lastItem = imageList->count() - 1;
        auto item = imageList->item(lastItem);
        if (it->second.type==ImageData::TRAIN)
        {
            item->setBackgroundColor(QColor(164, 252, 186));
        }
        else if (it->second.type == ImageData::TEST)
        {
            item->setBackgroundColor(QColor(250, 252, 164));
        }
    }
    imageList->setCurrentRow(0);
    setRunEnabled(true);
}

void MainSplitterGUI::change_ImageListSelected()
{
    auto loadedData = dataset_->getLoadedData();
    auto item = ui->imgList->currentItem();
    auto data = loadedData[item->text()];

    auto segList = ui->segList;
    segList->clear();
    segList->setCurrentItem(nullptr);
    for (QString ground_truth : data.groundTruths)
    {
        segList->addItem(ground_truth);
    }

    auto mimage = dataset_->readImage(data.absPath);
    setImageTo(mimage, ui->objImage);
    ui->saveCount->setValue(0);
}

void MainSplitterGUI::setImageTo(cv::Mat mimage, QLabel* object)
{
    if (mimage.rows > mimage.cols)
    {
        mimage = mimage.t();
    }
    auto tmp = cv::Mat();
    cv::cvtColor(mimage, tmp, CV_BGR2RGB);
    auto qimage = Mat2QImage(tmp);

    object->setPixmap(QPixmap::fromImage(qimage));
}

void MainSplitterGUI::change_SegListSelected()
{
    auto segm = ui->segList->currentRow();
    auto image = ui->imgList->currentItem()->text();
    auto mimage = dataset_->getSegmentedImage(image, segm);
    setImageTo(mimage, ui->objImage);
}

void MainSplitterGUI::runSuperpixel()
{
    int region_size = ui->regionSizeVal->value();
    float ratio = ui->ratioVal->value();
    int connectivity = ui->connectivityVal->value();
    splitter.setRatio(ratio);
    splitter.setRegionSize(region_size);
    splitter.setConnectivityMinElement(connectivity);

    if (dataset_->segments().empty())
    {
        ui->segList->setCurrentRow(0);
    }

    auto input = dataset_->segments();
    
    cv::Mat output;
    splitter.run(input, output);
    setImageTo(output, ui->supImage);
}

void MainSplitterGUI::saveSuperpixels()
{
    dataset_->saveSegment2SuperpixelLabels(splitter.getLabels());
    increaseCounter();
}

void MainSplitterGUI::autoRun()
{
    ui->imgList->disconnect();
    ui->segList->disconnect();
    int counter = 0;
    for (auto imgIt = 0; imgIt < ui->imgList->count(); ++imgIt)
    {
        counter = 0;
        ui->imgList->setCurrentRow(imgIt);
        change_ImageListSelected();
        for (auto segIt = 0; segIt < ui->segList->count(); ++segIt)
        {
            ui->segList->setCurrentRow(segIt);
            change_SegListSelected();
            runSuperpixel();
            saveSuperpixels();
            counter++;
            dataset_->setSaveCounter(counter);
        }
    }
    ui->saveCount->setValue(counter);
    connect(ui->imgList, SIGNAL(itemSelectionChanged()), this, SLOT(change_ImageListSelected()));
    connect(ui->segList, SIGNAL(itemSelectionChanged()), this, SLOT(change_SegListSelected()));
}

void MainSplitterGUI::unloadDatasetActions()
{
}

void MainSplitterGUI::initCentralSplitter()
{
    ui->centralSplitter->setSizes(QList<int>({ 1000 - ui->optionsLabel->width(), ui->optionsLabel->width() }));
}

void MainSplitterGUI::initSuperpixelOptions()
{
    ui->iterationsVal->setValue(splitter.getNumberOfIterations());
    ui->regionSizeVal->setValue(splitter.getRegionSize());
    ui->ratioVal->setValue(splitter.getRatio());
    ui->connectivityVal->setValue(splitter.getConectivityMinElement());
}

void MainSplitterGUI::setRunEnabled(bool enable)
{
    ui->runButton->setEnabled(enable);
    ui->saveButton->setEnabled(enable);
    ui->autoRunSaveButton->setEnabled(enable);
}

void MainSplitterGUI::increaseCounter()
{
    ui->saveCount->setValue(ui->saveCount->value() + 1);
}

void MainSplitterGUI::resetCounter()
{
    ui->saveCount->setValue(0);
}

