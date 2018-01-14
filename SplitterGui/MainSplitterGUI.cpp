#include "MainSplitterGUI.h"

#include "ui_SplitterGUI.h"

#include "Berkeley.h"
#include "Pexel.h"
#include "Converter.h"

#include <iostream>
#include <opencv2/ximgproc/seeds.hpp>
#include <fstream>

MainSplitterGUI::MainSplitterGUI() : dataset_(nullptr)
{
    this->ui = new Ui_SplitterGUI;
    this->ui->setupUi(this);

    initCentralSplitter();
    setRunEnabled(false);
    initSuperpixelOptions();

    setLSCvisible(true);
    setSEEDSvisible(false);
    setSQUARESvisible(false);

    connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(on_actionExit()));
    connect(ui->actionNext, SIGNAL(triggered()), this, SLOT(on_next()));
    connect(ui->actionBack, SIGNAL(triggered()), this, SLOT(on_previous()));
    connect(ui->actionBerkeley_dataset, SIGNAL(triggered()), this, SLOT(on_actionBerkeley()));
    connect(ui->actionPexel_texture, SIGNAL(triggered()), this, SLOT(on_actionPexel()));
    connect(ui->loadDataButton, SIGNAL(clicked()), this, SLOT(on_LoadAndViewData()));
    connect(ui->imgList, SIGNAL(itemSelectionChanged()), this, SLOT(change_ImageListSelected()));
    connect(ui->segList, SIGNAL(itemSelectionChanged()), this, SLOT(change_SegListSelected()));
    connect(ui->algorithmBox, SIGNAL(currentIndexChanged(int)), this, SLOT(change_AlgorithmSelection(int)));
    connect(ui->runButton, SIGNAL(clicked()), this, SLOT(runSuperpixel()));
    connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(saveSuperpixels()));
    connect(ui->saveSuperpixelMask, SIGNAL(toggled(bool)), this, SLOT(saveSuperpixelMask(bool)));

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
        if (dynamic_cast<Berkeley *>(dataset_) != nullptr)
        {
            return;
        }
        ui->menuBar->removeAction(dataset_->getDatasetMenu()->menuAction());
        delete dataset_;
        dataset_ = new Berkeley();
    }
    ui->menuBar->addMenu(dataset_->getDatasetMenu());
    connect(ui->actionSave_Folder, SIGNAL(triggered()), dataset_, SLOT(changeSavePattern()));
    connect(dataset_, SIGNAL(saved(QString)), ui->saveLbl, SLOT(setText(QString)));
    connect(ui->saveCount, SIGNAL(valueChanged(int)), dataset_, SLOT(setSaveCounter(int)));
    setRunEnabled(false);
}

void MainSplitterGUI::on_actionPexel()
{
    if (!dataset_)
    {
        dataset_ = new Pexel();
    }
    else
    {
        if (dynamic_cast<Pexel *>(dataset_) != nullptr)
        {
            return;
        }
        ui->menuBar->removeAction(dataset_->getDatasetMenu()->menuAction());
        delete dataset_;
        dataset_ = new Pexel();
        
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
    if (!mimage.empty())
    {
        setImageTo(mimage, ui->objImage);
    }
}

void MainSplitterGUI::change_AlgorithmSelection(int algo)
{
    splitter.superpixelAlgorithm(static_cast<Splitter::Algorithm>(algo));
    if (algo == Splitter::Algorithm::LSC)
    {
        setLSCvisible(true);
        setSEEDSvisible(false);
        setSQUARESvisible(false);
    }
    if (algo == Splitter::Algorithm::SEEDS)
    {
        setLSCvisible(false);
        setSEEDSvisible(true);
        setSQUARESvisible(false);
    }
    if (algo == Splitter::Algorithm::SQUARES)
    {
        setLSCvisible(false);
        setSEEDSvisible(false);
        setSQUARESvisible(true);
    }
}

void MainSplitterGUI::runSuperpixel()
{
    if (ui->algorithmBox->currentIndex() == Splitter::Algorithm::LSC)
    {
        int region_size = ui->regionSizeVal->value();
        float ratio = ui->ratioVal->value();
        int connectivity = ui->connectivityVal->value();
        splitter.setRatio(ratio);
        splitter.setRegionSize(region_size);
        splitter.setConnectivityMinElement(connectivity);
    }
    if (ui->algorithmBox->currentIndex() == Splitter::Algorithm::SEEDS)
    {
        int number = ui->seedsNumberVal->value();
        int levels = ui->seedsLevelVal->value();
        int prior = ui->seedsPriorVal->value();

        splitter.setNumberOfSuperpixels(number);
        splitter.setLevels(levels);
        splitter.setPrior(prior);
    }
    if (ui->algorithmBox->currentIndex() == Splitter::Algorithm::SQUARES)
    {
        int size = ui->squaresSizeVal->value();
        double shift = ui->squaresShiftVal->value();
        splitter.setSquareSize(size);
        splitter.setSquareShift(shift);
    }
    if (dataset_->segments().empty())
    {
        ui->segList->setCurrentRow(0);
    }

    auto input = dataset_->segments();
    cv::Mat output;
    splitter.run(input, output, dataset_->mask());
    setImageTo(output, ui->supImage);
}

void MainSplitterGUI::saveSuperpixels()
{
    if (splitter.superpixelAlgorithm() == Splitter::SQUARES)
    {
        dataset_->saveSegment2SuperpixelLabels(splitter.getSquares());
    }
    else
    {
        dataset_->saveSegment2SuperpixelLabels(splitter.getLabels());
    }
    increaseCounter();
}

void MainSplitterGUI::saveSuperpixelMask(const bool save)
{
    dataset_->setSaveSuperpixelsMask(save);
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
        if (ui->segList->count() == 0)
        {
            runSuperpixel();
            saveSuperpixels();
            counter++;
            dataset_->setSaveCounter(counter);
        }
        for (auto segIt = 0; segIt < ui->segList->count(); ++segIt)
        {
            ui->segList->setCurrentRow(segIt);
            change_SegListSelected();
            runSuperpixel();
            saveSuperpixels();
            counter++;
            dataset_->setSaveCounter(counter);
            if (!ui->useAllAnotations->isChecked())
            {
                break;
            }
        }
    }
    ui->saveCount->setValue(counter);
    connect(ui->imgList, SIGNAL(itemSelectionChanged()), this, SLOT(change_ImageListSelected()));
    connect(ui->segList, SIGNAL(itemSelectionChanged()), this, SLOT(change_SegListSelected()));
}

void MainSplitterGUI::setLSCvisible(bool visible)
{
    ui->iterationsVal->setVisible(visible);
    ui->regionSizeVal->setVisible(visible);
    ui->ratioVal->setVisible(visible);
    ui->connectivityVal->setVisible(visible);
    ui->iterationsLbl->setVisible(visible);
    ui->regionSizeLbl->setVisible(visible);
    ui->ratioLbl->setVisible(visible);
    ui->connectivityLbl->setVisible(visible);
}

void MainSplitterGUI::setSEEDSvisible(bool visible)
{
    ui->seedsLevelLbl->setVisible(visible);
    ui->seedsNumberLbl->setVisible(visible);
    ui->seedsPriorLbl->setVisible(visible);
    ui->seedsLevelVal->setVisible(visible);
    ui->seedsNumberVal->setVisible(visible);
    ui->seedsPriorVal->setVisible(visible);

}

void MainSplitterGUI::setSQUARESvisible(bool visible)
{
    ui->squaresSizeLbl->setVisible(visible);
    ui->squaresSizeVal->setVisible(visible);
    ui->squaresShiftLbl->setVisible(visible);
    ui->squaresShiftVal->setVisible(visible);
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
    ui->seedsNumberVal->setValue(splitter.getNumberOfSuperpixels());
    ui->seedsLevelVal->setValue(splitter.getLevels());
    ui->seedsPriorVal->setValue(splitter.getPrior());
    ui->squaresSizeVal->setValue(splitter.getSquareSize());
    ui->squaresShiftVal->setValue(splitter.getSquareShift());
}

void MainSplitterGUI::setRunEnabled(bool enable)
{
    ui->runButton->setEnabled(enable);
    ui->saveButton->setEnabled(enable);
    ui->autoRunSaveButton->setEnabled(enable);
    ui->saveSuperpixelMask->setEnabled(enable);
}

void MainSplitterGUI::increaseCounter()
{
    ui->saveCount->setValue(ui->saveCount->value() + 1);
}

void MainSplitterGUI::resetCounter()
{
    ui->saveCount->setValue(0);
}

