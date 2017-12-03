#include "MainSplitterGUI.h"

#include "ui_SplitterGUI.h"

#include "Berkeley.h"
#include <iostream>

MainSplitterGUI::MainSplitterGUI() : dataset_(nullptr)
{
    this->ui = new Ui_SplitterGUI;
    this->ui->setupUi(this);

    initCentralSplitter();

    connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(on_actionExit()));
    connect(ui->actionNext, SIGNAL(triggered()), this, SLOT(on_next()));
    connect(ui->actionBack, SIGNAL(triggered()), this, SLOT(on_previous()));
    connect(ui->actionBerkeley_dataset, SIGNAL(triggered()), this, SLOT(on_actionBerkeley()));
    connect(ui->loadDataButton, SIGNAL(clicked()), this, SLOT(on_LoadAndViewData()));
    connect(ui->imgList, SIGNAL(itemSelectionChanged()), this, SLOT(change_ImageListSelected()));
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
}

void MainSplitterGUI::change_ImageListSelected()
{
    auto loadedData = dataset_->getLoadedData();
    auto item = ui->imgList->currentItem();
    auto data = loadedData.find(item->text());
    
    if (data->second.groundTruths.empty()) { return; }
    auto segList = ui->segList;
    segList->clear();
    for (QString ground_truth : data->second.groundTruths)
    {
        segList->addItem(ground_truth);
    }
}

void MainSplitterGUI::unloadDatasetActions()
{
}

void MainSplitterGUI::initCentralSplitter()
{
    ui->centralSplitter->setSizes(QList<int>({ 1000 - ui->optionsLabel->width(), ui->optionsLabel->width() }));
}

