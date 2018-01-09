#ifndef MAINSPLITTERGUI_H
#define MAINSPLITTERGUI_H

#include <QMainWindow>
#include <opencv2/core/base.hpp>
#include "SplitterAlgo/Splitter.h"
#include <qlabel.h>

class Ui_SplitterGUI;
class Dataset;

class MainSplitterGUI : public QMainWindow
{
    Q_OBJECT
public:
    MainSplitterGUI();
    ~MainSplitterGUI();

    public slots:
    void on_next();
    void on_previous();
    void on_actionExit();
    void on_actionBerkeley();
    void on_LoadAndViewData();

    void change_ImageListSelected();
    void setImageTo(cv::Mat mimage, QLabel* object);
    void change_SegListSelected();
    void change_AlgorithmSelection(int algo);

    void runSuperpixel();
    void saveSuperpixels();

    void autoRun();

private:
    Ui_SplitterGUI *ui;

    Dataset *dataset_;

    Splitter splitter;

    void setLSCvisible(bool visible);
    void setSEEDSvisible(bool visible);
    void setSQUARESvisible(bool visible);

    void unloadDatasetActions();
    void initCentralSplitter();
    void initSuperpixelOptions();
    void setRunEnabled(bool enable);
    void increaseCounter();
    void resetCounter();

};
#endif // MAINSPLITTERGUI_H
