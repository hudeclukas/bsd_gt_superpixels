#ifndef MAINSPLITTERGUI_H
#define MAINSPLITTERGUI_H
#include <QMainWindow>

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

private:
    Ui_SplitterGUI *ui;

    Dataset *dataset_;



    void unloadDatasetActions();
    void initCentralSplitter();

};
#endif // MAINSPLITTERGUI_H
