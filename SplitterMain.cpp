#include <QApplication>

#include "SplitterGui/MainSplitterGUI.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    MainSplitterGUI guiMain;
    guiMain.show();

    return app.exec();
}
