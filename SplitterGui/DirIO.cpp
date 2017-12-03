#include "DirIO.h"

#include <QFileDialog>
#include <QDirIterator>

std::vector<QString> GetAllFiles(QString title, QString root, QStringList &&filters, QString &&selectedDirectory)
{
    selectedDirectory = QFileDialog::getExistingDirectory(nullptr, title, root, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (selectedDirectory.isEmpty())
    {
        return std::vector<QString>();
    }

    QDirIterator it(selectedDirectory, filters, QDir::Files, QDirIterator::Subdirectories);
    std::vector<QString> files;
    while (it.hasNext())
    {
        files.push_back(it.next());
    }
    return files;
}

QString SaveFileTo(QString title, QString dir)
{
    return QFileDialog::getSaveFileName(nullptr, title, dir);
}
