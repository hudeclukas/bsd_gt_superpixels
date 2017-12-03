#ifndef DIRIO_H
#define DIRIO_H

#include <vector>
#include <QStringList>

/**
 * Searches and finds all files recursively from User selected directory
 */
std::vector<QString> GetAllFiles(QString title, QString root = QString(), QStringList &&filters = QStringList(), QString &&selectedDirectory = QString());

QString SaveFileTo(QString title, QString dir = QString());

#endif // DIRIO_H
