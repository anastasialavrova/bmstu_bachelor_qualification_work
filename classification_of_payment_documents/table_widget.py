from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from PyQt5.uic.properties import QtGui, QtWidgets


class TableWidget(QDialog):
    def __init__(self, parent=None):
        super(TableWidget, self).__init__(parent)
        self.ui = uic.loadUi("Table.ui", self)
        self.table: QTableWidget = self.findChild(QTableWidget, 'tableWidget')


    def show(self, dictionary):
        for i in range(self.table.rowCount()):
            self.table.removeRow(i)

        header = self.table.horizontalHeader()

        for key in dictionary:
            rowPosition = self.table.rowCount()
            self.table.setRowCount(rowPosition + 1)
            self.table.setItem(rowPosition, 0, QTableWidgetItem(key))
            self.table.setItem(rowPosition, 1, QTableWidgetItem(str(round(dictionary.get(key), 6))))

        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        super().show()