from PyQt5 import uic, QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit, QComboBox, QLabel, QSpinBox, QPushButton, QMessageBox
from save_classificator import *
from predict import *
from fit_classificator import *
from selection_of_terms import *
from table_widget import TableWidget
from tf_idf import *
from work_with_file import *
from save_classificator import *
from predict import *



class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("MainWindow.ui", self)
        self.input_text: QLineEdit = self.findChild(QLineEdit, 'lineEdit')
        self.select_classificator: QComboBox = self.findChild(QComboBox, 'comboBox')
        self.classificate: QPushButton = self.findChild(QPushButton, 'pushButton')
        self.table: QPushButton = self.findChild(QPushButton, 'pushButton_2')
        self.show_result: QLabel = self.findChild(QLabel, 'label')
        self.table_widget = TableWidget(self)
        self.dictionary = None

    @pyqtSlot(name='on_pushButton_clicked')
    def find_result(self):
        self.dictionary = None
        text = self.input_text.text()
        clafficator = self.select_classificator.currentText()
        if (clafficator == "Наивный Байесовский классификатор"):
            name_file = "clf_NB2.pkl"
        else:
            name_file = "clf_LR2.pkl"

        clf = load_clf(name_file)
        description = []
        description.append(text)
        print(description)
        filtered_example = filter(description)
        print(filtered_example)
        res_example = tf_idf(filtered_example)
        predicted, name, dictionary = predict(clf, res_example)
        self.dictionary = dictionary
        print("dict: ", self.dictionary)

        print(dictionary[predicted] < 0.5)
        if (float(dictionary[predicted]) < 0.3):
            self.show_result.setText(predicted + " " + name)
            QMessageBox.about(self, "Error!", "Недостаточная точность!")

        else:
            self.show_result.setText(predicted + " " + name)

    @pyqtSlot(name='on_pushButton_2_clicked')
    def show_table(self):
        self.table_widget.show(self.dictionary)
