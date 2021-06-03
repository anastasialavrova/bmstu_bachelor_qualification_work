import sys
from PyQt5.QtWidgets import QApplication
from fit_classificator import *
from main_window import MainWindow
from selection_of_terms import *
from tf_idf import *
from work_with_file import *
from save_classificator import *
from predict import *


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())




