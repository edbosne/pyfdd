from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys

from qt_designer.datapattern_widget import Ui_Form


class DataPattern_widget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, *args, obj=None, **kwargs):
        super(DataPattern_widget, self).__init__(*args, **kwargs)
        # Convert .ui file to .py with,
        # pyuic5 datapattern-widget.ui -o datapattern-widget.py
        #uic.loadUi('datapattern-widget.ui', self)
        self.setupUi(self)
        #self.show()


app = QtWidgets.QApplication(sys.argv)
window = DataPattern_widget()
window.show()
sys.exit(app.exec())
