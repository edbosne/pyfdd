import sys
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5 import QtCore, QtGui, QtWidgets
from pyfdd.gui.mainwindow import WindowedPyFDD


def run_fbs():
    ctx = ApplicationContext()
    window = WindowedPyFDD()
    window.show()
    sys.exit(ctx.app.exec_())


def run():
    app = QtWidgets.QApplication(sys.argv)
    window = WindowedPyFDD()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()