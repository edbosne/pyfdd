# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'windowedpyfdd.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_WindowedPyFDD(object):
    def setupUi(self, WindowedPyFDD):
        WindowedPyFDD.setObjectName("WindowedPyFDD")
        WindowedPyFDD.resize(1150, 670)
        self.centralwidget = QtWidgets.QWidget(WindowedPyFDD)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.maintabs = QtWidgets.QTabWidget(self.centralwidget)
        self.maintabs.setObjectName("maintabs")
        self.gridLayout.addWidget(self.maintabs, 0, 0, 1, 1)
        WindowedPyFDD.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(WindowedPyFDD)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1150, 22))
        self.menubar.setObjectName("menubar")
        WindowedPyFDD.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(WindowedPyFDD)
        self.statusbar.setObjectName("statusbar")
        WindowedPyFDD.setStatusBar(self.statusbar)

        self.retranslateUi(WindowedPyFDD)
        self.maintabs.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(WindowedPyFDD)

    def retranslateUi(self, WindowedPyFDD):
        _translate = QtCore.QCoreApplication.translate
        WindowedPyFDD.setWindowTitle(_translate("WindowedPyFDD", "PyFDD"))
