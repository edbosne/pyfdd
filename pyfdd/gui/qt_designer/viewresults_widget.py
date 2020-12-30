# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'viewresults_widget.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ViewResultsWidget(object):
    def setupUi(self, ViewResultsWidget):
        ViewResultsWidget.setObjectName("ViewResultsWidget")
        ViewResultsWidget.resize(1150, 670)
        self.verticalLayout = QtWidgets.QVBoxLayout(ViewResultsWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mplframe = QtWidgets.QFrame(ViewResultsWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mplframe.sizePolicy().hasHeightForWidth())
        self.mplframe.setSizePolicy(sizePolicy)
        self.mplframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mplframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mplframe.setObjectName("mplframe")
        self.mpl_hlayout = QtWidgets.QHBoxLayout(self.mplframe)
        self.mpl_hlayout.setObjectName("mpl_hlayout")
        self.verticalLayout.addWidget(self.mplframe)
        self.ResultsTable = QtWidgets.QTableView(ViewResultsWidget)
        self.ResultsTable.setObjectName("ResultsTable")
        self.verticalLayout.addWidget(self.ResultsTable)

        self.retranslateUi(ViewResultsWidget)
        QtCore.QMetaObject.connectSlotsByName(ViewResultsWidget)

    def retranslateUi(self, ViewResultsWidget):
        _translate = QtCore.QCoreApplication.translate
        ViewResultsWidget.setWindowTitle(_translate("ViewResultsWidget", "Form"))
