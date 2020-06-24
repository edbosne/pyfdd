# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'libexplorer_widget.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DataPatternWidget(object):
    def setupUi(self, DataPatternWidget):
        DataPatternWidget.setObjectName("DataPatternWidget")
        DataPatternWidget.resize(1150, 670)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DataPatternWidget.sizePolicy().hasHeightForWidth())
        DataPatternWidget.setSizePolicy(sizePolicy)
        self.horizontalLayout = QtWidgets.QHBoxLayout(DataPatternWidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mplwindow = QtWidgets.QWidget(DataPatternWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mplwindow.sizePolicy().hasHeightForWidth())
        self.mplwindow.setSizePolicy(sizePolicy)
        self.mplwindow.setMinimumSize(QtCore.QSize(550, 550))
        self.mplwindow.setMaximumSize(QtCore.QSize(6000, 6000))
        self.mplwindow.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.mplwindow.setAutoFillBackground(False)
        self.mplwindow.setObjectName("mplwindow")
        self.mplvl = QtWidgets.QHBoxLayout(self.mplwindow)
        self.mplvl.setContentsMargins(9, 9, 9, 9)
        self.mplvl.setSpacing(6)
        self.mplvl.setObjectName("mplvl")
        self.horizontalLayout.addWidget(self.mplwindow)
        self.widget_2 = QtWidgets.QWidget(DataPatternWidget)
        self.widget_2.setMaximumSize(QtCore.QSize(340, 16777215))
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.info_groupBox = QtWidgets.QGroupBox(self.widget_2)
        self.info_groupBox.setObjectName("info_groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.info_groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.infotext = QtWidgets.QTextBrowser(self.info_groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.infotext.sizePolicy().hasHeightForWidth())
        self.infotext.setSizePolicy(sizePolicy)
        self.infotext.setMaximumSize(QtCore.QSize(16777215, 80))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.infotext.setFont(font)
        self.infotext.setAcceptRichText(True)
        self.infotext.setObjectName("infotext")
        self.verticalLayout_3.addWidget(self.infotext)
        self.verticalLayout_2.addWidget(self.info_groupBox)
        self.groupBox = QtWidgets.QGroupBox(self.widget_2)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.listWidget = QtWidgets.QListWidget(self.groupBox)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_4.addWidget(self.listWidget)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.pb_setticks = QtWidgets.QGroupBox(self.widget_2)
        self.pb_setticks.setObjectName("pb_setticks")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.pb_setticks)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pb_colorscale = QtWidgets.QPushButton(self.pb_setticks)
        self.pb_colorscale.setObjectName("pb_colorscale")
        self.verticalLayout.addWidget(self.pb_colorscale)
        self.pb_setlabels = QtWidgets.QPushButton(self.pb_setticks)
        self.pb_setlabels.setObjectName("pb_setlabels")
        self.verticalLayout.addWidget(self.pb_setlabels)
        self.verticalLayout_2.addWidget(self.pb_setticks)
        self.horizontalLayout.addWidget(self.widget_2)

        self.retranslateUi(DataPatternWidget)
        QtCore.QMetaObject.connectSlotsByName(DataPatternWidget)

    def retranslateUi(self, DataPatternWidget):
        _translate = QtCore.QCoreApplication.translate
        DataPatternWidget.setWindowTitle(_translate("DataPatternWidget", "Form"))
        self.info_groupBox.setTitle(_translate("DataPatternWidget", "Info"))
        self.infotext.setHtml(_translate("DataPatternWidget", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Noto Sans\';\">Pattern dimentions (nx, ny): 0, 0</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Noto Sans\';\">Angular step (x, y): 0, 0</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Noto Sans\';\">Angular range (x, y): 0, 0</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Number of simulations: 0</p></body></html>"))
        self.groupBox.setTitle(_translate("DataPatternWidget", "Simulations"))
        self.pb_setticks.setTitle(_translate("DataPatternWidget", "Pattern Visualization"))
        self.pb_colorscale.setText(_translate("DataPatternWidget", "Edit Color Scale"))
        self.pb_setlabels.setText(_translate("DataPatternWidget", "Set Labels"))
