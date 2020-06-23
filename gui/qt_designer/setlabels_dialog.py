# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'setlabels_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SetLabelsDialog(object):
    def setupUi(self, SetLabelsDialog):
        SetLabelsDialog.setObjectName("SetLabelsDialog")
        SetLabelsDialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(SetLabelsDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(SetLabelsDialog)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.le_title = QtWidgets.QLineEdit(SetLabelsDialog)
        self.le_title.setObjectName("le_title")
        self.verticalLayout.addWidget(self.le_title)
        self.label_2 = QtWidgets.QLabel(SetLabelsDialog)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.le_x_axis = QtWidgets.QLineEdit(SetLabelsDialog)
        self.le_x_axis.setObjectName("le_x_axis")
        self.verticalLayout.addWidget(self.le_x_axis)
        self.label_3 = QtWidgets.QLabel(SetLabelsDialog)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.le_y_axis = QtWidgets.QLineEdit(SetLabelsDialog)
        self.le_y_axis.setObjectName("le_y_axis")
        self.verticalLayout.addWidget(self.le_y_axis)
        self.label_4 = QtWidgets.QLabel(SetLabelsDialog)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.le_z_axis = QtWidgets.QLineEdit(SetLabelsDialog)
        self.le_z_axis.setObjectName("le_z_axis")
        self.verticalLayout.addWidget(self.le_z_axis)
        self.buttonBox = QtWidgets.QDialogButtonBox(SetLabelsDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(SetLabelsDialog)
        self.buttonBox.accepted.connect(SetLabelsDialog.accept)
        self.buttonBox.rejected.connect(SetLabelsDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(SetLabelsDialog)

    def retranslateUi(self, SetLabelsDialog):
        _translate = QtCore.QCoreApplication.translate
        SetLabelsDialog.setWindowTitle(_translate("SetLabelsDialog", "Set Labels"))
        self.label.setText(_translate("SetLabelsDialog", "Title"))
        self.label_2.setText(_translate("SetLabelsDialog", "x-axis label"))
        self.label_3.setText(_translate("SetLabelsDialog", "y-axis label"))
        self.label_4.setText(_translate("SetLabelsDialog", "z-axis label"))

