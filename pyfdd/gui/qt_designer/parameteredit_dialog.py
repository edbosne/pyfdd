# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'parameteredit_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ParameterEditDialog(object):
    def setupUi(self, ParameterEditDialog):
        ParameterEditDialog.setObjectName("ParameterEditDialog")
        ParameterEditDialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(ParameterEditDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(ParameterEditDialog)
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 3, 1, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(self.widget)
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout.addWidget(self.checkBox, 4, 1, 1, 1)
        self.verticalLayout.addWidget(self.widget)
        self.buttonBox = QtWidgets.QDialogButtonBox(ParameterEditDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(ParameterEditDialog)
        self.buttonBox.accepted.connect(ParameterEditDialog.accept)
        self.buttonBox.rejected.connect(ParameterEditDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(ParameterEditDialog)

    def retranslateUi(self, ParameterEditDialog):
        _translate = QtCore.QCoreApplication.translate
        ParameterEditDialog.setWindowTitle(_translate("ParameterEditDialog", "Dialog"))
        self.label_2.setText(_translate("ParameterEditDialog", "Range min."))
        self.label_4.setText(_translate("ParameterEditDialog", "Step modifier"))
        self.label.setText(_translate("ParameterEditDialog", "Initial value"))
        self.label_3.setText(_translate("ParameterEditDialog", "Range max."))
        self.label_5.setText(_translate("ParameterEditDialog", "Optimize"))
        self.checkBox.setText(_translate("ParameterEditDialog", "CheckBox"))
