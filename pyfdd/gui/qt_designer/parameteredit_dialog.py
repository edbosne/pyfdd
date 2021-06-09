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
        self.le_initial_value = QtWidgets.QLineEdit(self.widget)
        self.le_initial_value.setObjectName("le_initial_value")
        self.gridLayout.addWidget(self.le_initial_value, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.le_range_min = QtWidgets.QLineEdit(self.widget)
        self.le_range_min.setObjectName("le_range_min")
        self.gridLayout.addWidget(self.le_range_min, 1, 1, 1, 1)
        self.le_range_max = QtWidgets.QLineEdit(self.widget)
        self.le_range_max.setObjectName("le_range_max")
        self.gridLayout.addWidget(self.le_range_max, 2, 1, 1, 1)
        self.le_step_mod = QtWidgets.QLineEdit(self.widget)
        self.le_step_mod.setObjectName("le_step_mod")
        self.gridLayout.addWidget(self.le_step_mod, 3, 1, 1, 1)
        self.cb_fixed = QtWidgets.QCheckBox(self.widget)
        self.cb_fixed.setText("")
        self.cb_fixed.setChecked(False)
        self.cb_fixed.setObjectName("cb_fixed")
        self.gridLayout.addWidget(self.cb_fixed, 4, 1, 1, 1)
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
        ParameterEditDialog.setWindowTitle(_translate("ParameterEditDialog", "Fit Parameter"))
        self.label_2.setText(_translate("ParameterEditDialog", "Range min."))
        self.label_4.setText(_translate("ParameterEditDialog", "Step modifier"))
        self.label.setText(_translate("ParameterEditDialog", "Initial value"))
        self.label_3.setText(_translate("ParameterEditDialog", "Range max."))
        self.label_5.setText(_translate("ParameterEditDialog", "Fixed"))
