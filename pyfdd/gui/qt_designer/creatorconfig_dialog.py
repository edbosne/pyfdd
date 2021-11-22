# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'creatorconfig_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CreatorConfigDialog(object):
    def setupUi(self, CreatorConfigDialog):
        CreatorConfigDialog.setObjectName("CreatorConfigDialog")
        CreatorConfigDialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(CreatorConfigDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(CreatorConfigDialog)
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setObjectName("gridLayout")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.cb_gen_method = QtWidgets.QComboBox(self.widget)
        self.cb_gen_method.setObjectName("cb_gen_method")
        self.cb_gen_method.addItem("")
        self.cb_gen_method.addItem("")
        self.cb_gen_method.addItem("")
        self.cb_gen_method.addItem("")
        self.gridLayout.addWidget(self.cb_gen_method, 3, 1, 1, 1)
        self.sb_numsites = QtWidgets.QSpinBox(self.widget)
        self.sb_numsites.setMinimum(1)
        self.sb_numsites.setMaximum(999)
        self.sb_numsites.setObjectName("sb_numsites")
        self.gridLayout.addWidget(self.sb_numsites, 0, 1, 1, 1)
        self.sb_subpixels = QtWidgets.QSpinBox(self.widget)
        self.sb_subpixels.setMinimum(1)
        self.sb_subpixels.setMaximum(999)
        self.sb_subpixels.setObjectName("sb_subpixels")
        self.gridLayout.addWidget(self.sb_subpixels, 1, 1, 1, 1)
        self.le_normalization = QtWidgets.QLineEdit(self.widget)
        self.le_normalization.setObjectName("le_normalization")
        self.gridLayout.addWidget(self.le_normalization, 2, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.widget)
        self.buttonBox = QtWidgets.QDialogButtonBox(CreatorConfigDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(CreatorConfigDialog)
        self.cb_gen_method.setCurrentIndex(0)
        self.buttonBox.accepted.connect(CreatorConfigDialog.accept)
        self.buttonBox.rejected.connect(CreatorConfigDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(CreatorConfigDialog)

    def retranslateUi(self, CreatorConfigDialog):
        _translate = QtCore.QCoreApplication.translate
        CreatorConfigDialog.setWindowTitle(_translate("CreatorConfigDialog", "Dialog"))
        self.label_5.setText(_translate("CreatorConfigDialog", "Generator method"))
        self.cb_gen_method.setItemText(0, _translate("CreatorConfigDialog", "Channeling yield"))
        self.cb_gen_method.setItemText(1, _translate("CreatorConfigDialog", "Ideal"))
        self.cb_gen_method.setItemText(2, _translate("CreatorConfigDialog", "Poisson noise"))
        self.cb_gen_method.setItemText(3, _translate("CreatorConfigDialog", "Monte Carlo"))
        self.label_3.setText(_translate("CreatorConfigDialog", "Num. sites"))
        self.label.setText(_translate("CreatorConfigDialog", "Normalization"))
        self.label_4.setText(_translate("CreatorConfigDialog", "Sub-pixels"))
