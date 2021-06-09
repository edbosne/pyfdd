# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editorientation_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_EditOrientationDialog(object):
    def setupUi(self, EditOrientationDialog):
        EditOrientationDialog.setObjectName("EditOrientationDialog")
        EditOrientationDialog.resize(400, 174)
        self.verticalLayout = QtWidgets.QVBoxLayout(EditOrientationDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(EditOrientationDialog)
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setObjectName("gridLayout")
        self.le_phi = QtWidgets.QLineEdit(self.widget)
        self.le_phi.setMaximumSize(QtCore.QSize(50, 16777215))
        self.le_phi.setObjectName("le_phi")
        self.gridLayout.addWidget(self.le_phi, 1, 5, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.le_y = QtWidgets.QLineEdit(self.widget)
        self.le_y.setMaximumSize(QtCore.QSize(50, 16777215))
        self.le_y.setObjectName("le_y")
        self.gridLayout.addWidget(self.le_y, 1, 3, 1, 1)
        self.le_x = QtWidgets.QLineEdit(self.widget)
        self.le_x.setMaximumSize(QtCore.QSize(50, 16777215))
        self.le_x.setObjectName("le_x")
        self.gridLayout.addWidget(self.le_x, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setMaximumSize(QtCore.QSize(5, 16777215))
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setMaximumSize(QtCore.QSize(5, 16777215))
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 4, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 6, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setMaximumSize(QtCore.QSize(16777215, 28))
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 7)
        self.verticalLayout.addWidget(self.widget)
        self.buttonBox = QtWidgets.QDialogButtonBox(EditOrientationDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(EditOrientationDialog)
        self.buttonBox.accepted.connect(EditOrientationDialog.accept)
        self.buttonBox.rejected.connect(EditOrientationDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(EditOrientationDialog)

    def retranslateUi(self, EditOrientationDialog):
        _translate = QtCore.QCoreApplication.translate
        EditOrientationDialog.setWindowTitle(_translate("EditOrientationDialog", "Pattern Orientation"))
        self.le_phi.setText(_translate("EditOrientationDialog", "0"))
        self.label.setText(_translate("EditOrientationDialog", "(x, y, phi)  ="))
        self.le_y.setText(_translate("EditOrientationDialog", "0"))
        self.le_x.setText(_translate("EditOrientationDialog", "0"))
        self.label_3.setText(_translate("EditOrientationDialog", ","))
        self.label_2.setText(_translate("EditOrientationDialog", ","))
        self.label_4.setText(_translate("EditOrientationDialog", "Edit the (x, y, phi) orientation of the main channeling axis."))
