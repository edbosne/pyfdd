# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'buildmesh_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_BuildMeshDialog(object):
    def setupUi(self, BuildMeshDialog):
        BuildMeshDialog.setObjectName("BuildMeshDialog")
        BuildMeshDialog.resize(401, 300)
        BuildMeshDialog.setWindowOpacity(1.0)
        self.verticalLayout = QtWidgets.QVBoxLayout(BuildMeshDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(BuildMeshDialog)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.le_pixelsize = QtWidgets.QLineEdit(self.groupBox)
        self.le_pixelsize.setMaximumSize(QtCore.QSize(50, 16777215))
        self.le_pixelsize.setObjectName("le_pixelsize")
        self.horizontalLayout.addWidget(self.le_pixelsize)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.le_distance = QtWidgets.QLineEdit(self.groupBox)
        self.le_distance.setMaximumSize(QtCore.QSize(50, 16777215))
        self.le_distance.setObjectName("le_distance")
        self.horizontalLayout.addWidget(self.le_distance)
        self.rb_detector = QtWidgets.QRadioButton(self.groupBox)
        self.rb_detector.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_detector.sizePolicy().hasHeightForWidth())
        self.rb_detector.setSizePolicy(sizePolicy)
        self.rb_detector.setChecked(True)
        self.rb_detector.setObjectName("rb_detector")
        self.horizontalLayout.addWidget(self.rb_detector)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(BuildMeshDialog)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.le_angstep = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_angstep.setMaximumSize(QtCore.QSize(50, 16777215))
        self.le_angstep.setObjectName("le_angstep")
        self.horizontalLayout_2.addWidget(self.le_angstep)
        self.rb_step = QtWidgets.QRadioButton(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rb_step.sizePolicy().hasHeightForWidth())
        self.rb_step.setSizePolicy(sizePolicy)
        self.rb_step.setObjectName("rb_step")
        self.horizontalLayout_2.addWidget(self.rb_step)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.buttonBox = QtWidgets.QDialogButtonBox(BuildMeshDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(BuildMeshDialog)
        self.buttonBox.accepted.connect(BuildMeshDialog.accept)
        self.buttonBox.rejected.connect(BuildMeshDialog.reject)
        self.rb_detector.clicked.connect(self.rb_step.toggle)
        self.rb_step.clicked.connect(self.rb_detector.toggle)
        QtCore.QMetaObject.connectSlotsByName(BuildMeshDialog)

    def retranslateUi(self, BuildMeshDialog):
        _translate = QtCore.QCoreApplication.translate
        BuildMeshDialog.setWindowTitle(_translate("BuildMeshDialog", "Angular Mesh"))
        self.groupBox.setTitle(_translate("BuildMeshDialog", "Use detector configuration"))
        self.label_2.setText(_translate("BuildMeshDialog", "Pixel size"))
        self.le_pixelsize.setText(_translate("BuildMeshDialog", "0.055"))
        self.label_3.setText(_translate("BuildMeshDialog", "Distance"))
        self.le_distance.setText(_translate("BuildMeshDialog", "315"))
        self.rb_detector.setText(_translate("BuildMeshDialog", "Selected"))
        self.groupBox_2.setTitle(_translate("BuildMeshDialog", "Use angular step configuration"))
        self.label_5.setText(_translate("BuildMeshDialog", "Angular step (degrees)"))
        self.le_angstep.setText(_translate("BuildMeshDialog", "0.1"))
        self.rb_step.setText(_translate("BuildMeshDialog", "Selected"))
