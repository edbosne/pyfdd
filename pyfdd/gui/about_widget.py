import sys
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QDialogButtonBox
from PyQt5.QtGui import QFont, QDesktopServices
from PyQt5.QtCore import Qt, QUrl

import pyfdd


class AboutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("About PyFDD")
        self.setFixedSize(400, 200)

        APP_NAME = "PyFDD"
        VERSION_NUMBER = pyfdd.__version__
        COPYRIGHT = "GPL3"
        DEVELOPER = "E David-Bosne"
        WEBSITE = "https://github.com/edbosne/pyfdd"

        layout = QVBoxLayout()

        # Name label
        name_label = QLabel(f"<h2>{APP_NAME}</h2>")
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)

        # Version label
        version_label = QLabel(f"Version: {VERSION_NUMBER}")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        # Copyright label
        copyright_label = QLabel(f"Copyright © {COPYRIGHT}")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)

        # Developer label
        developer_label = QLabel(f"Developer: {DEVELOPER}")
        developer_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(developer_label)

        # Website label
        website_label = QLabel(f"<a href=\"{WEBSITE}\">{WEBSITE}</a>")
        website_label.setAlignment(Qt.AlignCenter)
        website_label.setOpenExternalLinks(True)
        layout.addWidget(website_label)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    about_dialog = AboutDialog()
    about_dialog.show()

    sys.exit(app.exec_())
