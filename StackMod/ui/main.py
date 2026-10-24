"""Desktop Application entrypoint."""
from __future__ import annotations

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import AppConfig
from ui.main_window import MainWindow


def main() -> None:
    # High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("AIFaceStudio")
    app.setOrganizationName("AIFaceStudio")
    app.setFont(QFont("Segoe UI", 10))

    cfg_path = os.path.join("data", "config.json")
    cfg = AppConfig.load_json(cfg_path) if os.path.exists(cfg_path) else AppConfig()

    window = MainWindow(cfg)
    window.show()

    exit_code = app.exec()
    window.cfg.save_json(cfg_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
