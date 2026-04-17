import sys
from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Neural Brain")
    print("Neural Brain starting...")
    sys.exit(0)

if __name__ == "__main__":
    main()
