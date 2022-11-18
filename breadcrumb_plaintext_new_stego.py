from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication
import new_stego_v1 as stego
import sys, cv2

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breadcrumb")
        self.setGeometry(570, 300, 780, 480)
        #self.setWindowIcon(QIcon('duck.ico'))
        self.bitmap = None
        self.blobData = None
        self.blobExpanded = None
        self.noiseMap = None
        self.imFilePath = ''
        self.image = None
        self.msg = ''
        self.UiComponents()
        self.show()

    def UiComponents(self):
        self.layout = QVBoxLayout()
        self.status = QStatusBar()
        self.menubar = self.menuBar()
        self.setStatusBar(self.status)
        self.editor = QPlainTextEdit()
        self.layout.addWidget(self.editor)
        container = QWidget()
        container.setLayout(self.layout)

        # making container as central widget
        self.setCentralWidget(container)

        save_image_action = QAction("Save Image", self)
        save_image_action.triggered.connect(self.saveImg)
        self.menubar.addAction(save_image_action)

        embed_action = QAction("Embed Image", self)
        embed_action.triggered.connect(self.embedImageWithMsg)
        self.menubar.addAction(embed_action)

        open_img_action = QAction("Open Image", self)
        open_img_action.triggered.connect(self.openImage)
        self.menubar.addAction(open_img_action)

        show_img_action = QAction("Show Image", self)
        show_img_action.triggered.connect(self.showImage)
        self.menubar.addAction(show_img_action)

        decode_msg_action = QAction("Decode Msg", self)
        decode_msg_action.triggered.connect(self.decodeMsg)
        self.menubar.addAction(decode_msg_action)

        show_noise_action = QAction("Show Noise", self)
        show_noise_action.triggered.connect(self.showNoiseMap)
        self.menubar.addAction(show_noise_action)

        save_noise_action = QAction("Save Noise", self)
        save_noise_action.triggered.connect(self.saveNoise)
        self.menubar.addAction(save_noise_action)

    def saveImg(self):
        self.save_file_name,_ = QFileDialog.getSaveFileName(self, "Save file", "", "Images (*.png)")
        if not self.save_file_name:
            return
        cv2.imwrite(self.save_file_name, self.image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def embedImageWithMsg(self):
        self.msg = self.editor.toPlainText()
        self.image = stego.image_write_processing(self.image, self.msg)

    def decodeMsg(self):
        try:
            self.msg = stego.image_read_processing(self.image)
            self.editor.setPlainText(self.msg)
        except:
            e = sys.exc_info()[0]
            self.editor.setPlainTest("<p>Error: %s</p>" % e)

    def openImage(self):
        self.file_path, filter_type = QFileDialog.getOpenFileName(self, "Open new file", "",
                                                                  "Images (*.png *.jpeg *.jpg *.bmp *.gif)")
        if not self.file_path:
            return
        self.image = cv2.imread(self.file_path)
        self.noiseMap = None

    def showImage(self):
        cv2.imshow('Image', self.image)
        cv2.waitKey(0)

    def saveNoise(self):
        self.save_file_name, _ = QFileDialog.getSaveFileName(self, "Save file", "", "Images (*.png)")
        if not self.save_file_name:
            return
        cv2.imwrite(self.save_file_name, self.noiseMap, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def showNoiseMap(self):
        if self.noiseMap is None:
            #self.noiseMap = tW.genImageMap(self.image)
            self.noiseMap = stego.image_lsb_display(self.image)
        cv2.imshow('NoiseMap',self.noiseMap)
        cv2.waitKey(0)


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())