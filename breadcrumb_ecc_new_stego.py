from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMenu
from PyQt5.QtGui import QIcon
import new_stego_v1 as stego
import sys, cv2
import ECC

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breadcrumb")
        self.setGeometry(570, 300, 780, 480)
        self.setWindowIcon(QIcon('duck.ico'))
        self.imFilePath = ''
        self.image, self.noiseImg = None, None
        self.privReadKey = None
        self.pubKey = None
        self.ECCKeys = ECC.getPubPrivKeys()
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

        self.setCentralWidget(container)

        imageMenu = QMenu('&Images', self)
        msgMenu = QMenu('&Message', self)
        keyMenu = QMenu('&Keys', self)
        self.menubar.addMenu(imageMenu)
        self.menubar.addMenu(msgMenu)
        self.menubar.addMenu(keyMenu)

        save_image_action = QAction("Save Image", self)
        save_image_action.triggered.connect(self.saveImg)

        embed_action = QAction("Embed Message Into Image", self)
        embed_action.triggered.connect(self.embedImageWithMsg)
        msgMenu.addAction(embed_action)

        open_img_action = QAction("Open Image", self)
        open_img_action.triggered.connect(self.openImage)
        imageMenu.addAction(open_img_action)
        imageMenu.addAction(save_image_action)

        show_img_action = QAction("Show Image", self)
        show_img_action.triggered.connect(self.showImage)
        imageMenu.addAction(show_img_action)

        decode_msg_action = QAction("Decode Message From Image", self)
        decode_msg_action.triggered.connect(self.decodeMsg)
        msgMenu.addAction(decode_msg_action)

        save_keys_action = QAction("Save Private Key", self)
        save_keys_action.triggered.connect(self.saveECCKeys)
        keyMenu.addAction(save_keys_action)

        read_pKey_action = QAction("Open Private Key", self)
        read_pKey_action.triggered.connect(self.readPKey)
        keyMenu.addAction(read_pKey_action)

        save_pubkeys_action = QAction("Save Public Key", self)
        save_pubkeys_action.triggered.connect(self.savePubKey)
        keyMenu.addAction(save_pubkeys_action)

        read_pubKey_action = QAction("Open Public Key", self)
        read_pubKey_action.triggered.connect(self.getPubKey)
        keyMenu.addAction(read_pubKey_action)

        show_noise_action = QAction("Show Noise", self)
        show_noise_action.triggered.connect(self.showNoise)
        imageMenu.addAction(show_noise_action)

        save_noise_action = QAction("Save Noise", self)
        save_noise_action.triggered.connect(self.saveNoise)
        imageMenu.addAction(save_noise_action)

    def saveImg(self):
        self.save_file_name,_ = QFileDialog.getSaveFileName(self, "Save file", "", "Images (*.png)")
        if not self.save_file_name:
            return
        cv2.imwrite(self.save_file_name, self.image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def embedImageWithMsg(self):
        if self.image is None:
            self.openImage()
        if self.ECCKeys is None:
            self.ECCKeys = ECC.getPubPrivKeys()
        self.msg = self.editor.toPlainText()
        if self.pubKey is None:
            gMsg = ECC.encrypt_Plain(self.msg, self.ECCKeys[1])
        else:
            gMsg = ECC.encrypt_Plain(self.msg, self.pubKey)
        try:
            self.enMsg = ECC.enMsg2Hex(gMsg)
            self.image = stego.image_write_processing(self.image, self.enMsg)
        except:
            self.editor.setPlainTest("Error")

    def decodeMsg(self):
        if self.privReadKey is None:
            self.readPKey()
        try:
            readMsg = stego.image_read_processing(self.image)
            self.enMsg = ECC.hex2EnMsg(readMsg)
            self.msg = ECC.decrypt_Plain(self.enMsg, self.privReadKey)
            self.editor.setPlainText(self.msg)
        except:
            self.editor.setPlainTest("Error")

    def openImage(self):
        self.file_path, filter_type = QFileDialog.getOpenFileName(self, "Open new file", "",
                                                                  "Images (*.png *.jpeg *.jpg *.bmp *.gif)")
        if not self.file_path:
            return
        self.image = cv2.imread(self.file_path)

    def showImage(self):
        cv2.imshow('Image', self.image)
        cv2.waitKey(0)

    def saveECCKeys(self):
        keypath,_ = QFileDialog.getSaveFileName(self, "Save file", "", "txt (*.txt)")
        if not keypath:
            return
        with open(keypath, 'w') as f:
            f.write(hex(self.ECCKeys[0]))

    def readPKey(self):
        keyFilePath,_ = QFileDialog.getOpenFileName(self, "Open new file", "",
                                                                  "Text (*.txt)")
        if not keyFilePath:
            return
        with open(keyFilePath, 'r') as f:
            self.privReadKey = int(f.readline(), 16)

    def showNoise(self):
        self.noiseImg = stego.image_lsb_display(self.image)
        cv2.imshow('Noise Img', self.noiseImg)
        cv2.waitKey(0)

    def saveNoise(self):
        if self.noiseImg is None:
            self.noiseImg = stego.image_lsb_display(self.image)
        self.save_file_name, _ = QFileDialog.getSaveFileName(self, "Save file", "", "Images (*.png)")
        if not self.save_file_name:
            return
        cv2.imwrite(self.save_file_name, self.noiseImg, [cv2.IMWRITE_PNG_COMPRESSION, 9])


    def savePubKey(self):
        keypath, _ = QFileDialog.getSaveFileName(self, "Save file", "", "txt (*.txt)")
        keyTextMake = ECC.makePubKeyText(self.ECCKeys[1])
        if not keypath:
            return
        with open(keypath, 'w') as f:
            f.write(keyTextMake)

    def getPubKey(self):
        keyFilePath, _ = QFileDialog.getOpenFileName(self, "Open new file", "",
                                                     "Text (*.txt)")
        if not keyFilePath:
            return
        strGet = ''
        with open(keyFilePath, 'r') as f:
            strGet += f.readline()
            strGet += f.readline()
        self.pubKey = ECC.pubKeyFromText(strGet)
        print(self.pubKey)

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())