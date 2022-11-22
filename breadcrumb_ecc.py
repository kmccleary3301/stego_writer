from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMenu
from PyQt5.QtGui import QIcon
import stego
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

        self.base_keys = [None for i in range(4)]
        self.threshold = None
        self.gap = None
        self.sanitize_second_step = None


        self.file_string = None

        self.file_name = None
        self.file_content = None
        self.encoding = None

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

        update_params_action = QAction("Update Params", self)
        update_params_action.triggered.connect(self.update_params)
        keyMenu.addAction(update_params_action)

        show_noise_action = QAction("Show Noise", self)
        show_noise_action.triggered.connect(self.showNoise)
        imageMenu.addAction(show_noise_action)

        save_noise_action = QAction("Save Noise", self)
        save_noise_action.triggered.connect(self.saveNoise)
        imageMenu.addAction(save_noise_action)

        load_file_string_action = QAction("Load File To String", self)
        load_file_string_action.triggered.connect(self.load_file_as_string)
        msgMenu.addAction(load_file_string_action)

        embed_file_action = QAction("Embed File String Into Image", self)
        embed_file_action.triggered.connect(self.embed_file_string_into_image)
        msgMenu.addAction(embed_file_action)

        decode_file_from_image_action = QAction("Decode File From Image", self)
        decode_file_from_image_action.triggered.connect(self.decode_file_from_image)
        msgMenu.addAction(decode_file_from_image_action)

        save_file_from_data_action = QAction("Save File", self)
        save_file_from_data_action.triggered.connect(self.save_file_data)
        msgMenu.addAction(save_file_from_data_action)

    def saveImg(self):
        self.save_file_name,_ = QFileDialog.getSaveFileName(self, "Save file", "", "Images (*.png)")
        if not self.save_file_name:
            return
        cv2.imwrite(self.save_file_name, self.image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def embedImageWithMsg(self):
        self.msg = self.editor.toPlainText()
        self.embedImageString(self.msg)

    def embedImageString(self, message):
        if self.image is None:
            self.openImage()
        if self.ECCKeys is None:
            self.ECCKeys = ECC.getPubPrivKeys()

        self.msg = message

        if self.pubKey is None:
            gMsg = ECC.encrypt_Plain(self.msg, self.ECCKeys[1])
        else:
            gMsg = ECC.encrypt_Plain(self.msg, self.pubKey)
        try:
            self.enMsg = ECC.enMsg2Hex(gMsg)
            self.image = stego.image_write_check_readable(self.image, self.enMsg, base_key=self.base_keys[0],
                            string_shuffle_key=self.base_keys[1], custom_key_shuffle_key=self.base_keys[2],
                            initial_points_shuffle_key=self.base_keys[3], threshold=self.threshold, gap=self.gap,
                            sanitize_second_step=self.sanitize_second_step)
        except:
            e = sys.exc_info()[0]
            self.editor.setPlainText("<p>Error: %s</p>" % e)

    def decodeMsg(self):
        if self.privReadKey is None:
            self.readPKey()
        try:
            readMsg = stego.image_read_processing(self.image, base_key=self.base_keys[0],
                            string_shuffle_key=self.base_keys[1], custom_key_shuffle_key=self.base_keys[2],
                            initial_points_shuffle_key=self.base_keys[3], threshold=self.threshold, gap=self.gap)
            self.enMsg = ECC.hex2EnMsg(readMsg)
            self.msg = ECC.decrypt_Plain(self.enMsg, self.privReadKey)
            self.editor.setPlainText(self.msg)
        except:
            e = sys.exc_info()[0]
            self.editor.setPlainText("<p>Error: %s</p>" % e)

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

    def update_params(self):
        for i in range(len(self.base_keys)):
            prev_value = self.base_keys[i]
            self.base_keys[i], ok_pressed = QInputDialog.getInt(self, "Get integer", "Base Key %i:" % (i), 0, 0,
                                              1000000000, 1)
            if not ok_pressed:
                self.base_keys[i] = prev_value
        self.threshold, ok_pressed = QInputDialog.getInt(self, "Get integer", "Threshold:", 50, 5,
                                              100, 1)
        self.gap, ok_pressed = QInputDialog.getInt(self, "Get integer", "Gap:", 15, 5,
                                                               100, 1)
        self.sanitize_second_step, ok_pressed = QInputDialog.getText(self, 'Sanitize Second Step',
                                                                     'Checkerboard Sanitize (T/F):')
        if self.sanitize_second_step == 'T':
            self.sanitize_second_step = True
        else:
            self.sanitize_second_step = False

        for key in self.base_keys:
            print(key)
        print(self.threshold)
        print(self.gap)
        print(self.sanitize_second_step)

    def load_file_as_string(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if not file_path:
            return
        self.file_string = stego.convert_file_to_string_data(file_path)
        self.editor.setPlainText(self.file_string)

    def embed_file_string_into_image(self):
        if not self.file_string:
            self.load_file_as_string()
        self.embedImageString(self.file_string)

    def decode_file_from_image(self):
        if self.privReadKey is None:
            self.readPKey()
        self.decodeMsg()
        self.file_string = self.msg

        self.file_content, self.file_name, self.encoding = stego.convert_string_to_file_data(self.file_string)

    def save_file_data(self):
        if not self.file_name:
            return
        if not self.file_content:
            return
        if not self.encoding:
            return
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if not folder_path:
            folder_path = None
        stego.write_file_from_tuple_data(self.file_name, self.file_content, self.encoding, path_to=folder_path)




App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())