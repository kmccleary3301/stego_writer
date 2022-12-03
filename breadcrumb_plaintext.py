from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import stego
import sys, cv2
import threading
import numpy as np
import ctypes
from time import sleep
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


class image_process(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(float)

    image = None
    image_size_assignment = None
    lsb_layer = None
    progress_values = {
        "value": 0,
        "step_integer": 0
    }
    message = ""
    previous_step = 0
    max_steps = 2

    def reset_values(self):
        self.progress_values = {
            "value": 0,
            "step_integer": 0
        }

    def persistent_value_update(self):
        if (self.previous_step != self.progress_values["step_integer"]):
            self.previous_step = self.progress_values["step_integer"]
            self.progress_values["value"] = 100*self.progress_values["step_integer"] / max(1, self.max_steps-1)
            self.progress.emit(self.progress_values["value"])
            print("presistent new value ->",self.progress_values["value"])

    def calculate_image_assignment(self):
        print("starting new object thread")
        self.reset_values()
        self.image_size_assignment = stego.image_size_assignment(stego.isolate_bit_image(self.image, 7), bar_values=self.progress_values)
        self.finished.emit()
    
    def embed_message_in_image(self, message, threshold=None, blob_size=None, key=None, encryption=None, shuffle_key=None):
        self.reset_values()
        self.message = message
        self.image = stego.image_write_new(self.image, message, shuffle_key=shuffle_key, threshold=threshold, blob_expand_size=blob_size,
                                            bar_values=self.progress_values, size_map=self.image_size_assignment, encryption=encryption,
                                            key=key)
        self.finished.emit()
    
    def read_message_from_image(self, threshold=None, blob_size=None, key=None, encryption=None, shuffle_key=None):
        self.reset_values()
        message = stego.image_read_new(self.image, shuffle_key=shuffle_key, threshold=threshold, blob_expand_size=blob_size,
                                        bar_values=self.progress_values, size_map=self.image_size_assignment, encryption=encryption,
                                        key=key)
        self.message = message
        self.reset_values()
        self.finished.emit()
        return self.message




class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breadcrumb")
        self.setGeometry(570, 300, 780, 480)
        self.bitmap = None
        self.blobData = None
        self.blobExpanded = None
        self.noiseMap = None
        self.imFilePath = ''
        self.image = None
        self.image_size_assignment = None
        self.size_assignment_made = False
        self.msg = ''
        self.p_bar_process_total_steps = 0
        self.bar_update_active = False
        self.shuffle_key = 17876418
        self.threshold = 20
        self.blob_size = 5
        self.smart_cover = False
        self.file_string = None
        self.file_name = None
        self.file_content = None
        self.encoding = None

        self.UiComponents()
        self.show()

    def UiComponents(self):
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.layout = QVBoxLayout()
        self.status = QStatusBar()
        self.menubar = self.menuBar()
        self.setStatusBar(self.status)
        self.editor = QPlainTextEdit()
        self.layout.addWidget(self.editor)
        container = QWidget()
        container.setLayout(self.layout)
        self.p_bar = QProgressBar()
        self.p_bar.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.p_bar)
        self.p_bar_update_thread = None

        #self.p_bar.deleteLater()
        #self.p_bar = None

        self.widget_progress_values = {
            "value": 0,
            "p_bar_label": "",
            "step_integer": 0
        } #this is a dict, so it's mutable and can be effectively passed by reference

        self.p_bar_persistent_thread = QTimer()
        #self.thread = QThread()
        self.image_data_object = image_process()
        self.p_bar_persistent_thread.timeout.connect(self.image_data_object.persistent_value_update)
        self.image_data_object.progress.connect(self.update_p_bar)
        self.p_bar_persistent_thread.start(16)
        self.thread = QThread()
        self.image_data_object.moveToThread(self.thread)

        # making container as central widget
        self.setCentralWidget(container)

        imageMenu = QMenu('&Images', self)
        msgMenu = QMenu('&Message', self)
        self.menubar.addMenu(imageMenu)
        self.menubar.addMenu(msgMenu)

        save_image_action = QAction("Save Image", self)
        save_image_action.triggered.connect(self.saveImg)
        imageMenu.addAction(save_image_action)

        embed_action = QAction("Embed Message Into Image", self)
        #embed_action.triggered.connect(self.embedImageWithMsg)
        embed_action.triggered.connect(self.write_image)
        msgMenu.addAction(embed_action)

        open_img_action = QAction("Open Image", self)
        open_img_action.triggered.connect(self.openImage)
        imageMenu.addAction(open_img_action)

        show_img_action = QAction("Show Image", self)
        show_img_action.triggered.connect(self.showImage)
        imageMenu.addAction(show_img_action)

        decode_msg_action = QAction("Decode Message From Image", self)
        #decode_msg_action.triggered.connect(self.decodeMsg)
        decode_msg_action.triggered.connect(self.read_image)
        msgMenu.addAction(decode_msg_action)

        show_noise_action = QAction("Show Noise", self)
        show_noise_action.triggered.connect(self.showNoiseMap)
        imageMenu.addAction(show_noise_action)

        save_noise_action = QAction("Save Noise", self)
        save_noise_action.triggered.connect(self.saveNoise)
        imageMenu.addAction(save_noise_action)

        show_pool_action = QAction("Show Artifacts", self)
        show_pool_action.triggered.connect(self.show_artifact_map)
        imageMenu.addAction(show_pool_action)

        update_params_action = QAction("Update Params", self)
        update_params_action.triggered.connect(self.update_params)
        imageMenu.addAction(update_params_action)

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

    def set_p_bar(self, label):
        self.p_bar.setValue(0)
        self.p_bar.setFormat(label)
    
    def reset_p_bar(self):
        self.p_bar.setFormat("")
        self.p_bar.setValue(0)
        self.image_data_object.progress_values = {
            "value": 0,
            "step_integer": 0
        }

    def saveImg(self):
        self.save_file_name,_ = QFileDialog.getSaveFileName(self, "Save file", "", "Images (*.png)")
        if not self.save_file_name:
            return
        cv2.imwrite(self.save_file_name, self.image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def openImage(self):
        self.file_path, filter_type = QFileDialog.getOpenFileName(self, "Open new file", "",
                                                                  "Images (*.png *.jpeg *.jpg *.bmp *.gif)")
        if not self.file_path:
            return
        self.update_image(cv2.imread(self.file_path), calc_assignments=True)

    def showImage(self):
        cv2.imshow('Image', self.image)
        cv2.waitKey(0)

    def saveNoise(self):
        self.save_file_name, _ = QFileDialog.getSaveFileName(self, "Save file", "", "Images (*.png)")
        if not self.save_file_name:
            return
        self.noiseMap = stego.convert_255(stego.isolate_bit_image(self.image, 7, return_rgb_image=True))
        cv2.imwrite(self.save_file_name, self.noiseMap, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def showNoiseMap(self):
        if self.image is None:
            return
        self.noiseMap = stego.convert_255(stego.isolate_bit_image(self.image, 7, return_rgb_image=True))
        cv2.imshow('Noise Map',self.noiseMap)
        cv2.waitKey(0)

    def show_artifact_map(self):
        artifact_map_visual = stego.pool_mask_visual(self.image_data_object.image_size_assignment, is_size_assignment=True)
        cv2.imshow("Artifact Map", artifact_map_visual)
        cv2.waitKey(0)

    def update_params(self):
        prev_value = self.shuffle_key
        self.shuffle_key, ok_pressed = QInputDialog.getInt(self, "Get integer", "Shuffle Key:", self.shuffle_key, 0, 2**31 - 3, 1)
        if not ok_pressed:
            self.shuffle_key = prev_value
        self.threshold, ok_pressed = QInputDialog.getInt(self, "Get integer", "Threshold:", self.threshold, 5, 100, 1)
        self.blob_size, ok_pressed = QInputDialog.getInt(self, "Get integer", "Blob Thickness:", self.blob_size, 5, 100, 1)
        self.smart_cover, ok_pressed = QInputDialog.getText(self, 'Smart Cover', 'Smart Cover (T/F):')
        if self.smart_cover == 'T':
            self.smart_cover = True
        elif self.smart_cover == 'F':
            self.smart_cover = False

    def update_message(self, message=None):
        if not message is None:
            self.editor.setPlainText(message)

    def update_image_direct(self):
        self.image = self.image_data_object.image
    
    def update_p_bar(self):
        val_set = min(100, int(self.image_data_object.progress_values["value"]))
        self.p_bar.setValue(val_set)

    def update_image(self, image, calc_assignments=None):
        self.image_data_object.image = image
        if calc_assignments is None:
            calc_assignments = True
        if calc_assignments:
            self.set_p_bar("Calculating Image Assignments")
            self.image_data_object.max_steps = 7
            self.thread.started.connect(self.image_data_object.calculate_image_assignment)
            self.image_data_object.finished.connect(self.finish_image_update)
            self.thread.start()
    
    def write_image(self):
        self.msg = self.editor.toPlainText()
        if self.msg == "":
            return
        self.set_p_bar("Writing To Image")
        self.image_data_object.max_steps = 7
        self.thread.started.connect(lambda: self.image_data_object.embed_message_in_image(
            self.msg, shuffle_key=self.shuffle_key, threshold=self.threshold,
            blob_size=self.blob_size
        ))
        self.image_data_object.finished.connect(self.finish_image_write)
        self.thread.start()

    def read_image(self):
        self.msg = self.editor.toPlainText()
        self.set_p_bar("Reading From Image")
        self.image_data_object.max_steps = 7
        #self.thread = QThread()
        self.image_data_object.moveToThread(self.thread)
        self.thread.started.connect(lambda: self.image_data_object.read_message_from_image(
            shuffle_key=self.shuffle_key, threshold=self.threshold,
            blob_size=self.blob_size
        ))
        self.image_data_object.finished.connect(self.finish_image_read)
        self.thread.start()
    

    def finish_image_update(self):
        print("finish image update called")
        self.reset_p_bar()
        self.thread.exit()
        #self.thread.deleteLater()
        self.image = self.image_data_object.image

    def finish_image_write(self):
        self.reset_p_bar()
        self.thread.exit()
        #self.thread.deleteLater()
        self.image = self.image_data_object.image

    def finish_image_read(self):
        self.reset_p_bar()
        self.thread.exit()
        #self.thread.deleteLater()
        self.editor.setPlainText(self.image_data_object.message)

        
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