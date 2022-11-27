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
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

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


        self.old_p_bar_label = ""
        #self.make_p_bar()

        #self.p_bar_update_thread = threading.Timer(0.01666, self.update_p_bar)
        #self.p_bar_update_thread.start()

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
        embed_action.triggered.connect(self.embedImageWithMsg)
        msgMenu.addAction(embed_action)

        open_img_action = QAction("Open Image", self)
        open_img_action.triggered.connect(self.openImage)
        imageMenu.addAction(open_img_action)

        show_img_action = QAction("Show Image", self)
        show_img_action.triggered.connect(self.showImage)
        imageMenu.addAction(show_img_action)

        decode_msg_action = QAction("Decode Message From Image", self)
        decode_msg_action.triggered.connect(self.decodeMsg)
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

    def make_p_bar(self):
        self.p_bar.setValue(0)
        #self.p_bar.setFormat(self.widget_progress_values["p_bar_label"])
        self.bar_update_active = True
        #p_bar_update_thread = threading.Timer(0.01666, self.update_p_bar)
        #p_bar_update_thread.start()
    
    def remove_p_bar(self):
        self.bar_update_active = False
    
    def finish_removing_p_bar(self):
        self.p_bar_process_total_steps = 2
        self.widget_progress_values["p_bar_label"] = ""
        self.widget_progress_values["value"] = 0
        self.widget_progress_values["step_integer"] = 0
        print("d")
        self.p_bar.setValue(0)
        print("e")
        #self.p_bar.setFormat("")
        print("f")

    def update_p_bar(self):
        try:
            if self.old_p_bar_label != self.widget_progress_values["p_bar_label"]:
                self.p_bar.setFormat(self.widget_progress_values["p_bar_label"])
            #print("step int ->",self.widget_progress_values["step_integer"])
            cap = int(np.maximum(1, self.p_bar_process_total_steps-1))
            self.widget_progress_values["value"] = 100*self.widget_progress_values["step_integer"] / cap
            self.p_bar.setValue(int(self.widget_progress_values["value"]))
            #if self.bar_update_active:
            self.old_p_bar_label = self.widget_progress_values["p_bar_label"]
            p_bar_update_thread = threading.Timer(0.01666, self.update_p_bar)
            p_bar_update_thread.start()
        except:
            e = sys.exc_info()[0]
            self.editor.setPlainText(e)

    def saveImg(self):
        self.save_file_name,_ = QFileDialog.getSaveFileName(self, "Save file", "", "Images (*.png)")
        if not self.save_file_name:
            return
        cv2.imwrite(self.save_file_name, self.image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def embedImageWithMsg(self):
        self.msg = self.editor.toPlainText()
        self.embedImageString(self.msg)

    def embedImageString(self, message):
        self.msg = message
        try:
            self.embed_thread = threading.Thread(target=self.embed_image_thread_target)
            self.embed_thread.start()
        except:
            e = sys.exc_info()[0]
            self.editor.setPlainText(e)

    def decodeMsg(self):
        try:
            self.decode_thread = threading.Thread(target=self.decode_image_thread_target)
            self.decode_thread.start()
        except:
            self.remove_p_bar()
            e = sys.exc_info()[0]
            self.editor.setPlainText("<p>Error: %s</p>" % e)

    def openImage(self):
        self.file_path, filter_type = QFileDialog.getOpenFileName(self, "Open new file", "",
                                                                  "Images (*.png *.jpeg *.jpg *.bmp *.gif)")
        if not self.file_path:
            return
        proc_thread = threading.Thread(target=self.update_image, args=(cv2.imread(self.file_path),), kwargs={'calc_assignments': True})
        proc_thread.start()
        #self.update_image(cv2.imread(self.file_path), True)
        self.noiseMap = None

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
        artifact_map_visual = stego.pool_mask_visual(self.image_size_assignment, is_size_assignment=True)
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

    def embed_image_thread_target(self):
        #self.p_bar_process_total_steps = 21
        #self.widget_progress_values["p_bar_label"] = "Encoding Image"
        #self.make_p_bar()
        self.update_image(stego.image_write_new(self.image, self.msg, shuffle_key=self.shuffle_key, threshold=self.threshold, size_map=self.image_size_assignment,
                                        cover_flag=self.smart_cover, blob_expand_size=self.blob_size, bar_values=self.widget_progress_values))

    def decode_image_thread_target(self):
        #self.p_bar_process_total_steps = 18
        #self.widget_progress_values["p_bar_label"] = "Decoding Image"
        #self.make_p_bar()
        self.msg = stego.image_read_new(self.image, shuffle_key=self.shuffle_key, threshold=self.threshold, blob_expand_size=self.blob_size, 
                                        bar_values=self.widget_progress_values)
        self.editor.setPlainText(self.msg)

        print("done")
        #self.widget_progress_values["p_bar_label"] = "Done"
        #self.widget_progress_values["value"] = 0
        #self.widget_progress_values["step_integer"] = 0
        #self.remove_p_bar()

    def update_image(self, image, calc_assignments=None):
        print("updating image")
        if calc_assignments is None:
            calc_assignments = True
        if calc_assignments:
            self.p_bar_process_total_steps = 7
            self.widget_progress_values["p_bar_label"] = "Generating LSB Groupings"
            self.make_p_bar()
            self.image_size_assignment = stego.image_size_assignment(stego.isolate_bit_image(image, 7), bar_values=self.widget_progress_values)
            self.size_assignment_made = True
            self.remove_p_bar()
        self.image = image
        self.remove_p_bar()
        print("image made")
        
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