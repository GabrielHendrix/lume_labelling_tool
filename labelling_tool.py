import os
import cv2
import sys
import math
import glob
import torch
import pickle
import numpy as np
from natsort import natsorted
from sam2.build_sam import build_sam2_video_predictor
from image_and_video_generator_from_log import ImagesFromLog

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QHBoxLayout,
    QCheckBox,
    QGridLayout,
    QDialog,
    QLineEdit,
    QFileDialog,
    QMessageBox,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


np.random.seed(3)


def get_initial_paths():
    # Janela para selecionar o diretório das imagens
    img_path = QFileDialog.getExistingDirectory(None, "Selecione o diretório das imagens")
    if not img_path:
        QMessageBox.critical(None, "Erro", "Diretório de imagens não selecionado.")
        sys.exit()

    # Janela para selecionar o diretório de saída
    outdir = QFileDialog.getExistingDirectory(None, "Selecione o diretório de saída")
    if not outdir:
        QMessageBox.critical(None, "Erro", "Diretório de saída não selecionado.")
        sys.exit()

    return img_path, outdir


# Converter todos os ndarrays para listas
def ndarray_converter(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Tipo {type(obj)} não é serializável para JSON")


def dist2p(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        

def show_masks(image, masks, point_coords=None, box_coords=None, input_labels=None, borders=True):

    overlay_image = image.copy()
    h, w, c = image.shape

    combined_mask = np.zeros((h, w), dtype=np.uint8)  # Acumulador da união de máscaras
    for i, mask in enumerate(masks):
        # Se for tensor do PyTorch, converte para NumPy
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # Remove dimensões extras e binariza
        mask = np.squeeze(mask)  # garante shape (H, W)
        mask = (mask > 0).astype(np.uint8) * 255

        # Cor da máscara
        mask_color = (0, 255, 0)  # Verde

        # Aplica cor somente nas regiões da máscara
        # overlay_image[mask > 0] = mask_color

        # Soma a máscara no acumulador
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Após o loop: encontra o contorno da união
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) > 0):
        # Obtém o bounding box que cobre todos os contornos
        x, y, w, h = cv2.boundingRect(np.vstack(contours))  # Junta todos os contornos e calcula o bbox
        bbox = [x, y, w, h]
    else:
        bbox = None
        contours = None
        combined_mask = None

    return overlay_image, contours, combined_mask, bbox


def cv_image_to_pixmap(cv_img):
    """Converte imagem OpenCV (BGR) para QPixmap"""
    height, width, channel = cv_img.shape
    bytes_per_line = channel * width
    # Converte BGR -> RGB
    qimg = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(qimg)



class ClickableLabel(QLabel):
    # clicked = pyqtSignal()  # sem coordenadas, só ação
    clicked = pyqtSignal(int, int, str)

    def mousePressEvent(self, event):
        x = int(event.position().x())
        y = int(event.position().y())
        
        if event.button() == Qt.MouseButton.LeftButton:      
            self.clicked.emit(x,y,"left")

        if event.button() == Qt.MouseButton.RightButton:
            self.clicked.emit(x,y,"right")
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.clicked.emit(x, y, "middle")
            

class ClassSelectionDialog(QDialog):
    def __init__(self, class_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Selecionar Classe")
        self.selected_class = None
        self.setModal(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Escolha uma classe:"))

        for cls in class_list:
            btn = QPushButton(cls)
            btn.clicked.connect(lambda checked, c=cls: self.select_class(c))
            layout.addWidget(btn)

        self.setLayout(layout)

    def select_class(self, selected):
        self.selected_class = selected
        self.accept()


class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuração Inicial")
        self.setModal(True)
        self.is_checked = False
        self.img_path = ""
        self.outdir = "/home/lume/Desktop/"
        self.camera_id = 1
        self.layout = QVBoxLayout()

        # Caminho da imagem
        self.img_path_edit = QLineEdit()
        self.img_path_btn = QPushButton("Selecionar Pasta de Imagens")
        self.img_path_btn.clicked.connect(self.select_img_path)

        # Caminho de saída
        self.outdir_edit = QLineEdit()
        self.outdir_edit.setText(self.outdir)
        self.outdir_btn = QPushButton("Selecionar Pasta de Saída")
        self.outdir_btn.clicked.connect(self.select_outdir) 

        # Botão iniciar
        start_btn = QPushButton("Iniciar")
        start_btn.clicked.connect(self.accept)
        self.layout.addWidget(start_btn)

        self.img_path_title = QLabel("Caminho de Entrada:")
        self.layout.addWidget(self.img_path_title)
        self.layout.addWidget(self.img_path_edit)
        self.layout.addWidget(self.img_path_btn)

        self.outdir_title = QLabel("Caminho de Saída:")
        self.layout.addWidget(self.outdir_title)
        self.layout.addWidget(self.outdir_edit)
        self.layout.addWidget(self.outdir_btn)
        self.outdir_title.setVisible(False)
        self.outdir_edit.setVisible(False)
        self.outdir_btn.setVisible(False)

        # Numero da camera
        self.camera_box_title = QLabel("Camera ID:")
        self.camera_box = QComboBox()
        self.layout.addWidget(self.camera_box_title)
        self.layout.addWidget(self.camera_box)
        self.camera_box.addItems(['1','2','3','4','5'])
        self.camera_box.setCurrentText(str(self.camera_id))
        self.camera_box.currentTextChanged.connect(self.update_camera_id)
        self.camera_box_title.setVisible(False)
        self.camera_box.setVisible(False)

        # Checkbox para controlar a visibilidade do campo
        self.use_outdir_checkbox = QCheckBox("Converter imagens a partir do log")
        self.use_outdir_checkbox.stateChanged.connect(self.toggle_outdir_edit)
        self.layout.addWidget(self.use_outdir_checkbox)

        self.setLayout(self.layout)
        self.adjustSize()               # Ajusta o tamanho da janela ao conteúdo
        self.setFixedSize(self.size())  # Define o tamanho como fixo

    def update_camera_id(self, new_id):
        if new_id.isdigit():
            self.camera_id = int(new_id)

    def select_img_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Imagens")
        if folder:
            self.img_path_edit.setText(folder)

    def select_outdir(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Saída")
        if folder:
            self.outdir_edit.setText(folder)

    def get_paths(self):
        return self.img_path_edit.text().strip(), self.outdir_edit.text().strip(), self.is_checked, int(self.camera_id)


    def toggle_outdir_edit(self, state):
        self.is_checked = (state == Qt.CheckState.Checked.value)
        if (self.is_checked):
            # self.layout.addWidget(self.outdir_title)
            # self.layout.addWidget(self.outdir_edit)
            # self.layout.addWidget(self.outdir_btn)
            self.outdir_title.setVisible(self.is_checked)
            self.outdir_edit.setVisible(self.is_checked)
            self.outdir_btn.setVisible(self.is_checked)
            self.camera_box_title.setVisible(self.is_checked)
            self.camera_box.setVisible(self.is_checked)
            # self.outdir_edit.show()
            
        else:
            self.outdir_title.setVisible(self.is_checked)
            self.outdir_edit.setVisible(self.is_checked)
            self.outdir_btn.setVisible(self.is_checked)
            self.camera_box_title.setVisible(self.is_checked)
            self.camera_box.setVisible(self.is_checked)
        self.setLayout(self.layout)
        self.adjustSize()               # Ajusta o tamanho da janela ao conteúdo
        self.setFixedSize(self.size())  # Define o tamanho como fixo


class MainWindow(QMainWindow):
    def __init__(self, img_path):
        super().__init__()

        self.setWindowTitle("Visualizador de Imagens")
        self.img_path = img_path
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(self.model_cfg, "../sam2/checkpoints/sam2.1_hiera_large.pt", device=device)    
        self.obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        self.idx = 0
        self.input_points = []
        self.input_labels = []
        self.predict = False
        self.show_all = False
        self.objs_ids = {'1':None}
        self.data = {}
        self.all_classes = ["pedestrian", "car", "truck", "bus"]  # Exemplo
        self.interval = 100

        if os.path.isfile(self.img_path):
            if self.img_path.endswith('txt'):
                with open(self.img_path, 'r') as f:
                    self.filenames = f.read().splitlines()
            else:
                self.filenames = [self.img_path]
        else:
            self.filenames = glob.glob(os.path.join(self.img_path, '**/*'), recursive=True)
        
        # self.filenames.sort()
        self.filenames = natsorted(self.filenames)

        if os.path.isfile((os.path.join(self.img_path, "data.pkl"))):
            self.load_data()
        else:
            self.objs_ids = {'1':None}
            self.data.update({'objs_ids': self.objs_ids})        
        # self.filenames = self.filenames[self.idx:self.idx+self.interval]
        self.inference_state = self.predictor.init_state(video_path=self.img_path, frame_names=self.filenames[self.idx:(self.idx + self.interval)], index=self.idx)

        # Layouts principais
        main_layout = QVBoxLayout()
        up_layout = QHBoxLayout()  # Combobox
        layout = QGridLayout()
        layout.setHorizontalSpacing(100)

        # Imagem
        self.image_label = ClickableLabel()
        self.image_label.clicked.connect(self.mouse_click)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.image_label.setScaledContents(True)
        up_layout.addWidget(self.image_label)

        self.predict_mode_checkbox = QCheckBox()
        self.predict_mode_checkbox.setText("Inference (p)")
        layout.addWidget(self.predict_mode_checkbox, 1, 0)
        self.show_all_checkbox = QCheckBox()
        self.show_all_checkbox.setText("Show all objs (h)")
        layout.addWidget(self.show_all_checkbox, 2, 0)

        # Dentro do seu __init__ ou método de setup da GUI
        self.predict_mode_checkbox.stateChanged.connect(self.on_predict_mode_checkbox_change)
        self.show_all_checkbox.stateChanged.connect(self.on_show_all_checkbox_change)
        
        # ComboBox e título
        self.combo_title = QLabel("Object ID:")
        self.combo_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.combo_box = QComboBox()
        self.combo_box.currentTextChanged.connect(self.on_id_changed)
        self.new_id = QPushButton("New Object")
        self.new_id.clicked.connect(self.increment_object_id)

        layout.addWidget(self.combo_title, 0, 2)
        layout.addWidget(self.combo_box, 1, 2)
        layout.addWidget(self.new_id, 2, 2) 


        # ComboBox e título
        self.frame_index_edit_title = QLabel("Frame Index:")
        self.frame_index_edit_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.frame_index_edit = QLineEdit()

        confirm_button = QPushButton("OK")
        confirm_button.clicked.connect(self.on_confirm)
        layout.addWidget(self.frame_index_edit_title, 0, 1)
        layout.addWidget(self.frame_index_edit, 1, 1)
        layout.addWidget(confirm_button, 2, 1)
        
        # Junte os layouts
        main_layout.addLayout(up_layout)
        main_layout.addLayout(layout)

        # Widget central
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)


        # Mostrar primeira imagem
        self.update_image([self.obj_id])


    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_P:
            self.on_predict_mode_checkbox_change(self.predict_mode_checkbox.isChecked())
        if event.key() == Qt.Key.Key_H:
            self.on_show_all_checkbox_change()
        # elif event.key() == Qt.Key_Left:


    def on_show_all_checkbox_change(self, state):
        if state == Qt.CheckState.Checked.value:
            self.show_all = True
            self.update_image(self.objs_ids.keys())
        else:
            self.show_all = False
            self.update_image([self.obj_id])
            

    def on_predict_mode_checkbox_change(self, state):
        if state == Qt.CheckState.Checked.value:
            self.predict = True
        else:
            self.predict = False

    def on_confirm(self):
        text = self.frame_index_edit.text().strip()
        if text:
            new_idx = int(text)
            cents_new_idx    = (abs(new_idx) // 100) % 10
            cents_idx        = (abs(self.idx) // 100) % 10
            if cents_new_idx == cents_idx:
                self.idx = new_idx
                self.update_image([self.obj_id])
            else:
                if cents_new_idx == 0:
                    self.idx = 99
                    self.show_previous_image()
                else:
                    self.idx = (cents_new_idx * 100)
                    self.show_next_image()
                self.idx = new_idx
                self.update_image([self.obj_id])


    def increment_object_id(self):
        dialog = ClassSelectionDialog(self.all_classes, self)
        if dialog.exec():
            selected_class = dialog.selected_class
            if selected_class:
                self.obj_id = len(self.objs_ids) + 1
                self.objs_ids.update({str(self.obj_id):selected_class})
                self.input_points.clear()
                self.input_labels.clear()
                self.predictor.reset_state(self.inference_state)


    def on_id_changed(self, new_id):
        if new_id.isdigit():
            self.obj_id = int(new_id)
            self.input_points.clear()
            self.input_labels.clear()
            self.predictor.reset_state(self.inference_state)
            self.update_image([self.obj_id])


    def update_combobox_ids(self):
        key = str(self.filenames[self.idx].split("/")[-1])
        self.combo_box.blockSignals(True)  # Evita trigger do evento ao limpar/adicionar
        self.combo_box.clear()
        self.combo_box.addItems(self.objs_ids.keys())
        self.combo_box.setCurrentText(str(self.obj_id))
        self.combo_box.blockSignals(False)


    def mouse_click(self, x, y, button_side):
        if (button_side == "middle"):
            self.predict_next_image()
        else:
            if (button_side == "left"):
                self.input_points.append([x,y])
                self.input_labels.append(1)
            elif (button_side == "right"):
                min_dist = 999
                min_index = -1  
                
                for index, point in enumerate(self.input_points):
                    dist = (dist2p((point[0], point[1]), (x,y)))
                    if (dist < min_dist):
                        min_dist = dist
                        min_index = index
                if (min_index != -1):
                    self.input_points.pop(min_index)
                    self.input_labels.pop(min_index)
            
            cv_img = cv2.imread(self.filenames[self.idx])
            
            if (len(self.input_points) > 0):
                local_input_points = np.array(self.input_points)
                local_input_labels = np.array(self.input_labels)

                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.idx % self.interval, #len(self.inference_state["images"]),
                    obj_id=self.obj_id,
                    points=local_input_points,
                    labels=local_input_labels,
                )
                cv_img, contours, masks, bbox = show_masks(cv_img, out_mask_logits, point_coords=local_input_points, input_labels=local_input_labels, borders=True)
                self.insert_info(contours, masks, bbox)
            else:
                self.predictor.reset_state(self.inference_state)
                self.delete_id()

        self.update_image([self.obj_id])


    def predict_next_image(self, reverse=False):
        #  run propagation throughout the video and collect the results in a dict
        self.input_points.clear()
        self.input_labels.clear()
        video_segments = {}  # video_segments contains the per-frame segmentation results

        if (len(self.inference_state['obj_ids']) > 0):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=self.idx % self.interval, max_frame_num_to_track=0, reverse=reverse):
                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                self.idx = self.idx - self.idx % self.interval
                self.idx += out_frame_idx
                cv_img = cv2.imread(self.filenames[self.idx])
                cv_img, contours, masks, bbox = show_masks(cv_img, out_mask_logits, point_coords=self.input_points, input_labels=self.input_labels, borders=True)
                self.insert_info(contours, masks, bbox)
        # else:
            # ADICIONAR JANELA DE ERRO AQUI

    def wheelEvent(self, event):
        angle = event.angleDelta().y()        
        if (angle > 0):
            if self.idx < len(self.filenames) - 1:
                self.idx += 1  # avançar
            self.show_next_image()
        elif (angle < 0):
            if self.idx > 0:
                self.idx -= 1  # voltar
            self.show_previous_image()


    def insert_info(self, contours, masks, bbox):
        selected_class = None
        key = str(self.filenames[self.idx].split("/")[-1])

        if (key not in self.data):
            self.data.update({key: {}})        

        if (self.objs_ids[str(self.obj_id)] is None):
            dialog = ClassSelectionDialog(self.all_classes, self)
            if dialog.exec():
                selected_class = dialog.selected_class 
                self.objs_ids[str(self.obj_id)] = selected_class
        else:
            selected_class = self.objs_ids[str(self.obj_id)]

        if ((key in self.data) and (selected_class is not None)):
            if (str(self.obj_id) in self.data[key]): 
                self.data[key][str(self.obj_id)]["class"] = selected_class
                self.data[key][str(self.obj_id)]["masks"] = masks
                self.data[key][str(self.obj_id)]["contours"] = contours
                self.data[key][str(self.obj_id)]["bbox"] = bbox
        
        
        if ((str(self.obj_id) not in self.data[key]) and (selected_class is not None)): 
            self.data[key].update({str(self.obj_id): {
                                "class": selected_class,
                                "masks": masks,
                                "contours": contours,
                                "bbox": bbox
                            }})


    def delete_id(self):
        key = str(self.filenames[self.idx].split("/")[-1])
        if (key in self.data):
            if (str(self.obj_id) in self.data[key]):
                del self.data[key][str(self.obj_id)]


    def update_image(self, objs_ids):
        if 0 <= self.idx < len(self.filenames):
            self.frame_index_edit.setText(str(self.idx))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            cv_img = cv2.imread(self.filenames[self.idx])
            key = str(self.filenames[self.idx].split("/")[-1])

            if (str(key) in self.data):
                for obj_id in objs_ids:
                    if (str(obj_id) in self.data[key]): 
                        if ("contours" in self.data[key][str(obj_id)]):
                            if (self.data[key][str(obj_id)]["contours"] is not None):
                                x, y, w, h = self.data[key][str(obj_id)]["bbox"]
                                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Caixa vermelha
                                cv2.drawContours(cv_img, self.data[key][str(obj_id)]["contours"], -1, (0, 255, 255), 2)
                                    
                                # Desenha pontos se existirem
                                if self.input_points is not None:
                                    assert self.input_labels is not None
                                    for point, label in zip(self.input_points, self.input_labels):
                                        # Verifique se as coordenadas do ponto são válidas
                                        point = tuple(map(int, point))  # Garante que as coordenadas sejam inteiras

                                        # Desenhe o ponto e o rótulo
                                        cv2.circle(cv_img, point, radius=2, color=(255, 0, 0), thickness=-1)  # Ponto vermelho
            
            if cv_img is not None:
                pixmap = cv_image_to_pixmap(cv_img)
                self.image_label.setPixmap(pixmap)
                self.save_data()
            else:
                print(f"Erro ao carregar imagem: {self.filenames[self.idx]}")
        self.update_combobox_ids()
    
    def centroid_calc(self, idx):
        key = str(self.filenames[idx].split("/")[-1])
        cx = cy = -1
        if (key in self.data):
            if (str(self.obj_id) in self.data[key]):
                if "contours" in self.data[key][str(self.obj_id)]:
                    contours = self.data[key][str(self.obj_id)]["contours"]
                    for contour in contours:
                        # Calcula os momentos
                        moments = cv2.moments(contour)

                        # Verifica se a área (M["m00"]) não é zero para evitar divisão por zero
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])  # Coordenada x do centro de massa
                            cy = int(moments["m01"] / moments["m00"])  # Coordenada y do centro de massa
        return cx, cy


    def show_previous_image(self):
        # if self.idx > 0:
        #     self.idx -= 1  # voltar
        if ((self.idx % self.interval) == 99 and (self.idx != 0)):
            self.predictor.reset_state(self.inference_state)
            self.inference_state = self.predictor.init_state(video_path=self.img_path, frame_names=self.filenames[(self.idx - (self.idx % self.interval)):self.idx+1], index=self.idx-self.interval)
            if (self.inference_state["obj_id_to_idx"].get(self.obj_id, None) is None) and (self.idx + 1 >= 0):
                cx, cy = self.centroid_calc(self.idx + 1)
                if cx != -1:
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=self.idx % self.interval,
                        obj_id=self.obj_id,
                        points=[[cx, cy]],
                        labels=[1],
                    )
        if (self.predict):
            self.predict_next_image(True)
        if (self.show_all):
            self.update_image(self.objs_ids.keys())
        else:
            self.update_image([self.obj_id])


    def show_next_image(self):
        # if self.idx < len(self.filenames) - 1:
        #     self.idx += 1  # avançar
        if ((self.idx % self.interval) == 0):
            self.predictor.reset_state(self.inference_state)
            if (self.idx + self.interval) < len(self.filenames) - 1:
                self.inference_state = self.predictor.init_state(video_path=self.img_path, frame_names=self.filenames[self.idx:(self.idx + self.interval)], index=self.idx)
            else:
                self.inference_state = self.predictor.init_state(video_path=self.img_path, frame_names=self.filenames[self.idx:-1], index=self.idx)
            if (self.inference_state["obj_id_to_idx"].get(self.obj_id, None) is None) and (self.idx - 1 >= 0):
                cx, cy = self.centroid_calc(self.idx - 1)
                if cx != -1:
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=self.idx % self.interval,
                        obj_id=self.obj_id,
                        points=[[cx, cy]],
                        labels=[1],
                    )
        if (self.predict):
            self.predict_next_image()

        if (self.show_all):
            self.update_image(self.objs_ids.keys())
        else:
            self.update_image([self.obj_id])


    def save_data(self):
        with open(os.path.join(self.img_path, "data.pkl"), "wb") as f:
            pickle.dump(self.data, f)


    def load_data(self):
        with open(os.path.join(self.img_path, "data.pkl"), "rb") as f:
            self.data = pickle.load(f)
        # for i, file in enumerate(self.filenames):
        key = str(self.filenames[self.idx].split("/")[-1])
        if 'objs_ids' in self.data:
            self.objs_ids = self.data['objs_ids']
    
        # # Salvando em um arquivo JSON
        # with open(os.path.join(self.img_path, "dados.json"), "w", encoding="utf-8") as f:
        #     json.dump(self.data, f, default=ndarray_converter, ensure_ascii=False, indent=4)


app = QApplication(sys.argv)

# Abre o diálogo inicial
dialog = InputDialog()
if dialog.exec():
    img_path, outdir, from_log, camera_id = dialog.get_paths()
    if from_log:
        outdir = os.path.join(outdir, img_path.split("/")[-1])
        image_conversor = ImagesFromLog(img_path, outdir, camera_id)
        image_conversor.process()
        img_path = outdir
    if img_path:
        window = MainWindow(img_path)
        window.show()
        sys.exit(app.exec())
    else:
        print("Caminhos não definidos.")
else:
    print("Execução cancelada.")

