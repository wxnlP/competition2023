from pathlib import Path
from threading import Thread
from PyQt5.QtCore import Qt,QCoreApplication
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.Qt import QUrl

from omxplayer import OMXPlayer
import atexit
from pathlib import Path
import sys
import time
import serial
import RPi.GPIO as GPIO
import cv2
import numpy as np
import onnxruntime as ort
import random as rd

# LED_white = 12
# LED_yellow = 16
# LED_blue = 18
# LED_red = 22
# 
# GPIO.setmode(GPIO.BOARD)
# GPIO.setwarnings(False)
# GPIO.setup(LED_white,GPIO.OUT)
# GPIO.setup(LED_yellow,GPIO.OUT)
# GPIO.setup(LED_blue,GPIO.OUT)
# GPIO.setup(LED_red,GPIO.OUT)
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
video1 = 16
video2 = 18
GPIO.setup(video1,GPIO.IN)
GPIO.setup(video2,GPIO.OUT)

def get_key(dict, m_num):
    for key,values in dict.items():
        if values == m_num:
            return key
   

      


# yolov5识别
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    x=x.squeeze()
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
 
def _make_grid( nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
 
def cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride):
    
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w/ stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)
 
        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs
 
 
 
def post_process_opencv(outputs,model_h,model_w,img_h,img_w,thred_nms,thred_cond):
    conf = outputs[:,4].tolist()
    c_x = outputs[:,0]/model_w*img_w
    c_y = outputs[:,1]/model_h*img_h
    w  = outputs[:,2]/model_w*img_w
    h  = outputs[:,3]/model_h*img_h
    p_cls = outputs[:,5:]
    if len(p_cls.shape)==1:
        p_cls = np.expand_dims(p_cls,1)
    cls_id = np.argmax(p_cls,axis=1)
 
    p_x1 = np.expand_dims(c_x-w/2,-1)
    p_y1 = np.expand_dims(c_y-h/2,-1)
    p_x2 = np.expand_dims(c_x+w/2,-1)
    p_y2 = np.expand_dims(c_y+h/2,-1)
    areas = np.concatenate((p_x1,p_y1,p_x2,p_y2),axis=-1)
    
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas,conf,thred_cond,thred_nms)
    if len(ids)>0:
        return  np.array(areas)[ids],np.array(conf)[ids],cls_id[ids]
    else:
        return [],[],[]
def infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5):
    # 图像预处理
    img = cv2.resize(img0, (model_w,model_h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
 
    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
 
    # 输出坐标矫正
    outs = cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride)
 
    # 检测框计算
    img_h,img_w,_ = np.shape(img0)
    boxes,confs,ids = post_process_opencv(outs,model_h,model_w,img_h,img_w,thred_nms,thred_cond)
 
    return  boxes,confs,ids
 
 



class Demo:
    def __init__(self):

        """控件初始化"""
        self.ui = QMainWindow()
        self.label = QLabel(self.ui)
        self.map_label = QLabel(self.ui)
        self.map2_label = QLabel(self.ui)

        self.video_button = QPushButton("创铭", self.ui)
        self.map_button = QPushButton("智造", self.ui)
        self.close_button = QPushButton("结束", self.ui)

        # self.textLabel = QLabel(self.ui)
        self.list = QListWidget(self.ui)
        # 创建表格
        rows = 4  # 行
        columns = 2  # 列
        self.table = QTableWidget(rows, columns, self.ui)
        thread_serial = Thread(
            target=self.thread_serial,
            args=(9600,)
            )
        thread_text = Thread(
            target=self.Text_time
        )
        thread_yolov5 = Thread(
            target=self.thread_yolov5)
        
        
        self.pic_control()
        # 属性初始化函数调用
        self.LayoutInit()
        self.WidgetInit()
        self.tabel_items()
#         self.add_list_items()
        self.Signal()
#         thread_text.start()
#         thread_serial.start()
        thread_yolov5.start()
        

    # Layout界面
    def LayoutInit(self):
        central_widget = QWidget(self.ui)
        self.ui.setCentralWidget(central_widget)
        layout_v = QVBoxLayout(central_widget)
        layout_h = QHBoxLayout(central_widget)
        layout_h2 = QHBoxLayout(central_widget)
        layout_h3 = QHBoxLayout(central_widget)

        layout_h.addWidget(self.label)
        layout_h.addWidget(self.video_button)
        layout_h.addWidget(self.map_button)
        layout_h.addWidget(self.close_button)
        layout_h.setStretch(0, 7)
        layout_h.setStretch(1, 1)
        layout_h.setStretch(2, 1)
        layout_h.setStretch(3, 1)
        layout_h.setSpacing(50)

        layout_h2.addWidget(self.map2_label)
        layout_h2.addWidget(self.map_label)
        layout_h2.setStretch(0, 5)
        layout_h2.setStretch(1, 7)
        layout_h2.setSpacing(0)

        layout_h3.addWidget(self.table)
        layout_h3.addWidget(self.list)
        layout_h3.setStretch(0, 4)
        layout_h3.setStretch(1, 10)
        layout_h3.setSpacing(50)

        layout_v.addLayout(layout_h)
        layout_v.addLayout(layout_h2)
        layout_v.addLayout(layout_h3)
        layout_v.setStretch(0, 1)
        layout_v.setStretch(1, 6)
        layout_v.setStretch(2, 3)

        self.ui.setLayout(layout_v)
    


    def thread_yolov5(self):
        # 模型加载
        model_pb_path = "/home/pi/Py_Projects/G_final.onnx"
        so = ort.SessionOptions()
        net = ort.InferenceSession(model_pb_path, so)
        
        # 标签字典
      
        dic_labels={
            0:"0-plastic_bottle",
            1:"0-drink_can",
            2:"0-paper",
            3:"0-carton",
            4:"0-milkCarton",
            5:"1-pericarp",
            6:"1-vegetable_leaf",
            7:"1-radish",
            8:"1-potato",
            9:"1-fruits",
            10:"2-battery",
            11:"2-Expired_drug",
            12:"2-button cell",
            13:"2-thermometer",
            14:"3-tile",
            15:"3-cobblestone",
            16:"3-brick",
            17:"3-paperCup",
            18:"3-tableware",
            19:"3-chopsticks",
            20:"3-butt",
            21:"3-mask"
            }
        recoverable = [0, 1, 2, 3, 4]
        kitchen_garbage = [6, 7, 8, 9, 5]
        harmful = [11, 12, 13, 10]
        others = [15, 16, 17, 18, 19, 20, 21, 14]
        # 模型参数
        model_h = 320
        model_w = 320
        nl = 3
        na = 3
        stride=[8.,16.,32.]
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)
        
        
        video = -1
        cap = cv2.VideoCapture(video)
        ser = serial.Serial("/dev/ttyAMA0", 9600)
        num=0
        id_list=[]
        temp=1
        time_able=1
        enable=0
        start=0
        VIDEO_PATH = Path("/home/pi/Videos/video.mp4")#加粗的文字请自行替换成自己的路径跟文件名
        global player
        player = OMXPlayer(VIDEO_PATH,args=['--loop', '--no-osd'])
        while True:

            success, img0 = cap.read()
            if success:
                t1 = time.time()
                det_boxes, scores, ids = infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid,
                                                           thred_nms=0.4, thred_cond=0.5)

                t2 = time.time()
                if temp==0:
                    player.quit()
#                 else:
#                     if player.is_playing():
#                         print("视频正在播放")
#                     else:
#                         print("视频未在播放")
#                 
                if GPIO.input(video1)==GPIO.LOW:
#                     print(GPIO.input(video1)==0)
                    
                    for box,score,id in zip(det_boxes,scores,ids):
                        print(f"box{box}")
                        
                        x1 = int(box[0][0])
                        y1 = int(box[0][1])
                        x2 = int(box[0][2])
                        y2 = int(box[0][3])
                        area = (x2-x1)*(y2-y1)
#                         print(f"面积：{area}")
                        if area > 100000:
                            break
                        label = '%s:%.2f'%(dic_labels[id.item()],score)
#                         print(f"label:{label}")
                        if label:
                            enable=1
                            #刷新计数
                            start=time.time()
                            temp=0
                        
                        r = rd.randint(0, 255)
                        g = rd.randint(0, 255)
                        b = rd.randint(0, 255)
                        plot_one_box(box.astype(np.int16), img0, color=(r,g,b), label=label, line_thickness=2)
                        id = id.item()
                        id_list.append(id)
                        if len(id_list)>=5:
                            id_dict={}
                            for id in id_list:
                                id_dict[id] = id_dict.get(id, 0)+1
                            max_num = max(id_dict.values())
    #                         if max_num < 3:
    #                             id_list=[]
    #                             break
                            id = get_key(id_dict, max_num)
                        
                            if id in recoverable:
                                ser.write(bytes('1', 'utf-8')) #可回收
                                num+=1
                                self.add_list_items(f"{num}-可回收垃圾-1-okk")
                        
                            if id in kitchen_garbage:
                                ser.write(bytes('3', 'utf-8')) # 厨余垃圾
                                
                                num+=1
                                self.add_list_items(f"{num}-厨余垃圾-1-okk")
                            if id in harmful:
                                ser.write(bytes('2', 'utf-8')) #有害垃圾
                                num+=1
                                self.add_list_items(f"{num}-有害垃圾-1-okk")
                            if id in others:
                                ser.write(bytes('4', 'utf-8'))
                                num+=1
                                self.add_list_items(f"{num}-其他垃圾-1-okk")
                            
                            id_list=[]
                else:
                    id_list=[]
                    start=0
                    time_able=1
                        
                str_FPS = "FPS: %.2f"%(1./(t2-t1))
                cv2.putText(img0,str_FPS,(40,40),cv2.FONT_HERSHEY_COMPLEX,1,(90,10,70),2)                 
                
                show = cv2.resize(img0, (640, 480))  # 把读到的帧的大小重新设置为 640x480
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
                showImage = QImage(show.data, show.shape[1], show.shape[0],
                                                  QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.map_label.setPixmap(QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
                self.map_label.setScaledContents(True)  

                            

    def stop_video(self):
        
        if player.is_playing():
            player.quit()
            print("视频已停止播放")
        self.ui.close()

    def thread_serial(self, boot):
        ser = serial.Serial("/dev/ttyACM0", boot)
        data=['w', 'W', 'Y', 'y', 'R', 'r', 'B', 'b']
        while True:
            
            if not ser.isOpen():
                ser.open()
            count = ser.inWaiting()
            if count > 0:
                recv = ser.read().decode('utf-8',"ignore")
                print(recv)
                if recv == 'w':
                    self.item = QTableWidgetItem("满载")
                    self.table.setItem(0, 1, self.item)
#                     GPIO.output(LED_white, 1)
                    
                if recv == 'W':
                    self.item = QTableWidgetItem("未满载")
                    self.table.setItem(0, 1, self.item)
#                     GPIO.output(LED_white, 0)
                if recv == 'y':
                    self.item = QTableWidgetItem("满载")
                    self.table.setItem(1, 1, self.item)
#                     GPIO.output(LED_yellow, 1)
                    
                if recv == 'Y':
                    self.item = QTableWidgetItem("未满载")
                    self.table.setItem(1, 1, self.item)
#                     GPIO.output(LED_yellow, 0)
                if recv == 'b':
                    self.item = QTableWidgetItem("满载")
                    self.table.setItem(2, 1, self.item)
#                     GPIO.output(LED_blue, 1)
                    
                if recv == 'B':
                    self.item = QTableWidgetItem("未满载")
                    self.table.setItem(2, 1, self.item)
#                     GPIO.output(LED_blue, 0)
                if recv == 'r':
                    self.item = QTableWidgetItem("满载")
                    self.table.setItem(3, 1, self.item)
#                     GPIO.output(LED_red, 1)
                    
                if recv == 'R':
                    self.item = QTableWidgetItem("未满载")
                    self.table.setItem(3, 1, self.item)
#                     GPIO.output(LED_red, 0)
                if recv not in data:
                    continue
#                 print(f"recv:{recv}")
                

    def Text_time(self):
        while True:
            text = "❤❤❤❤❤地球环保卫士❤❤❤❤❤"
            for i in range(135):
                self.label.setText(text)
#                 time.sleep(0.1)
                text = ' ' + text
    
        

    def tabel_items(self):
        font_tabel = self.table.font()
        font_tabel.setPointSize(20)
        self.table.setFont(font_tabel)
        self.item1 = QTableWidgetItem("可回收垃圾")
        self.table.setItem(0, 0, self.item1)
        self.item2 = QTableWidgetItem("厨余垃圾")
        self.table.setItem(1, 0, self.item2)
        self.item3 = QTableWidgetItem("其他垃圾")
        self.table.setItem(2, 0, self.item3)
        self.item4 = QTableWidgetItem("有害垃圾")
        self.table.setItem(3, 0, self.item4)
        self.item5 = QTableWidgetItem("未满载")
        self.item6 = QTableWidgetItem("未满载")
        self.item7 = QTableWidgetItem("未满载")
        self.item8 = QTableWidgetItem("未满载")
        self.table.setItem(0, 1, self.item5)
        self.table.setItem(1, 1, self.item6)
        self.table.setItem(2, 1, self.item7)
        self.table.setItem(3, 1, self.item8)

    def add_list_items(self, label):
        
        print(f"列表数量{self.list.count()}")
        news = QListWidgetItem(label)
        self.list.addItem(news)
        self.list.scrollToItem(news, QListWidget.PositionAtBottom)
#         self.list.setCurrentRow(self.list.count() - 1)


    # 控件初始化
    def WidgetInit(self):
        self.ui.setWindowTitle("ui")
        self.ui.resize(177 * 4, 400)
        self.label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.table.setShowGrid(False)  # 是否显示网格
        self.table.setHorizontalHeaderLabels(["类别", "是否满载"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        font_list = self.list.font()
        font_list.setPointSize(18)
        self.list.setFont(font_list)
        # self.label.setAlignment(Qt.AlignCenter)
        map_widget = QPixmap("/home/pi/Pictures/女神/17.jpg")
        map_widget = map_widget.scaled(177 * 3, 300)
        self.map_label.setPixmap(map_widget)
        self.map_label.setScaledContents(True)

    def pic_control(self):
        pic_list = ["/home/pi/Pictures/工创赛图集/logo.jpg", "/home/pi/Pictures/6.jpg"]
#         for i in pic_list:
            
        map_widget = QPixmap(pic_list[0])
            
        map_widget = map_widget.scaled(177 * 3, 300)
        self.map2_label.setPixmap(map_widget)
        self.map2_label.setScaledContents(True)
        
    # 信号槽函数初始化
    def Signal(self):
        self.video_button.clicked.connect(self.GetMap2)
        self.map_button.clicked.connect(self.GetMap)
        self.close_button.clicked.connect(self.stop_video)
        

    def GetMap2(self):
        video = QFileDialog()
        url = video.getOpenFileUrl()[0].toLocalFile()
        map_widget = QPixmap(url)
        map_widget = map_widget.scaled(177 * 3, 300)
        self.map2_label.setPixmap(map_widget)
        self.map2_label.setScaledContents(True)


    def GetMap(self):
        video = QFileDialog()
        url = video.getOpenFileUrl()[0].toLocalFile()
        map_widget = QPixmap(url)
        map_widget = map_widget.scaled(177 * 3, 300)
        self.map_label.setPixmap(map_widget)
        self.map_label.setScaledContents(True)


if __name__ == "__main__":


        app = QApplication(sys.argv)
        bar = Demo()
        bar.ui.show()
        bar.ui.showFullScreen()
        
        sys.exit(app.exec_())
       

        
        
