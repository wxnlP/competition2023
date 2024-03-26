import cv2
import numpy as np
import onnxruntime as ort
import time
import random as rd
import serial


def get_key(dict, m_num):
    for key,values in dict.items():
        if values == m_num:
            return key
        
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
    color = color or [rd.randint(0, 255) for _ in range(3)]
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
 
 
 
 
if __name__ == "__main__":
 
    # 模型加载
    model_pb_path = "G_final.onnx"
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
    
    # 模型参数
    model_h = 320
    model_w = 320
    nl = 3
    na = 3
    stride=[8.,16.,32.]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)
    
    video = 0
    cap = cv2.VideoCapture(video)
    flag_det = True
    # ser = serial.Serial("/dev/ttyAMA0", 9600)
    num=0
    id_list=[]
    while True:
        success, img0 = cap.read()
        if success:
            
            if flag_det:
                t1 = time.time()
                time.sleep(0.01)
                det_boxes,scores,ids = infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5)
                t2 = time.time()
            
                
                for box,score,id in zip(det_boxes,scores,ids):
                    label = '%s:%.2f'%(dic_labels[id.item()],score)
                    r = rd.randint(0, 255)
                    g = rd.randint(0, 255)
                    b = rd.randint(0, 255)
                    plot_one_box(box.astype(np.int16), img0, color=(r,g,b), label=label, line_thickness=None)
                    id = id.item()
                    id_list.append(id)
                    if len(id_list)>=10:
                        id_dict={}
                        for id in id_list:
                            id_dict[id] = id_dict.get(id, 0)+1
                        max_num = max(id_dict.values())
                        if max_num < 7:
                            id_list=[]
                            break
                        id = get_key(id_dict, max_num)
                        x1 = int(det_boxes[0][0])
                        y1 = int(det_boxes[0][0])
                        print(f"x1:{x1}\ny1:{y1}")
                        # ser.write(str(1).encode())
                        # time.sleep(1)
#                         ser.write(str(1).encode())
#                         time.sleep(0.1)
                        id_list=[]

                str_FPS = "FPS: %.2f"%(1./(t2-t1))
                cv2.putText(img0,str_FPS,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
                
                
                
                cv2.imshow("video",img0)
                
            