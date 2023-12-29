import torch

from data import CHARS
from model.LPRNet import build_lprnet
import numpy as np
import cv2

def transform(img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))

    return img
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

filename = r"D:\source\LPRNet_Pytorch\data\mytest\lanpai.jpg"
img = cv_imread(filename)
img = cv2.resize(img, (94,24))
img = transform(img)
im = img[np.newaxis, :]
ims = torch.Tensor(im)
lprnet = build_lprnet(lpr_max_len=8, phase=True, class_num=68, dropout_rate=0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lprnet.to(device)
print("Successful to build network!")
lprnet.load_state_dict(torch.load(r"D:\source\LPRNet_Pytorch\weights\Final_LPRNet_model.pth", map_location=torch.device('cpu')))
print("load pretrained model successful!")

prebs = lprnet(ims.to(device))
prebs = prebs.cpu().detach().numpy()
preb_labels = list()
for i in range(prebs.shape[0]):
    preb = prebs[i, :, :]
    preb_label = list()
    for j in range(preb.shape[1]):
        preb_label.append(np.argmax(preb[:, j], axis=0))
    no_repeat_blank_label = list()
    pre_c = preb_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in preb_label:  # dropout repeate label and blank label
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    preb_labels.append(no_repeat_blank_label)

plate = np.array(preb_labels)
a=list()
for i in range(0, plate.shape[1]):
    b = CHARS[plate[0][i]]
    a.append(b)

print(a)
