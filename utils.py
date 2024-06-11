from torchvision import transforms,models
import pywt
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import cv2
import random
from RTNet import model,Conv
import math
from scipy import linalg as la
import os

def image_trans(img):
    trans = transforms.Compose([
        # transforms.Grayscale(), # 彩色图像转灰度图像num_output_channels默认1
        transforms.ToTensor(),
        # transforms.Resize((256,256))
    ])
    img = trans(img)
    _,h,w = img.size()
    img = img.unsqueeze(0)
    return transforms.Resize((256,256))(img),h,w


def normalize1 (img):
    minv = img.min()
    maxv = img.max()
    return (img-minv)/(maxv-minv)

def toPIL(img):
    img = normalize1(img)
    TOPIL = transforms.ToPILImage()
    img = img.squeeze()
    return TOPIL(img)


def save_PIL(img,path):
    img  = toPIL(img)
    img.save(path)

def dwt2(img):
    img = img.detach().numpy()
    ca,(ch,cv,cd) = pywt.dwt2(img,'db2')
    return ca,(ch,cv,cd)

def idwt2(ca,ch,cv,cd):
    coffe = ca,(ch,cv,cd)
    img = pywt.idwt2(coffe,'db2')
    img = torch.as_tensor(img)
    return img

def calcGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256])
    for r in range(rows):
        for c in range(cols):
            grayHist[int(image[r,c])] += 1

    return grayHist

def calcEntropy(img):
    img= img*255
    if torch.is_tensor(img):
        img = img.detach().numpy()
        img = img.squeeze()
    img = img.astype(np.uint8)
    #hist,_ = np.histogram(img, np.arange(0, 256), normed=True)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist.ravel()/hist.sum()
    logs = np.log2(hist+0.00001)
    entropy = -1 * (hist*logs).sum()
    return entropy  

def part_En(img,m):
    zeroPad2d = nn.ZeroPad2d(m//2)
    img = zeroPad2d(img)
    img = img.squeeze()     #numpy:257,257
    a,b = img.shape
    entropy = np.ones([a,b])
    for i in range(1,a+1):
        for j in range(1,b+1):
            temp = img[i-1:i+2,j-1:j+2] 
            entropy[i-1,j-1] = calcEntropy(temp)
    return normalize1(entropy)

def part_EN_fusion(img1,img2,m):
    img1 = torch.as_tensor(img1)
    img2 = torch.as_tensor(img2)
    img1 = img1.squeeze()
    img2 = img2.squeeze()   #tensor:256,256
    a,b = img1.shape
    img = torch.ones(a,b)
    EN1 = part_En(img1,m)
    EN2 = part_En(img2,m)
    for i in range(a):
        for j in range(b):
            if EN1[i,j] > EN2[i,j]:
                img[i,j] = img1[i,j]
            else:
                img[i,j] = img2[i,j]
    return img  #tensor:256,256

def EN_fusion(img1,img2):
    img1 = torch.as_tensor(img1)    
    img2 = torch.as_tensor(img2)
    img1 = img1.squeeze()    
    img2 = img2.squeeze()    
    EN1 = calcEntropy(img1)
    EN2 = calcEntropy(img2)
    u = EN1/(EN1+EN2)
    return u*img1 + (1-u)*img2

def diff_img(img):
    img = torch.as_tensor(img)  #1，1，256，256
    img = img.squeeze()         #256,256
    img = img * 255
    zeroPad2d = nn.ZeroPad2d(1)
    img = zeroPad2d(img)        #257，257
    x,y = img.size()
    dx = torch.ones(x,y)
    dy = torch.ones(x,y)
    for i in range(x-1):
        for j in range(y-1):
            dx[i,j] = (img[i+1,j] - img[i,j]).abs()
            dy[i,j] = (img[i,j+1] - img[i,j]).abs()
    grad = torch.sqrt((torch.pow(dx,2)+torch.pow(dy,2))/2)
    return grad #[256,256]

#全局梯度均值融合法
def global_grad_fusion(img1,img2):
    a = img1.sum()/(img1.sum() + img2.sum())
    return a*img1+(1-a)*img2

def part_grad(img,m):
    img = img.squeeze()
    a,b = img.shape
    grad = diff_img(img)
    zeroPad2d = nn.ZeroPad2d(m//2)
    grad = zeroPad2d(grad)        #257，257
    par_grad = torch.ones(a,b)
    for i in range(1,a+1):
        for j in range(1,b+1):
            temp = grad[i-1:i+2,j-1:j+2]
            par_grad[i-1,j-1] = temp.mean()
    return normalize1(par_grad)

#局部梯度均值融合
def part_grad_fusion(img1,img2):
    img1 = torch.as_tensor(img1)
    img2 = torch.as_tensor(img2)
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    a,b  = img1.shape
    img  = img1.clone().detach()
    par_grad1 = part_grad(img1,3)
    par_grad2 = part_grad(img2,3)
    for i in range(a):
        for j in range(b):
            # if par_grad1[i,j] == 0 and par_grad2[i,j] == 0:
            #     img[i,j] = 0
            if par_grad1[i,j] > par_grad2[i,j]:
                img[i,j] = img1[i,j]
            else:
                img[i,j] = img2[i,j]
    return img  #tensor:256,256

def DW_img(img):
    b,c,x,y = img.size()
    mean = img.mean()
    DW = (torch.pow(img-mean,2).sum())/(x*y)
    return DW

def cal_x(img):
    EN = calcEntropy(img)
    H  = diff_img(img)
    DW = DW_img(img)
    x  = 0.3*EN+0.5*H+0.2*DW 
    return x

def enchan_mix(img1,img2,img3):
    x1 = cal_x(img1)
    x2 = cal_x(img2)
    x3 = cal_x(img3)
    a1 = x1/(x1+x2+x3)
    a2 = x2/(x1+x2+x3)
    a3 = x3/(x1+x2+x3)
    img = a1*img1+a2*img2+a3*img3
    return normalize1(img)

def Contrast(img):
    img= img*255
    img = img.detach().numpy()
    img = img.squeeze()
    img = img.astype(np.uint8)
    # 图像方差
    std = np.sqrt(np.var(img))
    if std <= 3:
        p = 3.0
    elif std <= 10:
        p = (27 - 2 * std) / 7
    else:
        p = 1.0

    In = img / 255.0
    G = cv2.GaussianBlur(img, (9,9), 0)  

    E = np.power(((G + 0.1) / (img + 0.1)), p)
    S = np.power(In, E)
    S = torch.as_tensor(S)
    S = S.unsqueeze(0).unsqueeze(0)
    return S

def arctan_img(img):
    mean = img.mean()
    lamda = 10 + (1-mean)/mean
    img = (2/math.pi) * torch.arctan(lamda*img)
    return normalize1(img)


def comp_save(ca,ch,cv,cd,k,path):
    ca = torch.as_tensor(ca)   
    ch = torch.as_tensor(ch)   
    cv = torch.as_tensor(cv)   
    cd = torch.as_tensor(cd)   
    ca = toPIL(ca)
    ch = toPIL(ch)
    cv = toPIL(cv)
    cd = toPIL(cd)
    if k != None:
        ca_path = path + '/ca' + str(k) + '.PNG'
        ch_path = path + '/ch' + str(k) + '.PNG'
        cv_path = path + '/cv' + str(k) + '.PNG'
        cd_path = path + '/cd' + str(k) + '.PNG'
    else:
        ca_path = path + '/ca' + '.PNG'
        ch_path = path + '/ch' + '.PNG'
        cv_path = path + '/cv' + '.PNG'
        cd_path = path + '/cd' + '.PNG'
    ca.save(ca_path)
    ch.save(ch_path)
    cv.save(cv_path)
    cd.save(cd_path)


def bright_weight(img):
    w_b = torch.exp(-(torch.pow(img-0.5,2))/(2*0.25*0.25))
    return w_b

def con_calcula(img):    
    zeroPad2d = nn.ZeroPad2d(1)
    img_ext = zeroPad2d(img)        #a+1,b+1
    m,n = img_ext.shape
    b = 0
    for i in range(1,m-1):
        for j in range (1,n-1):
            # b += img_ext *c + img_ext[i,j]
            b += ((img_ext[i,j]-img_ext[i,j+1])**2 + (img_ext[i,j]-img_ext[i,j-1])**2 + 
                    (img_ext[i,j]-img_ext[i+1,j])**2 + (img_ext[i,j]-img_ext[i-1,j])**2)
    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4)
    return cg

def con_weight(img):
    img = img * 255
    if torch.is_tensor(img):
        img = img.squeeze()
    else:
        img = torch.as_tensor(img)
    zeroPad2d = nn.ZeroPad2d(1)
    img = zeroPad2d(img)        #a+2,b+2
    m,n = img.shape
    w_c = torch.ones(m-2,n-2)
    for i in range(1,m-1):
        for j in range(1,n-1):
            w_c[i-1,j-1] = ((img[i,j]-img[i,j+1])**2 + (img[i,j]-img[i,j-1])**2 + (img[i,j]-img[i+1,j])**2 + (img[i,j]-img[i-1,j])**2)
    w_c = w_c.unsqueeze(0).unsqueeze(0)
    return normalize1(w_c)


def weight(img):
    w_b = visible(img)
    w_c = con_weight(img)
    w_g = part_grad(img,3)
    # w = w_b * w_c * w_b
    w = w_b * w_c * w_g
    return w


def fusion_enhance(img1,img2,img3):
    if torch.is_tensor(img1):
        tag = 1
    else:
        tag = 0
    w1 = weight(img1)
    w2 = weight(img2)
    w3 = weight(img3)
    n_w1 = w1/(w1+w2+w3)
    n_w2 = w2/(w1+w2+w3)
    n_w3 = w3/(w1+w2+w3)
    n_w1 = torch.where(
        torch.isnan(n_w1), 
        torch.full_like(n_w1, 1/3), 
        n_w1)
    n_w2 = torch.where(
        torch.isnan(n_w2), 
        torch.full_like(n_w2, 1/3), 
        n_w2)
    n_w3 = torch.where(
        torch.isnan(n_w3), 
        torch.full_like(n_w3, 1/3), 
        n_w3)
    if tag == 0:
        n_w1 = n_w1.squeeze()
        n_w2 = n_w2.squeeze()
        n_w3 = n_w3.squeeze()
        n_w1 = n_w1.detach().numpy()
        n_w2 = n_w2.detach().numpy()
        n_w3 = n_w3.detach().numpy()
    img = n_w1 * img1 + n_w2 * img2 + n_w3 * img3 
    return normalize1(img)


def gamma_his(img):
    img = img*255
    img = torch.floor(img)
    img = img.detach().numpy()
    img = img.squeeze()
    img = img.astype(np.uint8)

     #获取直方图
    grayHist = calcGrayHist(img)

    #计算原始直方图均值
    Tc = grayHist.mean()

    #计算裁剪直方图
    grayHist[grayHist>=Tc] = Tc

    #归一化，计算概率密度函数pdf
    pi = grayHist/grayHist.sum()

    #计算累加分布函数
    ci = np.zeros([256],np.float32)
    for k in range(256):
        if k==0:
            ci[k] = pi[k]
        else:
            ci[k] = pi[k-1]+pi[k]

    #计算加权直方图分布函数
    pwi = pi.max()*np.power((pi-pi.min())/(pi.max()-pi.min()),ci)
    # alpha = 0.5
    # pwi = pi.max()*np.power((pi-pi.min())/(pi.max()-pi.min()),alpha)
    #计算最大强度级别
    Imax = img.max()

    #计算CDF
    sum = 0.0
    for i in range(Imax):
        sum += pwi[i]
    cwi_a = (pi/sum)
    cwi = 0.0
    for i in range(Imax):
        cwi += cwi_a[i] 
    #计算gamma
    gamma = 1-cwi

    #gamma校正
    img = np.power(img/img.max(),gamma)

    #转化为tennsor类型
    img = torch.as_tensor(img)
    img = img.unsqueeze(0).unsqueeze(0)
    return img

def equalhist(img):
    img = img*255
    img = torch.floor(img)
    img = img.detach().numpy()
    img = img.squeeze()
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = torch.tensor(img)
    img = img.unsqueeze(0).unsqueeze(0)
    return normalize1(img)

def Electric(img):
    a,b = img.shape #256，256
    zeroPad2d = nn.ZeroPad2d(1)
    img = zeroPad2d(img)
    E =  torch.ones(a,b)
    w = 1/16*torch.tensor([[1,2,1],[2,4,2],[1,2,1]])
    for i in range(1,a):
        for j in range(1,b):
            temp = img[i-1:i+2,j-1:j+2]
            E[i-1,j-1] = (torch.abs(temp*w)).sum()
    return E   #tenspr:256,256

def Elefusion(img1,img2):
    img1 = torch.as_tensor(img1)
    img2 = torch.as_tensor(img2)
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    a,b = img1.shape    #tensor:256,256
    e = 0.7
    E1 = Electric(img1)
    E2 = Electric(img2)
    img =  torch.ones([a,b])
    C = E1 - E2
    C[C>0] = 1
    C[C<0] = 0
    img = C*img1 + (1-C)*img2
    # for i in range(a):
    #     for j in range(b):
    #         if E1[i,j] > E2[i,j]:
    #             img[i,j] = img1[i,j]
    #         else:
    #             img[i,j] = img2[i,j]
    img = img.detach().numpy()
    return img  #numpy:256,256

def Pyramid(A,B,k):
    # 构建A和IB高斯金字塔
    A = A.detach().numpy()
    B = B.detach().numpy()
    A = A.squeeze()
    B = B.squeeze()
    G = A.copy()
    G1 = B.copy()
    gpA = [G]
    gpB = [G1]
    '''
    进行下采样
    0:256,256
    1:128,128
    2:64,64
    3:32,32
    4:16:16
    5:8,8
    6:4,4
    7:2,2
    0代表最底层
    '''
    for i in range(k):  
        G = cv2.pyrDown(G)
        # print(G.shape)
        G1 = cv2.pyrDown(G1)
        # print(G.shape)
        gpA.append(G)
        gpB.append(G1)

    # 构建A的拉普拉斯金字塔
    LA = [gpA[k-1]]
    for i in range(k-1, 0, -1):
        LA.append(cv2.subtract(gpA[i-1],cv2.pyrUp(gpA[i])))
    # 构建B的拉普拉斯金字塔
    LB = [gpB[k-1]]
    for i in range(k-1, 0, -1):
        LB.append(cv2.subtract(gpB[i - 1], cv2.pyrUp(gpB[i])))

    # 金字塔融合
    Merge = []
    for i in range(k):
        if i == k-1:
            temp = part_grad_fusion(LA[i],LB[i])
            temp = temp.detach().numpy()
            Merge.append(temp) 
        else:
            temp = Elefusion(LA[i],LB[i])
            Merge.append(temp)
           
    result = Merge[0]
    for i in range(1,k):
        result = cv2.pyrUp(result)
        result = cv2.add(result, Merge[i])
    result = torch.as_tensor(result)
    result = result.unsqueeze(0).unsqueeze(0)
    return  result


def visible(img):
    img = torch.as_tensor(img)
    img = img.squeeze() #256，256
    a,b = img.shape
    p = 3
    q = 3
    gamma = 0.6
    zeroPad2d = nn.ZeroPad2d(1) #257，257
    img = zeroPad2d(img)
    IV =  torch.ones(a,b)
    for i in range(1,a+1):
        for j in range(1,b+1):
            temp = img[i-1:i+2,j-1:j+2]
            c = temp.mean().abs()
            # if c == 0:
            #     IV[i-1,j-1] = temp[i,j]
            # else:   
            IV[i-1,j-1] = ((pow(1/c,gamma)*torch.abs(temp-c)/c).sum())/(p*q)
    return normalize1(IV)   #tensor:256,256

def fusion_v(img1,img2):
    img1 = torch.as_tensor(img1)
    img2 = torch.as_tensor(img2)
    img1 = img1.squeeze()   #256，256
    img2 = img2.squeeze()   #256，256
    a,b = img1.shape
    IV1 = visible(img1)     #256,256
    IV2 = visible(img2)     #256,256
    img =  torch.ones(a,b)
    for i in range(a):
        for j in range(b):
            if IV1[i,j] > IV2[i,j]:
                img[i,j] = img1[i,j]
            else:
                img[i,j] = img2[i,j]
    return img  #tensor:256,256

def Parm_infusion(RR,VR,k):
    save_path = 'result/composent/'
    save_path_r = save_path + str(k+1) + '/R'
    save_path_v = save_path + str(k+1) + '/V'
    save_path_f = save_path + str(k+1) + '/F'
    ca1,(ch1,cv1,cd1) = dwt2(RR)
    ca2,(ch2,cv2,cd2) = dwt2(VR)

    comp_save(ca1,ch1,cv1,cd1,1,save_path_r)
    comp_save(ca2,ch2,cv2,cd2,2,save_path_v)

    ca1 = torch.as_tensor(ca1)
    ca2 = torch.as_tensor(ca2)
    

    ca = EN_fusion(ca1,ca2)
    ch = EN_fusion(ch1,ch2)
    cv = EN_fusion(cv1,cv2)
    cd = EN_fusion(cd1,cd2)


    comp_save(ca,ch,cv,cd,None,save_path_f)

    ca = ca.detach().numpy()
    ch = ch.detach().numpy()
    cv = cv.detach().numpy()
    cd = cd.detach().numpy()

    return idwt2(ca,ch,cv,cd)

def DWT_ELE(RR,VR,k):
    save_path = 'result/composent/'
    save_path_r = save_path + str(k+1) + '/R'
    save_path_v = save_path + str(k+1) + '/V'
    save_path_f = save_path + str(k+1) + '/F'

    ca1,(ch1,cv1,cd1) = dwt2(RR)
    ca2,(ch2,cv2,cd2) = dwt2(VR)


    ca1 = torch.as_tensor(ca1)
    ca2 = torch.as_tensor(ca2)
    
    ca = EN_fusion(ca1,ca2)
    ch = Elefusion(ch1,ch2)
    cv = Elefusion(cv1,cv2)
    cd = Elefusion(cd1,cd2)


    ca = ca.detach().numpy()

    return idwt2(ca,ch,cv,cd)

def gray_gamma(img,gamma):
    img = torch.pow(img,gamma)
    return normalize1(img)


def CLAHE(img):
    img = img*255
    img = torch.floor(img)
    img = img.detach().numpy()
    img = img.squeeze()
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE()
  
    cl2 = clahe.getClipLimit()
    clahe.setClipLimit(1)
    cl1 = clahe.getClipLimit()
    clahe.setTilesGridSize((8,8))

    imgEquA = clahe.apply(img)
    imgEquA = torch.as_tensor(imgEquA).unsqueeze(0).unsqueeze(0)
    return normalize1(imgEquA)


def fusion_img(RR,RI,VR,VI,k):
    #高频融合
    R = DWT_ELE(RR,VR,k)
    R = R.unsqueeze(0).unsqueeze(0)
    normalize1(R)

    #金字塔
    
    I = Pyramid(RI,VI,7)
    normalize1(I)
    return I,R


