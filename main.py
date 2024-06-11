'''
author: xu
Time: 2024/6/11
the offical code of MSPIF for test
'''

#导入所需要的包
import torch
from RTNet import model,Conv,ResidualBlock
from utils import gamma_his, save_PIL,fusion_img,CLAHE
import time
import utils
import os
from PIL import Image
from torchvision import transforms

#设置运行方式
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Ir_eval_path = 'test/ir'
Iv_eval_path = 'test/vis'
   
path = 'output/'
#模型加载
model_path = "model/RTNet.pth"
Model = torch.load(model_path,map_location='cpu')
image_names = [f for f in os.listdir(Ir_eval_path) if os.path.isfile(os.path.join(Ir_eval_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_names_iv = [f for f in os.listdir(Iv_eval_path) if os.path.isfile(os.path.join(Iv_eval_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#参数个数
total_params = sum(p.numel() for p in Model.parameters() if p.requires_grad)
print("Total parameters in the model: ", total_params)

# 定义transforms来将图片转换为tensor

def tensor_rgb2ycbcr(img_rgb):
	R = img_rgb[:,0, :, :]
	G = img_rgb[:,1, :, :]
	B = img_rgb[:,2, :, :]
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
	return Y.unsqueeze(0), Cb.unsqueeze(0), Cr.unsqueeze(0)

def tensor_ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    return torch.cat([R,G,B],1)

if __name__ =='__main__':
    #导入数据
  
    Time = 0
    Model.eval()
    with torch.no_grad():
        for i in range(len(image_names)):
            #模型使用
            #forward
            begin_time = time.time()
            Ir,_,_ = utils.image_trans(Image.open(os.path.join(Ir_eval_path,image_names[i])))
            Iv,h,w = utils.image_trans(Image.open(os.path.join(Iv_eval_path,image_names_iv[i])))
            _,c,_,_ = Iv.size()
            transform = transforms.Compose([transforms.Resize((h,w))])
            if c == 3:
                gray, img1_cb, img1_cr = tensor_rgb2ycbcr(Iv)
                Iv_gamma = gamma_his(gray)
                Iv_clahe = CLAHE(gray)
                Iv_enhance = utils.fusion_enhance(gray,Iv_gamma,Iv_clahe)
                RI,RR = Model(Ir)
                VI,VR = Model(Iv_enhance)
                I,R = fusion_img(RR,RI,VR,VI,i)
                S = transform(tensor_ycbcr2rgb(I*R, img1_cb, img1_cr))
            else:
                Iv_gamma = gamma_his(Iv)
                Iv_clahe = CLAHE(Iv)
                Iv_enhance = utils.fusion_enhance(Iv,Iv_gamma,Iv_clahe)
                RI,RR = Model(Ir)
                VI,VR = Model(Iv_enhance)
                I,R = fusion_img(RR,RI,VR,VI,i)
                S = transform(I*R)

            if os.path.exists(path):
                pass
            else:
                os.mkdir(path)

            path_S  = os.path.join(path,str(i+1)+'.PNG')        
            save_PIL(S,path_S)
            proc_time = time.time() - begin_time 
            Time += proc_time/len(image_names)
            print('Total processing time of {}: {:.3}s'.format(i+1,proc_time))
        print('the avg time is :{}'.format(Time))



       
        
