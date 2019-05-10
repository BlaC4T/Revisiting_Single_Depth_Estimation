import argparse
import torch
import torch.nn.parallel
import torch.nn.functional as F
from PIL import Image
from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb
import os
import json
import matplotlib.image
import matplotlib.pyplot as plt
# plt.set_cmap("jet")
# plt.set_cmap("gray")

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    model.eval()

    DetList = json.load( open( 'data/test/hoi_01.json' ) )
    indir = 'data/test/hoi_01'
    inframes = [f for f in os.listdir(indir) if os.path.isfile(os.path.join(indir, f))]
    inframes.sort(key = lambda x: int(x.split('.')[0]))


    for frame in inframes:
        fid = frame.split('.')[0]
        fname = os.path.join(indir, frame)
        ori_frame = Image.open(fname).convert("RGB")

        if str(int(fid)) not in DetList.keys():
            outframe = ori_frame
            outframe.save('data/test/outputs/'+fid+'.jpg')
            print(fid)


        else:
        #     ori_size = ori_frame.size
            nyu2_loader = loaddata.readNyu2(fname)
            detframe = DetList[str(int(fid))]
                
            with torch.no_grad():
                test(fid, detframe, nyu2_loader, model, ori_frame)


def test(fid, detframe, nyu2_loader, model, ori_frame):
    ori_size = ori_frame.size
    for i, image in enumerate(nyu2_loader):     
        image = torch.autograd.Variable(image, volatile=True).cuda()
        out = model(image)
        # print(out.shape)
        out = F.upsample(out, size=(ori_size[1], ori_size[0]), mode='bilinear')
##################################BBOX#################################33

        # print(out.shape)
        im_file = 'data/test/depths/'+ fid +'.jpg'
        # print(out.view(out.size(2),out.size(3)).data.cpu().numpy().shape)
##################################################################################
        matplotlib.image.imsave(im_file, out.view(out.size(2),out.size(3)).data.cpu().numpy())

        im_data = plt.imread(im_file)
        print("shape before : ", im_data.shape)
        gray_im = 0.2989 * im_data[:,:,0] + 0.5870 * im_data[:,:,1] + 0.1140 * im_data[:,:,2]

        im_data = gray_im

        # print("type : ", type(im_data))


        # (height, width, nbands) = im_data.shape
        (height, width) = im_data.shape
        dpi = 106 #모니터 따라 바꿔줘야 함 https://www.infobyip.com/detectmonitordpi.php
        print("shape after : ", im_data.shape)
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize, dpi = 106)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        # ax.imshow(im_data, interpolation='nearest')
        ax.imshow(np.asarray(ori_frame), interpolation='nearest')
        print(im_data.shape)

        # O_box = [240.0, 180.0, 350.0, 260.0]
        O_box = [560.0, 95.0, 730.0, 290.0]
        
        QO_width = int(0.25*(O_box[2] - O_box[0]))
        QO_height = int(0.25*(O_box[3] - O_box[1]))
        NO_box = [int(O_box[0] + QO_width), int(O_box[1] + QO_height), int(O_box[2] - QO_width), int(O_box[3] - QO_height)]


        ax.add_patch(
                        plt.Rectangle((O_box[0], O_box[1]), O_box[2] - O_box[0], O_box[3] - O_box[1], fill=False, linewidth=3)
                    )

        ax.add_patch(
                        plt.Rectangle((NO_box[0], NO_box[1]), NO_box[2] - NO_box[0], NO_box[3] - NO_box[1], fill=False, linewidth=3)
                    )

        for obj in detframe.keys():
            if obj == 'person':
                PerList = detframe[obj]
                for person in PerList:
                    P_box = person['bbox']
                    P_box[0] = min(width, max(0.0, P_box[0]))
                    P_box[2] = min(width, max(0.0, P_box[2]))
                    P_box[1] = min(height, max(0.0, P_box[1]))
                    P_box[3] = min(height, max(0.0, P_box[3]))
                    P_score = person['score']

                    QP_width = int(0.25*(P_box[2] - P_box[0]))
                    QP_height = int(0.25*(P_box[3] - P_box[1]))
                    NP_box = [int(P_box[0] + QP_width), int(P_box[1] + QP_height), int(P_box[2] - QP_width), int(P_box[3] - QP_height)]


                    ax.add_patch(
                        plt.Rectangle((P_box[0], P_box[1]), P_box[2] - P_box[0], P_box[3] - P_box[1], fill=False, linewidth=3)
                    )
                    ax.add_patch(
                        plt.Rectangle((NP_box[0], NP_box[1]), NP_box[2] - NP_box[0], NP_box[3] - NP_box[1], fill=False, linewidth=3)
                    )

                    xA = int(max(P_box[0], O_box[0]))
                    yA = int(max(P_box[1], O_box[1]))
                    xB = int(min(P_box[2], O_box[2]))
                    yB = int(min(P_box[3], O_box[3]))
                    print(type(xA))
                    I_box = [xA, yA, xB, yB]

                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                    if interArea != 0:

                        InterDepth = im_data[xA:xB, yA:yB]
                        PDepth = im_data[int(P_box[0]):int(P_box[2]), int(P_box[1]):int(P_box[3])]
                        ODepth = im_data[int(O_box[0]):int(O_box[2]), int(O_box[1]):int(O_box[3])]
                        NPDepth = im_data[int(NP_box[0]):int(NP_box[2]), int(NP_box[1]):int(NP_box[3])]
                        NODepth = im_data[int(NO_box[0]):int(NO_box[2]), int(NO_box[1]):int(NO_box[3])]

                        if abs(np.mean(NPDepth) - np.mean(NODepth)) < 15:
                            box_color = (1, 0, 0)
                        else:
                            box_color = (1, 1, 1)

                        ax.add_patch(
                                plt.Rectangle((I_box[0], I_box[1]), I_box[2] - I_box[0], I_box[3] - I_box[1], color = box_color, fill=False, linewidth=3)
                                )

                        
                        # if fid == '00213' or fid == '01154' or fid == '01155' or fid == '01157':
                        #     print(fid)
                        #     print("type Inter : ", type(InterDepth))
                        #     print("shape Inter : ", InterDepth.shape)
                        #     print("Obj Inter : ", np.mean(ODepth))
                        #     print("New_Obj Inter : ", np.mean(NODepth))
                        #     print("Hum Inter : ", np.mean(PDepth))
                        #     print("New_Hum Inter : ", np.mean(NPDepth))
                        #     print("Avg Inter : ", np.mean(InterDepth))
                        #     print("Max Inter : ", np.max(im_data))
                        #     print("Min Inter : ", np.min(im_data))
        plt.savefig('data/test/outputs/'+ fid +'.jpg')
        plt.close()

if __name__ == '__main__':
    main()
