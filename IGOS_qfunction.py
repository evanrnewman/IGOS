from util import *
import os
import time
import scipy.io as scio
import datetime
import re
import matplotlib.pyplot as plt
import numpy as np
import pylab
import os
import csv
from skimage import transform, filters
from textwrap import wrap
import cv2

def main():

    use_cuda = 0

    model = load_model_new(use_cuda=use_cuda, model_name='q_function')
    print(model)


    data = torch.load('data/test_dataset.pt')
    data = np.asarray(data, dtype=np.float32)

    input_img = data[0][0]
    img_label = -1

    blurred_img = Get_blurred_img()

    see_top_labels(input_img, blurred_img, model, use_cuda)

    mask, category = Integrated_Mask(input_img, blurred_img, model, img_label, use_cuda,
                                    max_iterations=15, integ_iter=20, l1_coeff=0.01 * 100)


    interpret_mask_meaning(mask)



def Get_blurred_img():

    blurred_img = np.random.randint(5, size=68)
    blurred_img = blurred_img.astype(np.float32)

    return blurred_img

def see_top_labels(input_img, blurred_img, model, use_cuda):

    img_torch         = preprocess_qfunction(input_img,   use_cuda, require_grad = False)
    blurred_img_torch = preprocess_qfunction(blurred_img, use_cuda, require_grad = False)

    ori_output     = is_using_cuda( model(img_torch),         use_cuda )
    blurred_output = is_using_cuda( model(blurred_img_torch), use_cuda )

    top_idxs = np.argsort(ori_output)
    top_values = [ori_output[i] for i in top_idxs]

    print('Index\tPrediction Score')
    for i in reversed(range(len(ori_output))):
        print("  {}\t{}".format(top_idxs[i], top_values[i]))
    print("ori_output:\t{}".format(ori_output))

def is_using_cuda(data_term, use_cuda):
    if use_cuda:
        data_term = data_term.data.cpu().numpy()
    else:
        data_term = data_term.data.numpy()

    return data_term


def Integrated_Mask(img, blurred_img, model, category, use_cuda = 0, 
                    max_iterations = 15, integ_iter = 20, l1_coeff = 0.01 * 300):

    img = preprocess_qfunction(img, use_cuda, require_grad=False)
    blurred_img = preprocess_qfunction(blurred_img, use_cuda, require_grad=False)


    mask = np.ones(68, dtype=np.float32)
    mask = preprocess_qfunction(mask, require_grad=True, use_cuda=0)

    optimizer = torch.optim.Adam([mask], lr=0.1)
    # optimizer = torch.optim.SGD([mask], lr=0.1)

    target = is_using_cuda( model(img), use_cuda )
    category_out = np.argmax(target)

    if category == -1:
        category = category_out

    print("Category with highest probability", category_out)
    print("Category want to generate mask", category)
    print("Optimizing.. ")

    curve1 = np.array([])
    curve2 = np.array([])
    curvetop = np.array([])

    alpha = 0.0001
    beta = 0.2

    for i in range(max_iterations):

        loss1 = l1_coeff * torch.mean(torch.abs(1 - mask))
        loss_all = loss1.clone()

        perturbated_input_base = img.mul(mask) + blurred_img.mul(1 - mask)

        for inte_i in range(integ_iter):

            integ_mask = 0.0 + ((inte_i + 1.0) / integ_iter) * mask

            perturbated_input_integ = img.mul(integ_mask) + blurred_img.mul(1 - integ_mask)


            noise = np.zeros(68, dtype=np.float32)
            noise = noise + cv2.randn(noise, 0, 0.2)
            noise = preprocess_qfunction(noise, use_cuda, require_grad=False)


            perturbated_input_img = perturbated_input_integ + noise

            outputs = model(perturbated_input_img)
            loss2 = outputs[category]
            loss_all = loss_all + loss2/20.0


        optimizer.zero_grad()
        loss_all.backward()

        whole_grad = mask.grad.clone()

        loss2_ori = model(perturbated_input_base)[category]
        loss_ori = loss1 + loss2_ori
        loss_oridata = is_using_cuda( loss_ori, use_cuda )

        if i==0:
            if use_cuda:
                curve1 = np.append(curve1, loss1.data.cpu().numpy())
                curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
                curvetop = np.append(curvetop, loss2_ori.data.cpu().numpy())

            else:
                curve1 = np.append(curve1, loss1.data.numpy())
                curve2 = np.append(curve2, loss2_ori.data.numpy())
                curvetop = np.append(curvetop, loss2_ori.data.numpy())


        step = 200.0

        while True:
            MaskClone = mask.data.clone()
            MaskClone -= step * whole_grad
            MaskClone = Variable(MaskClone, requires_grad=False)
            MaskClone.data.clamp_(0, 1)

            mask_LS = MaskClone
            Img_LS = img.mul(mask_LS) + blurred_img.mul(1 - mask_LS)

            outputsLS = model(Img_LS)
            loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + outputsLS[category]

            loss_LSdata = is_using_cuda( loss_LS, use_cuda )


            new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            new_condition = new_condition.sum()
            new_condition = alpha * step * new_condition


            loss_with_new_condition = loss_oridata - new_condition.cpu().numpy()

            if not (loss_LSdata > loss_with_new_condition) or step < 0.00001:
                break

            step *= beta


        if use_cuda:
            curve1 = np.append(curve1, loss1.data.cpu().numpy())
            curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
        else:
            curve1 = np.append(curve1, loss1.data.numpy())
            curve2 = np.append(curve2, loss2_ori.data.numpy())


        mask.data -= step * whole_grad
        mask.data.clamp_(0, 1)
        maskdata = is_using_cuda( mask, use_cuda )

        # EVAN: This is a necessary process to pick the most important pixels to mask out, how many is up to us
        maskdata, imgratio = topmaxPixel(maskdata, 3)

        Masktop = preprocess_qfunction(maskdata, use_cuda, require_grad=False)

        MasktopLS = Masktop

        Img_topLS = img.mul(MasktopLS) + blurred_img.mul(1 - MasktopLS)

        outputstopLS = model(Img_topLS)
        loss_top1 = l1_coeff * torch.mean(torch.abs(1 - Masktop))

        print("iteration:{}".format(i))
        print("outputstopLS:\t{}".format(outputstopLS))
        print("category:\t{}".format(category))
        print("mask:\t{}".format(maskdata))

        loss_top2 = outputstopLS[category]


        if max_iterations >3:

            if i == int(max_iterations / 2):
                if np.abs(curve2[0] - curve2[i]) <= 0.001:
                    print('Adjust Parameter l1_coeff at iteration:', int(max_iterations / 2))
                    l1_coeff = l1_coeff / 10

            elif i == int(max_iterations / 1.25):
                if np.abs(curve2[0] - curve2[i]) <= 0.01:
                    print('Adjust Parameters l1_coeff again at iteration:', int(max_iterations / 1.25))
                    l1_coeff = l1_coeff / 5

    mask = is_using_cuda( mask, use_cuda ).copy()

    return mask, category


def interpret_mask_meaning(mask):

    key_input = ["player 1 unspent minerals",
                 "player 1 top lane buildings ",
                 "player 1 bottom lane buildings ",
                 "player 1 pylons",
                 "player 2 top lane buildings ",
                 "player 2 bottom lane buildings ",
                 "player 2 pylons",
                 "player 1 units top lane grid 1 ",
                 "player 1 units top lane grid 2 ",
                 "player 1 units top lane grid 3 ",
                 "player 1 units top lane grid 4 ",
                 "player 1 units bottom lane grid 1 ",
                 "player 1 units bottom lane grid 2 ",
                 "player 1 units bottom lane grid 3 ",
                 "player 1 units bottom lane grid 4 ",
                 "player 2 units top lane grid 1 ",
                 "player 2 units top lane grid 2 ",
                 "player 2 units top lane grid 3 ",
                 "player 2 units top lane grid 4 ",
                 "player 2 units bottom lane grid 1 ",
                 "player 2 units bottom lane grid 2 ",
                 "player 2 units bottom lane grid 3 ",
                 "player 2 units bottom lane grid 4 ",
                 "player 1 nexus HP top lane",
                 "player 1 nexus HP bottom lane",
                 "player 2 nexus HP top lane",
                 "player 2 nexus HP bottom lane",
                 "wave number"]

    units = ["marine", "bane", "immortal"]

    idx = 0
    for key in key_input:
        if key[-1] == " ":
            for unit in units:
                print("{} | {:<50} = {}".format(str(idx), (key + unit), mask[idx]))
                idx += 1
        else:
            print("{} | {:<50} = {}".format(str(idx), key, mask[idx]))
            idx += 1
            




if __name__ == "__main__":
    main()
