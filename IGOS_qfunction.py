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

    # TODO: make function that will do the check of use_cuda and return the proper data information

    mask, category = Integrated_Mask(input_img, blurred_img, model, img_label, use_cuda,
                                    max_iterations=15, integ_iter=20, l1_coeff=0.01 * 100)





def Get_blurred_img():

    blurred_img = np.random.randint(5, size=68)
    blurred_img = blurred_img.astype(np.float32)

    return blurred_img

def see_top_labels(input_img, blurred_img, model, use_cuda):

    img_torch = preprocess_qfunction(input_img, use_cuda, require_grad = False)
    blurred_img_torch = preprocess_qfunction(blurred_img, use_cuda, require_grad = False)

    ori_output = model(img_torch).data.numpy()
    blurred_output = model(blurred_img_torch).data.numpy()

    top_idxs = np.argsort(ori_output)
    top_values = [ori_output[i] for i in top_idxs]

    print('Index\tPrediction Score')
    for i in reversed(range(len(ori_output))):
        print("  {}\t{}".format(top_idxs[i], top_values[i]))


def Integrated_Mask(input_img, blurred_img, model, img_label, use_cuda = 0, 
                    max_iterations = 15, integ_iter = 20, l1_coeff = 0.01 * 300):

    input_img = preprocess_qfunction(input_img, use_cuda, require_grad=False)
    blurred_img = preprocess_qfunction(blurred_img, use_cuda, require_grad=False)


    mask = np.ones(68, dtype=np.float32)
    mask = preprocess_qfunction(mask, require_grad=True, use_cuda=0)

    optimizer = torch.optim.Adam([mask], lr=0.1)
    # optimizer = torch.optim.SGD([mask], lr=0.1)

    target = model(img).data.numpy()
    category_out = np.argmax(target.data.numpy())

    if category == -1:
        category = category_out

    print("Category with highest probability", category_out)
    print("Category want to generate mask", category)
    print("Optimizing.. ")

    alpha = 0.0001
    beta = 0.2

    for i in range(max_iterations):

        loss1 = l1_coeff * torch.mean(torch.abs(1 - mask))
        loss_all = loss1.clone()

        perturbated_input_base = img.mul(mask) + blurred_img.mul(1 - mask)

        for inte_i in range(integ_iter):

            integ_mask = 0.0 + ((inte_i + 1.0) / integ_iter) * upsampled_mask

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
        loss_oridata = loss_ori.data.numpy()


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

            loss_LSdata = loss_LS.data.numpy()


            new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            new_condition = new_condition.sum()
            new_condition = alpha * step * new_condition


            loss_with_new_condition = loss_oridata - new_condition.cpu().numpy()

            if not (loss_LSdata > loss_with_new_condition) or step < 0.00001:
                break

            step *= beta


        mask.data -= step * whole_grad
        mask.data.clamp_(0, 1)
        maskdata = mask.data.numpy()



if __name__ == "__main__":
    main()
