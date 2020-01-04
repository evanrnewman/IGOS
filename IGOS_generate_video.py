#coding=utf-8
# Generating video using I-GOS
# python Version: python3.6
# by Zhongang Qi (qiz@oregonstate.edu)
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






def Get_blurred_img(input_img, img_label, model, resize_shape=(224, 224), Gaussian_param = [51, 50], 
                    Median_param = 11, blur_type= 'Gaussian', use_cuda = 1):
    ########################
    # Generate blurred images as the baseline

    # Parameters:
    # -------------
    # input_img: the original input image
    # img_label: the classification target that you want to visualize (img_label=-1 means the top 1 classification label)
    # model: the model that you want to visualize
    # resize_shape: the input size for the given model
    # Gaussian_param: parameters for Gaussian blur
    # Median_param: parameters for median blur
    # blur_type: Gaussian blur or median blur or mixed blur
    # use_cuda: use gpu (1) or not (0)
    ####################################################

    # EVAN: all this is doing img manipulation not needed for an already 1d vector input we have
    # original_img = cv2.imread(input_img, 1)
    # original_img = cv2.resize(original_img, resize_shape)
    # img = np.float32(original_img) / 255
    original_img = input_img
    img = input_img

    if blur_type =='Gaussian':   # Gaussian blur
        Kernelsize = Gaussian_param[0]
        SigmaX = Gaussian_param[1]
        blurred_img = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)

    elif blur_type == 'Median': # Median blur
        Kernelsize_M = Median_param
        blurred_img = np.float32(cv2.medianBlur(original_img, Kernelsize_M)) / 255

    elif blur_type == 'Mixed': # Mixed blur
        Kernelsize = Gaussian_param[0]
        SigmaX = Gaussian_param[1]
        blurred_img1 = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)

        Kernelsize_M = Median_param
        blurred_img2 = np.float32(cv2.medianBlur(original_img, Kernelsize_M)) / 255

        blurred_img = (blurred_img1 + blurred_img2) / 2
    elif blur_type == 'q_function':
        # Totally random array at the moment for the blur
        blurred_img = np.random.randint(5, size=68)
        blurred_img = blurred_img.astype(np.float32)
        blurred_img = torch.from_numpy(blurred_img)
    # !! IDEA: make it so blurred image is where all labels are equally likely to happen (all 1/8 chance of being label)
        print("blurred img:\t{}".format(blurred_img))

    # EVAN: pre processing to turn data into torch versions and use cuda need be
    # TODO: SHOULD look into moving the preprocessing into a new function that does this instead of in line
    # img_torch = preprocess_image(img, use_cuda, require_grad = False)
    # blurred_img_torch = preprocess_image(blurred_img, use_cuda, require_grad = False)

    # already torch vectors so okay to pass into model
    ori_output = model(img)
    blurred_output = model(blurred_img)

    print("ori prediction:\t{}".format(ori_output))
    # EVAN: more preprocessing to turn image into 1D tensor
    # compute the outputs for the original image and the blurred image
    if use_cuda:
        logitori = ori_output.data.cpu().numpy().copy().squeeze()
        logitblur = blurred_output.data.cpu().numpy().copy().squeeze()
    else:
        logitori = ori_output.data.numpy().copy().squeeze()
        logitblur = blurred_output.data.numpy().copy().squeeze()


    # EVAN: Everything below here is just printing information and 
    # in case no top classification picked chooses default label though this logic is never returned?

    print("logitori:\t{}".format(logitori))

    top_5_idx = np.argsort(logitori)[-5:]
    top_5_values = [logitori[i] for i in top_5_idx]
    print('top_5_idx:', top_5_idx, top_5_values)

    # find the original top 1 classification label
    rew = np.where(logitori == np.max(logitori))
    #print(rew)
    output_label = rew[0][0]

    # if img_label=-1, choose the original top 1 label as the one that you want to visualize
    if img_label == -1:
        img_label = output_label

    rew_blur = np.where(logitblur == np.max(logitblur))
    output_label_blur = rew_blur[0][0]



    #print('ori_output:', ori_output[0, img_label], output_label)
    #print('blurred_output:', blurred_output[0, img_label], output_label_blur)
    # EVAN: Not really sure what's going on here
    # blur_ratio = blurred_output[0, img_label] / ori_output[0, img_label]
    #print('blur_ratio:', blur_ratio)


    return img, blurred_img, logitori


def Integrated_Mask(img, blurred_img, model, category, max_iterations = 15, integ_iter = 20,
                    tv_beta=2, l1_coeff = 0.01*300, tv_coeff = 0.2*300, size_init = 112, use_cuda =1):
    ########################
    # IGOS: using integrated gradient descent to find the smallest and smoothest area that maximally decrease the
    # output of a deep model

    # Parameters:
    # -------------
    # img: the original input image
    # blurred_img: the baseline for the input image
    # model: the model that you want to visualize
    # category: the classification target that you want to visualize (category=-1 means the top 1 classification label)
    # max_iterations: the max iterations for the integrated gradient descent
    # integ_iter: how many points you want to use when computing the integrated gradients
    # tv_beta: which norm you want to use for the total variation term
    # l1_coeff: parameter for the L1 norm
    # tv_coeff: parameter for the total variation term
    # size_init: the resolution of the mask that you want to generate
    # use_cuda: use gpu (1) or not (0)
    ####################################################

    # preprocess the input image and the baseline image
    # EVAN: TODO: make this it's own thing as in the get_blurred_img function
    # img = preprocess_image(img, use_cuda, require_grad=False)
    # blurred_img = preprocess_image(blurred_img, use_cuda, require_grad=False)

    # EVAN: already in vector form so no need to resize
    resize_size = img.data.shape
    print("resize_size:\t{}".format(resize_size))
    # NOTE: EVAN: as explaned in the following "NOTE" this would only be needed if blurred image is smaller than a 68 vector and 
    #             would need to scale up but because that's not the case commenting out for now
    # resize_wh = (img.data.shape[2], img.data.shape[3])

    # EVAN: this zero_img is never used so commenting out for now
    # if use_cuda:
    #     zero_img = Variable(torch.zeros(resize_size).cuda(), requires_grad=False)
    # else:
    #     zero_img = Variable(torch.zeros(resize_size), requires_grad=False)


    # initialize the mask
    # mask_init = np.ones((size_init, size_init), dtype=np.float32)
    mask_init = np.ones(68, dtype=np.float32)#.squeeze()
    # EVAN: NOTE: Not sure why numpyt_to_torch operation is used, seems like 
    #       from_numpy is just fine. Possibly seems like numpy_to_torch is deprecated b/c can't seem to find documentation
    # mask = numpy_to_torch(mask_init, requires_grad=True, use_cuda=1)
    mask = torch.from_numpy(mask_init)#, requires_grad=True, use_cuda=0)
    # mask = torch.squeeze(mask)
    print("mask shape: {}".format(mask.shape))
    print("mask:\t{}".format(mask))

    # EVAN: NOTE: This depends on the blurred image. If we were to use a small blur vector of 5 and then scale up to 68
    #               then we should look into upsampling techniques of 1D vectors. As of now that is not needed b/c 
    #               blurred image is already a 68 input vector
    # if use_cuda:
    #     upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    # else:
    #     upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)

    # You can choose any optimizer
    # The optimizer doesn't matter, because we don't need optimizer.step(), we just use it to compute the gradient
    optimizer = torch.optim.Adam([mask], lr=0.1)
    # optimizer = torch.optim.SGD([mask], lr=0.1)

    # EVAN: Model already goes through a softmax layer so removing for now
    # target = torch.nn.Softmax(dim=1)(model(img))
    target = model(img)
    if use_cuda:
        category_out = np.argmax(target.cpu().data.numpy())
    else:
        category_out = np.argmax(target.data.numpy())

    # if category=-1, choose the original top 1 category as the one that you want to visualize
    if category ==-1:
        category = category_out

    print("Category with highest probability", category_out)
    print("Category want to generate mask", category)
    print("Optimizing.. ")




    curve1 = np.array([])
    curve2 = np.array([])
    curvetop = np.array([])


    # Integrated gradient descent
    alpha = 0.0001
    beta = 0.2

    for i in range(max_iterations):
        # EVAN: No upsample as noted above in previose NOTEs
        # upsampled_mask = upsample(mask)
        upsampled_mask = mask

        # EVAN: not needed as vector so no RGB channels needed
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channels
        # upsampled_mask = \
        #     upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
        #                           upsampled_mask.size(3))


        # EVAN: NOTE: The toal variation is suppose to help with smoothness as the name suggests
        #               probably taking the total variation and normalizing it
        #               However, this seems only relevant to images as the smoothness helps the pixels between eachother
        #               not be so drastic. The vector we have though should be different from each other as each one
        #               reflects a different meaning, unlike an image which is to communicate one thing with all it's pixels
        #       NOTE: The l1 term will help penilize all 0 masks which we would like to use
        #               BUT having an all 0 mask would still symbolize a State so not sure
        #       TODO: Talk to Alan about this issue

        # the l1 term and the total variation term
        loss1 = l1_coeff * torch.mean(torch.abs(1 - mask)) #+ \
                # tv_coeff * tv_norm(mask, tv_beta)
        loss_all = loss1.clone()

        # compute the perturbed image
        perturbated_input_base = img.mul(upsampled_mask) + \
                                 blurred_img.mul(1 - upsampled_mask)


        for inte_i in range(integ_iter):


            # EVAN: the more loops the less the mask is perturbed
            #       seems like it's just getting a lower percent of the original mask which is just all 1's
            #       so first iteration will be getting a mask filled with just 1/20=0.05 values and ticks up with each iteration

            # Use the mask to perturbated the input image.
            integ_mask = 0.0 + ((inte_i + 1.0) / integ_iter) * upsampled_mask


            perturbated_input_integ = img.mul(integ_mask) + \
                                     blurred_img.mul(1 - integ_mask)

            # EVAN: not sure if noise is needed for our vector stuff I'm going to keep it in for now 
            #       NOTE: if causes issues further ahead should remove noise logic for mask

            # add noise
            # noise = np.zeros((resize_wh[0], resize_wh[1], 3), dtype=np.float32)
            noise = np.zeros(68, dtype=np.float32)
            noise = noise + cv2.randn(noise, 0, 0.2)
            noise = torch.from_numpy(noise)
            # EVAN: same issues noted above about this function so using from_numpy for now
            # noise = numpy_to_torch(noise, use_cuda, requires_grad=False)

            perturbated_input = perturbated_input_integ + noise

            new_image = perturbated_input
            # EVAN: model already has softmax so no need to pass it through this layer
            # outputs = torch.nn.Softmax(dim=1)(model(new_image))
            outputs = model(new_image)
            loss2 = outputs[0, category]

            loss_all = loss_all + loss2/20.0


        # compute the integrated gradients for the given target,
        # and compute the gradient for the l1 term and the total variation term
        optimizer.zero_grad()
        loss_all.backward()
        whole_grad = mask.grad.data.clone()

        # EVAN: model already has softmax layer
        # loss2_ori = torch.nn.Softmax(dim=1)(model(perturbated_input_base))[0, category]
        loss2_ori = model(perturbated_input_base)[0, category]



        loss_ori = loss1 + loss2_ori
        if i==0:
            if use_cuda:
                curve1 = np.append(curve1, loss1.data.cpu().numpy())
                curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
                curvetop = np.append(curvetop, loss2_ori.data.cpu().numpy())

            else:
                curve1 = np.append(curve1, loss1.data.numpy())
                curve2 = np.append(curve2, loss2_ori.data.numpy())
                curvetop = np.append(curvetop, loss2_ori.data.numpy())



        if use_cuda:
            loss_oridata = loss_ori.data.cpu().numpy()
        else:
            loss_oridata = loss_ori.data.numpy()



        # LINE SEARCH with revised Armijo condition
        step = 200.0
        MaskClone = mask.data.clone()
        MaskClone -= step * whole_grad
        MaskClone = Variable(MaskClone, requires_grad=False)
        MaskClone.data.clamp_(0, 1) # clamp the value of mask in [0,1]


        # EVAN: same issues mentioned above upsample not needed as of NOW
        # mask_LS = upsample(MaskClone)   # Here the direction is the whole_grad
        mask_LS = MaskClone
        Img_LS = img.mul(mask_LS) + \
                 blurred_img.mul(1 - mask_LS)
        # outputsLS = torch.nn.Softmax(dim=1)(model(Img_LS))
        outputsLS = model(Img_LS)
        loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) #+ \
                #   tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]

        if use_cuda:
            loss_LSdata = loss_LS.data.cpu().numpy()
        else:
            loss_LSdata = loss_LS.data.numpy()


        new_condition = whole_grad ** 2  # Here the direction is the whole_grad
        new_condition = new_condition.sum()
        new_condition = alpha * step * new_condition

        while loss_LSdata > loss_oridata - new_condition.cpu().numpy():
            step *= beta

            MaskClone = mask.data.clone()
            MaskClone -= step * whole_grad
            MaskClone = Variable(MaskClone, requires_grad=False)
            MaskClone.data.clamp_(0, 1)
            # mask_LS = upsample(MaskClone)
            mask_LS = MaskClone
            Img_LS = img.mul(mask_LS) + \
                     blurred_img.mul(1 - mask_LS)
            # outputsLS = torch.nn.Softmax(dim=1)(model(Img_LS))
            outputsLS = model(Img_LS)
            loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) # + \
                    #   tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]

            if use_cuda:
                loss_LSdata = loss_LS.data.cpu().numpy()
            else:
                loss_LSdata = loss_LS.data.numpy()


            new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            new_condition = new_condition.sum()
            new_condition = alpha * step * new_condition

            if step<0.00001:
                break

        mask.data -= step * whole_grad

        #######################################################################################################


        if use_cuda:
            curve1 = np.append(curve1, loss1.data.cpu().numpy())
            curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
        else:
            curve1 = np.append(curve1, loss1.data.numpy())
            curve2 = np.append(curve2, loss2_ori.data.numpy())

        mask.data.clamp_(0, 1)
        if use_cuda:
            maskdata = mask.data.cpu().numpy()
        else:
            maskdata = mask.data.numpy()

        maskdata = np.squeeze(maskdata)
        maskdata, imgratio = topmaxPixel(maskdata, 40)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        # MasktopLS = upsample(Masktop)
        MasktopLS = Masktop

        Img_topLS = img.mul(MasktopLS) + \
                    blurred_img.mul(1 - MasktopLS)
        # outputstopLS = torch.nn.Softmax(dim=1)(model(Img_topLS))
        outputstopLS = model(Img_topLS)
        loss_top1 = l1_coeff * torch.mean(torch.abs(1 - Masktop)) #+ \
                    # tv_coeff * tv_norm(Masktop, tv_beta)
        loss_top2 = outputstopLS[0, category]

        if use_cuda:
            curvetop = np.append(curvetop, loss_top2.data.cpu().numpy())
        else:
            curvetop = np.append(curvetop, loss_top2.data.numpy())


        if max_iterations >3:

            if i == int(max_iterations / 2):
                if np.abs(curve2[0] - curve2[i]) <= 0.001:
                    print('Adjust Parameter l1_coeff at iteration:', int(max_iterations / 2))
                    l1_coeff = l1_coeff / 10

            elif i == int(max_iterations / 1.25):
                if np.abs(curve2[0] - curve2[i]) <= 0.01:
                    print('Adjust Parameters l1_coeff again at iteration:', int(max_iterations / 1.25))
                    l1_coeff = l1_coeff / 5


            #######################################################################################

    # upsampled_mask = upsample(mask)
    upsampled_mask = mask

    if use_cuda:
        mask = mask.data.cpu().numpy().copy()
    else:
        mask = mask.data.numpy().copy()

    return mask, upsampled_mask, imgratio, curvetop, curve1, curve2, category







def save_new(mask, img, blurred):
    ########################
    # generate the perturbed image
    #
    # parameters:
    # mask: the generated mask
    # img: the original image
    # blurred: the baseline image
    ####################################################
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    img = np.float32(img) / 255
    perturbated = np.multiply(mask, img) + np.multiply(1-mask, blurred)
    perturbated = cv2.cvtColor(perturbated, cv2.COLOR_BGR2RGB)
    return perturbated



def showimage(del_img, insert_img, del_curve, insert_curve, target_path, xtick, title):
    ########################
    # generate the result frame used for videos
    #
    # parameters:
    # del_img: the deletion image
    # insert_img: the insertion image
    # del_curve: the deletion curve
    # insert_curve: the insertion curve
    # target_path: where to save the results
    # xtick: xtick
    # title: title
    ####################################################
    pylab.rcParams['figure.figsize'] = (10, 10)
    f, ax = plt.subplots(2,2)
    f.suptitle('Category ' + title, y=0.04, fontsize=13)
    f.tight_layout()
    plt.subplots_adjust(left=0.005, bottom=0.1, right=0.98, top=0.93,
                        wspace=0.05, hspace=0.25)



    ax[0,0].imshow(del_img)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_title("Deletion", fontsize=13)

    ax[1,0].imshow(insert_img)
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_title("Insertion", fontsize=13)




    ax[0,1].plot(del_curve,'r*-')
    ax[0,1].set_xlabel('number of blocks')
    ax[0,1].set_ylabel('classification confidence')
    ax[0,1].legend(['Deletion'])
    ax[0,1].set_xticks(range(0, xtick, 10))
    ax[0, 1].set_yticks(np.arange(0, 1.1, 0.1))


    ax[1,1].plot(insert_curve, 'b*-')
    ax[1, 1].set_xlabel('number of blocks')
    ax[1,1].set_ylabel('classification confidence')
    ax[1,1].legend(['Insertion'])
    ax[1, 1].set_xticks(range(0, xtick, 10))
    ax[1, 1].set_yticks(np.arange(0, 1.1, 0.1))
    print(insert_curve.shape[0])

    plt.savefig(target_path + 'video'+ str(insert_curve.shape[0])+ '.jpg')
    plt.close()
    #plt.clf()






def Deletion_Insertion(mask, model, output_path, img_ori, blurred_img_ori, logitori, 
                        category, pixelnum = 200, use_cuda =1, blur_mask=0, outputfig = 1):
    ########################
    # Compute the deletion and insertion scores
    #
    # parameters:
    # mask: the generated mask
    # model: the model that you want to visualize
    # output_path: where to save the results
    # img_ori: the original image
    # blurred_img_ori: the baseline image
    # logitori: the original output for the input image
    # category: the classification target that you want to visualize (category=-1 means the top 1 classification label)
    # pixelnum: how many points you want to compute the deletion and insertion scores
    # use_cuda: use gpu (1) or not (0)
    # blur_mask: blur the mask or not
    # outputfig: save figure or not
    ####################################################


    sizeM = mask.shape[2] * mask.shape[3]



    if blur_mask:
        mask = (mask - np.min(mask)) / np.max(mask)
        mask = 1 - mask
        mask = cv2.GaussianBlur(mask, (51, 51), 50)
        mask = 1-mask


    blurred_insert = blurred_img_ori.copy()                             #todo
    blurred_insert = preprocess_image(blurred_insert, use_cuda, require_grad=False)

    img = preprocess_image(img_ori, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img_ori, use_cuda, require_grad=False)
    resize_wh = (img.data.shape[2], img.data.shape[3])
    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)

    target = torch.nn.Softmax(dim=1)(model(img))
    if use_cuda:
        category_out = np.argmax(target.cpu().data.numpy())
    else:
        category_out = np.argmax(target.data.numpy())

    if category == -1:
        category = category_out

    print("Category with highest probability", category_out)
    print("Category want to generate mask", category)

    outmax = target[0, category].cpu().data.numpy()
    logitori = logitori[category]

    del_curve = np.array([])
    insert_curve = np.array([])




    if sizeM<pixelnum:
        intM = 1
    else:
        intM = int(sizeM/pixelnum)


    xtick = np.arange(0, int(sizeM/3.5), intM)
    xnum = xtick.shape[0]
    print('xnum:', xnum)

    xtick = xtick.shape[0]+ 10

    # get the ground truth label for the given category
    f_groundtruth = open('./GroundTruth1000.txt')
    line_i = f_groundtruth.readlines()[category]
    f_groundtruth.close()
    print('line_i:', line_i)



    for pix_num in range(0, int(sizeM/3.5), intM):
        maskdata = mask.copy()
        maskdata = np.squeeze(maskdata)

        # only keep the top pixels in the mask for deletion
        maskdata, imgratio = topmaxPixel(maskdata, pix_num)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        MasktopLS = upsample(Masktop)
        Img_topLS = img.mul(MasktopLS) + \
                    blurred_img.mul(1 - MasktopLS)


        outputstopLS_ori = model(Img_topLS)[0, category].data.cpu().numpy().copy()
        outputstopLS = torch.nn.Softmax(dim=1)(model(Img_topLS))
        delloss_top2 = outputstopLS[0, category].data.cpu().numpy().copy()
        del_mask = MasktopLS.clone()

        delimg_ratio = imgratio.copy()
        del_ratio = delloss_top2 / outmax
        del_curve = np.append(del_curve, delloss_top2)



        maskdata = mask.copy()

        maskdata = np.squeeze(maskdata)

        # only keep the top pixels in the mask for insertion
        maskdata, imgratio = topmaxPixel_insertion(maskdata, pix_num)

        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        MasktopLS = upsample(Masktop)
        Img_topLS = img.mul(MasktopLS) + \
                    blurred_insert.mul(1 - MasktopLS)

        outputstopLS_ori = model(Img_topLS)[0, category].data.cpu().numpy().copy()
        outputstopLS = torch.nn.Softmax(dim=1)(model(Img_topLS))
        insloss_top2 = outputstopLS[0, category].data.cpu().numpy().copy()
        ins_mask = MasktopLS.clone()


        insimg_ratio = imgratio.copy()
        ins_ratio = insloss_top2 / outmax
        insert_curve = np.append(insert_curve, insloss_top2)

        if outputfig == 1:
            deletion_img = save_new(del_mask, img_ori * 255, blurred_img_ori)

            insertion_img = save_new(ins_mask, img_ori * 255, blurred_img_ori)
            showimage(deletion_img, insertion_img, del_curve, insert_curve, output_path, xtick, line_i)



    outmax = np.around(outmax, decimals=3)
    delloss_top2 = np.around(delloss_top2, decimals=3)
    del_ratio = np.around(del_ratio, decimals=3)
    delimg_ratio = np.around(delimg_ratio, decimals=3)

    insloss_top2 = np.around(insloss_top2, decimals=3)
    ins_ratio = np.around(ins_ratio, decimals=3)
    insimg_ratio = np.around(insimg_ratio, decimals=3)


    return del_mask, ins_mask, delloss_top2, insloss_top2, del_ratio, ins_ratio, outmax, category, xnum



def write_video(inputpath, outputname, img_num, fps = 10):
    ########################
    # Generate videos
    #
    # parameters:
    # inputpath: the path for input images
    # outputname: the output name for the video
    # img_num: the number of the input images
    # fps: frames per second
    ####################################################


    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(outputname, fourcc, fps, (1000, 1000))
    for i in range(img_num):

        img_no = i+1
        print(inputpath+'video'+str(img_no) +'.jpg')
        img12 = cv2.imread(inputpath+'video'+str(img_no) +'.jpg',1)
        videoWriter.write(img12)
    videoWriter.release()





if __name__ == '__main__':


    input_path = './Images/'

    output_path = './Results/VIDEO/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    use_cuda = 0
    # if torch.cuda.is_available():
    #     use_cuda = 1

    files = os.listdir(input_path)
    model = load_model_new(use_cuda=use_cuda, model_name='q_function')  #
    # print(model)

    # EVAN: want input examples instead of pictures in imgs directory
    # EVAN: code was indented to this comment
# for imgname in files:
    # if imgname.endswith('jpg'):
    # if imgname.endswith('JPEG'):
        # input_img = input_path + imgname
        # print('imgname:', imgname)

        # EVAN: indeneted to this comment
        # EVAN: input_img -- needs to be vector
        # EVAN: img_label -- not sure what to put this as keeping as -1
    data = torch.load('data/test_dataset.pt')

    data = np.asarray(data, dtype=np.float32)
    data = Variable(torch.from_numpy(data))
    # print(data[0][0], data[0][1])
    print(data[0][0])
    print(type(data[0][0][0]))

    prediction = model(data[0][0])
    print(prediction)
    # to have blurred image just randomly select input at that column from dataset and create janky one that pulls from entire
    # data set. This will create an input that spans all over the input space being impossible to predict i.e. "blurry"
    #*********************************************
    input_img = data[0][0]
    img_label = -1

    # EVAN: This is generating blurred baseline a random array of 68 between 0 and 5 as of now
    img, blurred_img, logitori = Get_blurred_img(input_img, img_label, model, resize_shape=(224, 224),
                                                    Gaussian_param=[51, 50],
                                                    Median_param=11, blur_type='q_function', use_cuda=use_cuda)

    # EVAN: This is giving the mask back
    mask, upsampled_mask, imgratio, curvetop, curve1, curve2, category = Integrated_Mask(img, blurred_img, model,
                                                                                                img_label,
                                                                                                max_iterations=15,
                                                                                                integ_iter=20,
                                                                                                tv_beta=2,
                                                                                                l1_coeff=0.01 * 100,
                                                                                                tv_coeff=0.2 * 100,
                                                                                                size_init=28,
                                                                                                use_cuda=1)  #




    # outvideo_path = output_path + imgname[:-5] + '/'
    # if not os.path.isdir(outvideo_path):
    #     os.makedirs(outvideo_path)

    # output_file = output_path + imgname[:-4] + '_IGOS_'
    # save_heatmap(output_file, upsampled_mask, img * 255, blurred_img, blur_mask=0)

    # #scio.savemat(outvideo_path + imgname[:-5] + 'Mask' + '.mat',
    # #             mdict={'mask': mask},
    # #             oned_as='column')


    # output_file = outvideo_path + imgname[:-5] + '_IGOS_'
    # # EVAN: This is doing the metric of how well the heat map did
    # del_img, ins_img, delloss_top2, insloss_top2, del_ratio, ins_ratio, outmax, cateout, xnum = Deletion_Insertion(mask,
    #                                                                                                                 model,
    #                                                                                                                 output_file,
    #                                                                                                                 img,
    #                                                                                                                 blurred_img,
    #                                                                                                                 logitori,
    #                                                                                                                 category=-1,
    #                                                                                                                 pixelnum=200,
    #                                                                                                                 use_cuda=1,
    #                                                                                                                 blur_mask=0,
    #                                                                                                                 outputfig=1)

    # video_name = outvideo_path + 'AllVideo_fps10' + imgname[:-5] + '.avi'
    # write_video(output_file, video_name, xnum, fps=3)




