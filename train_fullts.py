### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model_fullts
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp

def main():
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model_fullts(opt)
    model.cuda()
    visualizer = Visualizer(opt)

    total_steps = (start_epoch-1) * dataset_size + epoch_iter    
    
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter  % dataset_size
        for data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            save_fake = total_steps % opt.display_freq == 0

            ############## Forward Pass ######################
            no_nexts = data['next_label'].dim() > 1

            if no_nexts:
                cond_zeros = torch.zeros(data['label'].clone().size()).float()

                label = Variable(data['label']).to(device)
                label = label.type(torch.cuda.FloatTensor)
                next_label = Variable(data['label']).to(device)
                next_label = next_label.type(torch.cuda.FloatTensor)
                image = Variable(data['image']).to(device)
                image = image.type(torch.cuda.FloatTensor)
                next_image = Variable(data['image']).to(device)
                next_image = next_image.type(torch.cuda.FloatTensor)
                face_coords = Variable(data['face_coords']).to(device)
                face_coords = face_coords.type(torch.cuda.FloatTensor)
                cond_zeros = Variable(cond_zeros).to(device)
                cond_zeros = cond_zeros.type(torch.cuda.FloatTensor)
                
                # label = Variable(data['label'])
                # next_label = Variable(data['label'])
                # image = Variable(data['image'])
                # next_image = Variable(data['image'])
                # face_coords = Variable(data['face_coords'])
                # cond_zeros = Variable(cond_zeros)
            
                model.cuda()

                losses, generated = model(label, next_label, image, next_image, face_coords, cond_zeros, infer=True)

                losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                loss_dict = dict(zip(model.loss_names, losses))

                loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + (loss_dict['D_realface'] + loss_dict['D_fakeface']) * 0.5
                loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_GANface']

                ############### Backward Pass ####################
                model.optimizer_G.zero_grad()
                loss_G.backward()
                model.optimizer_G.step()

                model.optimizer_D.zero_grad()
                loss_D.backward()
                model.optimizer_D.step()

                ############## Display results and errors ##########
                if total_steps % opt.print_freq == 0:
                    errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                    visualizer.plot_current_errors(errors, total_steps)

                if save_fake:
                    syn = generated[0].data[0]
                    inputs = torch.cat((data['label'], data['next_label']), dim=3).to(device)
                    targets = torch.cat((data['image'], data['next_image']), dim=3).to(device)
                    visuals = OrderedDict([('input_label', util.tensor2im(inputs[0].to(device), normalize=False)),
                                           ('synthesized_image', util.tensor2im(syn.to(device))),
                                           ('real_image', util.tensor2im(targets[0].to(device)))])
                    if opt.face_generator:
                        miny, maxy, minx, maxx = data['face_coords'][0]
                        res_face = generated[2].data[0]
                        syn_face = generated[1].data[0]
                        preres = generated[3].data[0]
                        visuals = OrderedDict([('input_label', util.tensor2im(inputs[0].to(device), normalize=False)),
                                               ('synthesized_image', util.tensor2im(syn.to(device))),
                                               ('synthesized_face', util.tensor2im(syn_face.to(device))),
                                               ('residual', util.tensor2im(res_face.to(device))),
                                               ('real_face', util.tensor2im(data['image'][0][:, miny:maxy, minx:maxx].to(device))),
                                               ('input_face', util.tensor2im(data['label'][0][:, miny:maxy, minx:maxx].to(device), normalize=False)),
                                               ('real_image', util.tensor2im(targets[0].to(device)))])
                    visualizer.display_current_results(visuals, epoch, total_steps)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')            
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            print('------------- finetuning Local + Global generators jointly -------------')
            model.update_fixed_params()

        if (opt.niter_fix_main != 0) and (epoch == opt.niter_fix_main):
            print('------------- training all the discriminators now and not just the face -------------')
            model.update_fixed_params_netD()

        if epoch > opt.niter:
            model.update_learning_rate()

if __name__ == '__main__':
    mp.freeze_support()
    main()
