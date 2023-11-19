import torch
import numpy as np
from result_processing import confusion_matrix, write_to_file, calculate_accuracy, plot_cm, plot_accuracy, plot_loss
import math

classes=['background', 'farmland', 'garden', 'woodland', 'grass', 'water', 'road', 'building']
palette=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7]]

# ---------------------------------------------Train history-------------------------------------------------------
def train_history(device, epochs, classNum, dataLoaderTrain, dataLoaderTest, net, lossFn, optimizer, scheduler, file_path):
 model_name = net.__class__.__name__
 net = net.to(device)
 lossFn = lossFn.to(device)
 
# Metric initialisation for train
 train_acc_list = []
 test_acc_list = []
 train_loss_list = []
 test_loss_list = []
 conf_matrix_train = torch.zeros(classNum, classNum)
 conf_matrix_test = torch.zeros(classNum, classNum)
 
# train start
 print("---train start---")
 for epoch in range(0,epochs):
     train_totalloss = 0
     train_accuracy = 0
     matrix_temp_train = torch.zeros(classNum, classNum)

     for batch_idx, (img,label) in enumerate(dataLoaderTrain):        
          img = img.float().to(device)
          label = label.long().to(device)
          out = net(img) # model output 
          #seg, bound, dis, color = net(img)
         
          _, pred = torch.max(out, 1) # class prediction   
          overall_accuracy = calculate_accuracy(pred, label)
          matrix_temp_train = confusion_matrix(pred, label, matrix_temp_train) # calculate temporary confusion matrix 
          print("Training Epoch {}, Batch {}: Overall Accuracy: {:.4f}".format(epoch+1, batch_idx+1, overall_accuracy))
          loss = lossFn(out,label)
          #loss_seg = lossFn(seg, label)
          #loss_bound = lossFn(bound, label)
          #loss_dis = lossFn(dis, label)
          #loss_color = lossFn(color, label)
          #total_loss = loss_seg + loss_bound + loss_dis + loss_color
          assert not math.isnan(loss.item()) # gradient vanishing or explosion   
          loss.backward()
          #assert not math.isnan(total_loss.item())   
          #total_loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          train_totalloss = train_totalloss + loss
          #train_totalloss = train_totalloss + total_loss
          train_accuracy = train_accuracy + overall_accuracy
          
     scheduler.step() # scheduler update every epoch
     train_accuracy  /= len(dataLoaderTrain)
     train_acc_list.append(train_accuracy)
     train_loss_list.append(float(train_totalloss))
     print("Training accuracy of epoch {} is {}".format(epoch+1, train_accuracy))
     print("Training loss of epoch {} is {}\n".format(epoch+1, train_totalloss))

     #Record the confusion matrix of the last iteration
     if epoch == epochs - 1:
         conf_matrix_train = matrix_temp_train
         
     # Metric initialisation for test
     test_totalloss = 0
     test_accuracy = 0
     matrix_temp_test = torch.zeros(classNum, classNum)
     ious = 0
     mIoU = 0
     
     with torch.no_grad():
      for batch_idx, (img,label) in enumerate(dataLoaderTest):        
          img = img.float().to(device)
          label = label.long().to(device)
          out = net(img)
          #seg, bound, dis, color = net(img)
    
          _, pred = torch.max(out, 1)   
          overall_accuracy = calculate_accuracy(pred, label)
          print("Testing Epoch {}, Batch {}: Overall Accuracy: {:.4f}".format(epoch+1, batch_idx+1, overall_accuracy))
    
          matrix_temp_test = confusion_matrix(pred, label, matrix_temp_test)
          #conf_matrix = confusion_matrix(out, label, conf_matrix)

          loss = lossFn(out,label)
          #loss_seg = lossFn(seg, label)
          #loss_bound = lossFn(bound, label)
          #loss_dis = lossFn(dis, label)
          #loss_color = lossFn(color, label)
          #total_loss = loss_seg + loss_bound + loss_dis + loss_color
          test_totalloss = test_totalloss + loss
          #test_totalloss = test_totalloss + total_loss
          test_accuracy  = test_accuracy + overall_accuracy
          
     # Metric calculation for test
     matrix_temp_test = matrix_temp_test.cpu()
     test_accuracy  = test_accuracy / len(dataLoaderTest)
     test_acc_list.append(test_accuracy)       
     test_loss_list.append(float(test_totalloss)) 
     ious = np.diag(matrix_temp_test) / (matrix_temp_test.sum(axis=1) + matrix_temp_test.sum(axis=0) - np.diag(matrix_temp_test))
     mIoU = np.nanmean(ious)
     print("Testing accuracy of epoch {} is {}".format(epoch+1, float(test_accuracy)))
     print("Testing loss of epoch {} is {}".format(epoch+1, float(test_totalloss)))
     print("mIoU of epoch {} is {}\n".format(epoch+1, float(mIoU)))
     for i, iou in enumerate(ious):
         print("IoU for class", i, ":", iou.item())
     print("")
     
     # Record metrics every ten iterations
     if epoch != 0 and epoch % 10 == 9:
         write_to_file(file_path + "history.txt", epoch, train_accuracy, train_totalloss, test_accuracy, test_totalloss, ious, mIoU)
     
    # Graph and save the metrics every fifty iterations
    # also draw confusion matrices
     if epoch != 0 and epoch % 50 == 49:
         figa = plot_cm(matrix_temp_train, classNum, classes, epoch+1, file_path, is_train = True)
         figb = plot_cm(matrix_temp_test, classNum, classes, epoch+1, file_path, is_train = False)
         figc = plot_accuracy(train_acc_list, test_acc_list, epoch+1, file_path)
         figd = plot_loss(train_loss_list, test_loss_list, epoch+1, file_path)
         torch.save(net, file_path + f"{model_name}_" + str(epoch+1) + ".pth")
    
    #Record the confusion matrix of the last iteration
     if epoch == epochs - 1:
         conf_matrix_test = matrix_temp_test
     
 print("---train over---")
 return conf_matrix_train, conf_matrix_test, train_acc_list, test_acc_list, train_loss_list, test_loss_list