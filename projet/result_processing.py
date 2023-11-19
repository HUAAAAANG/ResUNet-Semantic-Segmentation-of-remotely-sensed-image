import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

# calculate the confusion matrix
def confusion_matrix(preds, labels, conf_matrix):
    #preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
      conf_matrix[p, t] += 1
    return conf_matrix

# calculate metrics, i.e. tp, fp, fn, accuracy, recall, precision
def calculate_metrics(conf_matrix):
    num_classes = conf_matrix.shape[0]
    metrics = {}
    
    for class_idx in range(num_classes):
        true_positives = conf_matrix[class_idx, class_idx]
        false_positives = np.sum(conf_matrix[:, class_idx]) - true_positives
        false_negatives = np.sum(conf_matrix[class_idx, :]) - true_positives
        
        accuracy = true_positives / np.sum(conf_matrix[class_idx, :])
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        
        metrics[f'Class_{class_idx}'] = {'Accuracy': accuracy,
                                         'Precision': precision,
                                         'Recall': recall}
    return metrics

# calculate the overall accuracy from the correct predicted pixels
def calculate_accuracy(predictions, labels):
    correct_pixels = torch.eq(predictions, labels).float() 
    #class_accuracy = torch.mean(correct_pixels, dim=(0, 1)).cpu().numpy()
    overall_accuracy = torch.mean(correct_pixels).item()
    return overall_accuracy

# print out the confusion matrix and its corresponding indicators
def print_matrix(conf_matrix):
    conf_matrix = np.array(conf_matrix.cpu())
    corrects = conf_matrix.diagonal(offset = 0)
    nb_class = conf_matrix.sum(axis = 1)
    print("Train number:{0}".format(int(np.sum(conf_matrix))))
    print(conf_matrix)
    print("Number per class:",nb_class)
    print("Correct prediction:",corrects)
    print("Accuracy(%):{0}".format([rate*100 for rate in corrects/nb_class]))

    print(calculate_metrics(conf_matrix))

# Draw the confusion matrix and save it for train dataset and test dataset
def plot_cm(conf_matrix, classNum, classes, epoch, path, is_train):
    plt.figure()
    plt.imshow(conf_matrix, cmap=plt.cm.Blues, aspect='auto')
    thresh = conf_matrix.max()/2
    for x in range(classNum):
       for y in range(classNum):
         info = int(conf_matrix[y,x])
         plt.text(x, y, info, verticalalignment='center', horizontalalignment='center', color="white" if info > thresh else "black")
    
    plt.yticks(range(classNum), classes)
    plt.xticks(range(classNum), classes, rotation=45)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if is_train == True:
        plt.savefig(path + "cm_train_" + str(epoch) + ".jpg")
    else:
        plt.savefig(path + "cm_test_" + str(epoch) + ".jpg")
        
# Plot the accuracy evolution curve according to epoch for train and test
def plot_accuracy(train_list, test_list, epoch, path):
    plt.figure()
    epochs = len(train_list)
    plt.plot(range(epochs), train_list, label='Train Accuracy')
    plt.plot(range(epochs), test_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    
    train_max_acc = max(train_list)
    train_max_epoch = train_list.index(train_max_acc) + 1
    test_max_acc = max(test_list)
    test_max_epoch = test_list.index(test_max_acc) + 1

    plt.annotate(f'Train Max: {train_max_acc:.4f} (Epoch {train_max_epoch})', 
             xy=(train_max_epoch, train_max_acc),
             xytext=(10, -20),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->'))

    plt.annotate(f'Test Max: {test_max_acc:.4f} (Epoch {test_max_epoch})', 
             xy=(test_max_epoch, test_max_acc),
             xytext=(10, -40),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->'))
             
    plt.savefig(path + "acc_" + str(epoch) + ".jpg")

# Plot the loss evolution curve according to epoch for train and test
def plot_loss(loss_list_train, loss_list_test, epoch, path):
    plt.figure()
    epochs = len(loss_list_train)
    loss_list_train = np.array(loss_list_train)
    loss_list_test = np.array(loss_list_test)
    plt.plot(range(epochs), loss_list_train / 4, label='Train Loss')
    plt.plot(range(epochs), loss_list_test, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and test loss')
    plt.legend()
    plt.savefig(path + "loss_" + str(epoch) + ".jpg") 

# Record all metrics in a text file
def write_to_file(file_path, epoch, train_accuracy, train_totalloss, test_accuracy, test_totalloss, ious, mIoU):
    with open(file_path, 'a') as file:
       file.write("Training accuracy of epoch {} is {}\n".format(epoch+1, float(train_accuracy)))
       file.write("Training loss of epoch {} is {}\n".format(epoch+1, float(train_totalloss)))
       file.write("Test accuracy of epoch {} is {}\n".format(epoch+1, float(test_accuracy)))
       file.write("Test loss of epoch {} is {}\n".format(epoch+1, float(test_totalloss)))
       file.write("IoU of epoch {} is {}\n".format(epoch+1, np.array(ious)))
       file.write("mIoU of epoch {} is {}\n \n".format(epoch+1, float(mIoU)))

# The information about the images used to make the predictions is also recorded in this text file
def write_to_file_pred(file_path, files_images):
    with open(file_path, 'a') as file:
        for image in files_images:
            file.write("Image for prediction:" + str(image) + "\n\n")
