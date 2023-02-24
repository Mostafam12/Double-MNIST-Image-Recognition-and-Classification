import numpy as np
import torch
print(torch.backends.mps.is_built())

from PIL import Image
train_set  = np.load("D:\cmput 328/a8/train_X.npy")
# train_labels_set = np.load("D:\cmput 328/a7/train_Y.npy")

valid_set  = np.load("D:\cmput 328/a8/valid_X.npy")
#valid_labels_set = np.load("D:\cmput 328/a7/valid_Y.npy")


#train_bboxes_set = np.load("D:\cmput 328/a7/train_bboxes.npy")
#valid_bboxes_set = np.load("D:\cmput 328/a7/valid_bboxes.npy")





n_images = train_set.shape[0]
wh= 28/64
val = " "+str(wh)


for img_id in range(n_images):
    src_img = train_set[img_id, ...].squeeze().reshape((64, 64, 3)).astype(np.uint8)
    im = Image.fromarray(src_img)
    f_name= 'D:\cmput 328/a8/train/img_'+str(img_id)+ ".jpg"
    im.save(f_name)


# for i in range(n_images):
#     train_bboxes = train_bboxes_set[i].tolist()
#     train_labels = train_labels_set[i].tolist()
#     x_cor0 = ((train_bboxes[0][1]/64)+(train_bboxes[0][3]/64))/2
#     y_cor0 = ((train_bboxes[0][0]/64)+(train_bboxes[0][2])/64)/2
#     x_cor1 = ((train_bboxes[1][1]/64)+(train_bboxes[1][3]/64))/2
#     y_cor1 = ((train_bboxes[1][0]/64)+(train_bboxes[1][2])/64)/2
#     line1=str(train_labels[0]) +" "+str(x_cor0)+" "+str(y_cor0) + val+val
#     line2=str(train_labels[1]) +" "+str(x_cor1)+" "+str(y_cor1) + val+val   
#     print(line1)
#     print(line2)
#     file_name = 'D:\cmput 328/a7/training/'+str(i)+"_train.txt"
#     f = open(file_name, 'w')
#     f.write(line1+"\n"+line2)
#     f.close()

   
n_images = valid_set.shape[0]
wh= 28/64
val = " "+str(wh)

for img_id in range(n_images):
    src_img = valid_set[img_id, ...].squeeze().reshape((64, 64, 3)).astype(np.uint8)
    im = Image.fromarray(src_img)
    f_name= 'D:\cmput 328/a8/valid/img_'+str(img_id)+ ".jpg"
    im.save(f_name)

# for i in range(n_images):
#     train_bboxes = valid_bboxes_set[i].tolist()
#     train_labels = valid_labels_set[i].tolist()
#     x_cor0 = ((train_bboxes[0][1]/64)+(train_bboxes[0][3]/64))/2
#     y_cor0 = ((train_bboxes[0][0]/64)+(train_bboxes[0][2])/64)/2
#     x_cor1 = ((train_bboxes[1][1]/64)+(train_bboxes[1][3]/64))/2
#     y_cor1 = ((train_bboxes[1][0]/64)+(train_bboxes[1][2])/64)/2
#     line1=str(train_labels[0]) +" "+str(x_cor0)+" "+str(y_cor0) + val+val
#     line2=str(train_labels[1]) +" "+str(x_cor1)+" "+str(y_cor1) + val+val   
#     print(line1)
#     print(line2)
#     file_name = 'D:\cmput 328/a7/validation/'+str(i)+"_train.txt"
#     f = open(file_name, 'w')
#     f.write(line1+"\n"+line2)
#     f.close()


