import numpy as np
import torch
import tensorflow as tf



def classify_and_detect(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # add your code here to fill in pred_class and pred_bboxes
    
    path = "5s_12b_30n.pt"
    model = torch.hub.load("ultralytics/yolov5", 'custom', path="5s_12b_30n.pt")
    model.eval()
    unflatten = images.reshape(-1, 64, 64, 3)
    counter = 1
    with torch.no_grad():
        for i in range(N):
            print(str(counter) + '/5000')
            counter += 1

            outputs = model(unflatten[i, ...])
            outputArr = outputs.pred[0]
            #print(outputArr.size())
            if outputArr.size() >= torch.Size([2,6]) or outputArr.size() == torch.Size([3,6]):
                output1 = outputArr[0,:]
                output2 = outputArr[1,:]
                if output1[-1].item() >= output2[-1].item():
                    pred_class[i,0] = output2[-1].cpu().numpy()
                    pred_class[i,1] = output1[-1].cpu().numpy()

                    pred_bboxes[i,0,0] = np.rint(output2[1].cpu().numpy())
                    pred_bboxes[i,0,1] = np.rint(output2[0].cpu().numpy())
                    pred_bboxes[i,0,2] = np.rint(output2[3].cpu().numpy())
                    pred_bboxes[i,0,3] = np.rint(output2[2].cpu().numpy())

                    pred_bboxes[i,1,0] = np.rint(output1[1].cpu().numpy())
                    pred_bboxes[i,1,1] = np.rint(output1[0].cpu().numpy())
                    pred_bboxes[i,1,2] = np.rint(output1[3].cpu().numpy())
                    pred_bboxes[i,1,3] = np.rint(output1[2].cpu().numpy())

                else:
                    pred_class[i,0] = output1[-1].cpu().numpy()
                    pred_class[i,1] = output2[-1].cpu().numpy()

                    pred_bboxes[i,0,0] = np.rint(output1[1].cpu().numpy())
                    pred_bboxes[i,0,1] = np.rint(output1[0].cpu().numpy())
                    pred_bboxes[i,0,2] = np.rint(output1[3].cpu().numpy())
                    pred_bboxes[i,0,3] = np.rint(output1[2].cpu().numpy())

                    pred_bboxes[i,1,0] = np.rint(output2[1].cpu().numpy())
                    pred_bboxes[i,1,1] = np.rint(output2[0].cpu().numpy())
                    pred_bboxes[i,1,2] = np.rint(output2[3].cpu().numpy())
                    pred_bboxes[i,1,3] = np.rint(output2[2].cpu().numpy())

            elif outputArr.size() == torch.Size([1,6]):
                output1 = outputArr[0]
                output1bbox = output1[0:4]
                pred_class[i,0] = output1[-1].cpu().numpy()
                pred_class[i,1] = output1[-1].cpu().numpy()

                pred_bboxes[i,0,0] = np.rint(output1[1].cpu().numpy())
                pred_bboxes[i,0,1] = np.rint(output1[0].cpu().numpy())
                pred_bboxes[i,0,2] = np.rint(output1[3].cpu().numpy())
                pred_bboxes[i,0,3] = np.rint(output1[2].cpu().numpy())

                pred_bboxes[i,1,0] = np.rint(output1[1].cpu().numpy())
                pred_bboxes[i,1,1] = np.rint(output1[0].cpu().numpy())
                pred_bboxes[i,1,2] = np.rint(output1[3].cpu().numpy())
                pred_bboxes[i,1,3] = np.rint(output1[2].cpu().numpy())
            
            
            else:
                pred_class[i,0] = torch.tensor([0])
                pred_class[i,1] = torch.tensor([0])
                pred_bboxes[i,0,:] = torch.tensor([0,0,0,0])
                pred_bboxes[i,1,:] = torch.tensor([0,0,0,0])

            



    return pred_class, pred_bboxes
