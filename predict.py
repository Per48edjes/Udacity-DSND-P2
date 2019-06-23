'''
This script uses a trained model for inference with preprocessing help from load.py.
'''


## IMPORT DEPENDENCIES
import load
import torch
import argparse
import json


## COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(description="Training a transfer learning neural net")
parser.add_argument('img_path', action='store', help='Set location of image to be predicted')
parser.add_argument('checkpoint', action='store', help='Set location of model checkpoint')
parser.add_argument('--category_names', action='store', help='Set location of JSON class name mappings')
parser.add_argument('--top_k', action='store', help='Set number of classes to show predictions for', type=int, default=5)
parser.add_argument('--gpu', action='store_true', default=False, help='Toggle GPU')
results = parser.parse_args()


## SET VARIABLES AND DEFAULTS
img_path = results.img_path
checkpoint = results.checkpoint
mapping_path = results.category_names
top_k = results.top_k 
gpu = results.gpu


## PREDICTION FUNCTION
def predict(image_path, model_path, topk):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = load.checkpoint_loader(model_path, gpu)
    image_tensor = torch.from_numpy(load.process_image(image_path))
    
    # Move image_tensor and model to correct device for inference
    device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    
    inverted_class_to_idx = {v: k for k, v in model.class_to_idx.items()}

    ## Do the prediction
    model.eval()
    with torch.no_grad():
        logps = model.forward(image_tensor.unsqueeze_(0).float())
        ps = torch.exp(logps)
        top_k_p, top_k_classes = ps.topk(topk, dim=1)
        
    top_k_idx = [inverted_class_to_idx[top_k_classes[0,x].item()] for x in range(topk)]
    top_k_probs = [top_k_p[0,x].item() for x in range(topk)]
    
    return top_k_probs, top_k_idx


## PREDICTION
probs, classes = predict(img_path, checkpoint, top_k)


## OUTPUT
try: 
    with open(mapping_path, 'r') as f:
        cat_to_name = json.load(f)
        labels = [cat_to_name[x] for x in classes]
except:
    labels = classes
        
finally: 
    print(f"\nPrediction: Top {top_k} Classes")
    for i, (label, prob) in enumerate(zip(labels, probs), start=1):
        print(f"{i}: {label} (p = {prob:.5f})")
