__author__ = "David Taschjian"

import click
import hashlib
import os
import json
import time
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine
from edgetpu.detection.engine import DetectionEngine
import cv2
import numpy as np
from PIL import Image

PROPERTY_FILE_PATH = './properties.json'

def hash(string, hashtype):
   """
    This function returns a defined hash for a string 
    
    :param string: a string to hash
    :type filename: string
    :param hashtype: The hashing function which should be used. It has to be defined in hashlib
    :type hashtype: String
    :return type: String representation of a hex number
   """
   # hash binary representation of string and convert it to hex
   return eval('hashlib.{}(string.encode())'.format(hashtype)).hexdigest()

def load_properties():
    """
    This function is loading the data from the property file and checking if anything is missing.
    
    :return type: dictonary
    """
    with open(PROPERTY_FILE_PATH) as f:
        properties = json.load(f)
        
    required_entries = set({'user','classification','detection'})
    
    missing_entries = required_entries - properties.keys()
    if not len(missing_entries) == 0:
        raise Exception('Propeties corrupted! {} missing.'.format(str(missing_entries)))

    return properties

def save_properties(data):
    """
    Saves data to property file
    """
    with open(PROPERTY_FILE_PATH, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)    

def user_exist(username):
    """
        Checks if a user with the given username exists.

        :param username: the name of a user
        :type username: string
        :return type: boolean
    """
    #loading all user
    user = load_properties()['user'] 
    return username in user.keys()

class Username(click.ParamType):
    """
        Custom Type class for click for usernames, to check for new usernames or for current usernames.

        :param new: True if class shoud check whether a new username is already taken
        :type new: boolean
    """
    def __init__(self, new=True):
        self.new = new

    def convert(self, value, param, ctx):
        """
        This is a function used by click.

        :param value: the username in interest
        :type value : string
        :return type: string
        """
        if not type(value) == str:
            self.fail('expected string for username, got'f"{value!r} of type {type(value.__name__)}",param,ctx)
        if self.new:
            if user_exist(value):
                self.fail(f"{value!r} is already taken!", param, ctx)
            else:
                return value
        else:
            if user_exist(value):
                return value
            else:
                self.fail(f"{value!r} does not exist!", param, ctx)
            return value

def process_user_pictures(props,source,destination):
    click.echo('Fetching files...')
    image_list = [f for f in os.listdir(source)
            if os.path.isfile(os.path.join(source, f))]
    detection = DetectionEngine(props['detection'])
    click.echo('Looking for faces..')

    found_faces = 0 
    for filename in image_list:
        cv2_im = cv2.imread(os.path.join(source,filename))
        pil_im = Image.fromarray(cv2_im)
        #searching for faces in the picture 
        faces = detection.detect_with_image(pil_im,threshold = 0.1, top_k =3,keep_aspect_ratio=True,relative_coord=True)
        click.echo('Found {} faces in {}'.format(len(faces),filename))
        #check each face and append results to frame
        height, width, _= cv2_im.shape
        cnt = 0
        for face in faces:
            x0, y0, x1, y1 = face.bounding_box.flatten().tolist()
            x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
            #crop out face from camera picture
            face_im = cv2_im[y0:y1,x0:x1]
            #save image
            new_filename = filename[:filename.rfind('.')] + '_' + str(cnt) + '_' +filename[filename.rfind('.'):] 
            click.echo('Saving {}'.format(new_filename))
            cv2.imwrite(os.path.join(destination,new_filename), face_im)
            cnt += 1
        found_faces += cnt
    click.echo(click.style("Done! From {} images we could find {} faces ".format(len(image_list), found_faces), fg='green'))
 
def retrain_model(props):
    """
        This function is using the Imprinting technique to retrain the model by only changing the last layer.
        All classes will be abandoned while training multiple users
    """
    MODEL_PATH = props['classification']['default_path']

    click.echo('Parsing data for retraining...')
    train_set = {}
    test_set = {}
    for user in props['user'].keys():
        image_dir = props['user'][user]['images']
        images = [f for f in os.listdir(image_dir)
                if os.path.isfile(os.path.join(image_dir, f))]
        if images:
            # allocate the number of images for training an validation
            net_pictures =len(images)
            click.echo(click.style('We found {} pictures for {}'.format(net_pictures,user),fg='green'))
            while True: 
                k = int(click.prompt('How many pictures do you want for validating the training?'))
                if k > 0.25*net_pictures: 
                    click.echo(click.style('At most 25% ({} pictures) of the training data can be used for testing the model!'.format(int(0.25*net_pictures)), fg='yellow'))
                elif k <2:
                    click.echo(click.style('At least 3 pictues must be used for testing the model!', fg='yellow'))
                else:
                    break
            
            test_set[user] = images[:k]
            assert test_set, 'No images to test [{}]'.format(user)
            train_set[user] = images[k:]
            assert train_set, 'No images to train [{}]'.format(user)
        
    #get shape of model to retrain
    tmp = BasicEngine(MODEL_PATH)
    input_tensor = tmp.get_input_tensor_shape()
    shape = (input_tensor[2], input_tensor[1])

    #rezising pictures and creating new labels map
    train_input = []
    labels_map = {}
    for user_id, (user, image_list) in enumerate(train_set.items()):
        ret = []
        for filename in image_list:
            with Image.open(os.path.join(props['user'][user]['images'],filename)) as img:
                img = img.convert('RGB')
                img = img.resize(shape, Image.NEAREST)
                ret.append(np.asarray(img).flatten())
        train_input.append(np.array(ret))
        labels_map[user_id] = user
    #Train model
    click.echo('Start training')
    engine = ImprintingEngine(MODEL_PATH, keep_classes=False)
    engine.train_all(train_input)
    click.echo(click.style('Training finished!', fg='green'))
        
    #gethering old model files
    old_model = props['classification']['path']
    old_labels = props['classification']['labels']
    #saving new model
    props['classification']['path'] = os.getcwd() + '/Models/model{}.tflite'.format(''.join(['_' + u for u in labels_map.values()]))
    engine.save_model(props['classification']['path'])
    #saving labels
    props['classification']['labels'] = props['classification']['path'].replace('classification','labels').replace('tflite','json')
    with open(props['classification']['labels'] , 'w') as f:
        json.dump(labels_map, f, indent=4)
    #Evaluating how well the retrained model performed
    click.echo('Start evaluation')
    engine = ClassificationEngine(props['classification']['path'])
    top_k = 5
    correct = [0] * top_k
    wrong = [0] * top_k
    for user, image_list in test_set.items():
        for img_name in image_list:
            img = Image.open(os.path.join(props['user'][user]['images'],img_name))
            candidates = engine.classify_with_image(img, threshold=0.1, top_k=top_k)
            recognized = False
            for i in range(top_k):
                if i < len(candidates) and  user == labels_map[candidates[i][0]]:
                    recognized = True
                if recognized:
                    correct[i] = correct[i] + 1
                else:
                    wrong[i] = wrong[i] + 1
        click.echo('Evaluation Results:')
        for i in range(top_k):
            click.echo('Top {} : {:.0%}'.format(i+1, correct[i] / (correct[i] + wrong[i])))
        #  TODO  highlight with colors how well it perforemed

    if  not old_model == props['classification']['path'] and not old_labels == props['classification']['labels'] and (os.path.exists(old_labels) or os.path.exists(old_model)):
        if not click.confirm('Do you want to keep old models?'):
            os.remove(old_model)
            os.remove(old_labels)
            click.echo(click.style('Old models removed.', fg='green'))
    #saving properties
    save_properties(props)

def facial_recogntion(props):
    cap = cv2.VideoCapture(0)

    #get labels
    with open(props['classification']['labels']) as f:
        labels_map = json.load(f)

    #loading detection model
    detection = DetectionEngine(props['detection'])
    classification = ClassificationEngine(props['classification']['path'])
    while cap.isOpened():
        ret, cv2_im = cap.read()
        if not ret:
            break
        pil_im = Image.fromarray(cv2_im)
        #searching for faces in the picture 
        faces = detection.detect_with_image(pil_im,threshold = 0.5, top_k =3,keep_aspect_ratio=True,relative_coord=True)
        #check each face and append results to frame
        height, width, _= cv2_im.shape
        for face in faces:
            x0, y0, x1, y1 = face.bounding_box.flatten().tolist()
            x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
            #crop out face from camera picture
            face_im = Image.fromarray(cv2_im[y0:y1,x0:x1])
            #classify face
            results = classification.classify_with_image(face_im,threshold=0.1,top_k=3)   
            #annotate frame
            good_recognition = False
            text_lines = []
            for index, score in results:
                if score >= 0.7:
                    good_recognition = True
                text_lines.append('score=%.2f: %s' % (score, labels_map[str(index)]))
            
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0) if good_recognition else (255,0,0), 2)
            for y, line in enumerate(text_lines):
                cv2.putText(cv2_im,line,(x0,y0 + y*20+20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale =0.5, color=(255, 255, 255)) 
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class SmartLock(object):
    #TODO write class for the lock 
    def __init__(self):
        pass