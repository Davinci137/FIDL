__author__ = "David Taschjian"

import click
import hashlib
import os
import json
import time
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine
import cv2
import numpy as np
from PIL import Image

PROPERTY_FILE_PATH = './properties.json'
MODEL_PATH = './Models/mobilenet_with_user.tflite'
LABEL_PATH  = './Models/label_map.json' 

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
        
    required_entries = set({'user'})
    
    missing_entries = required_entries - properties.keys()
    if not len(missing_entries) == 0:
        click.echo(click.style('{} missing in properites'.format(str(missing_entries)), fg= 'red'))
        click.echo('Adding missing properties')
        properties = {**properties, **{entry:{} for entry in missing_entries}}
        #save to file
        save_properties(properties)

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
  """
  def __init__(self, new=True):
    self.new = new

  def convert(self, value, param, ctx):
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
 

def retrain_model(newClass, newClass_dir):
  """
    This function is using the Imprinting technique to retrain the model by only changing the last layer.
    Further, the original classifications will be kept and knew ones will be added.
  """
  ##IMPORTANT this is going thorgh a directroy and training multiple user, but we only want to train one user
  click.echo('Parsing data for retraining...')
  train_set = []
  test_set = []
  images = [f for f in os.listdir(newClass_dir)
          if os.path.isfile(os.path.join(newClass_dir, f)) and newClass_dir.endswith('jpg')]
  if images:
    #25% of the pictures will be used to test the retrained model
    k = max(int(0.25 * len(images)), 1)
    test_set = images[:k]
    assert test_set, 'No images to test [{}]'.format(newClass)
    train_set = images[k:]
    assert train_set, 'No images to train [{}]'.format(newClass)
  
  #get shape of model to retrain
  tmp = BasicEngine(MODEL_PATH)
  input_tensor = tmp.get_input_tensor_shape()
  shape = (input_tensor[2], input_tensor[1])

  #get current labels from the untrained model
  with open(LABEL_PATH) as f:
    label_map = json.load(f)

  #determining label for new class
  class_id = int(sorted(label_map.keys())[-1]) +1 

  click.echo('Processing {}'.format(newClass))
  click.echo('This may take longer...')

  #rezising pictures
  train_input = []
  for filename in train_set:
    with Image.open(os.path.join(newClass_dir, filename)) as img:
      img = img.convert('RGB')
      img = img.resize(shape, Image.NEAREST)
      train_input.append(np.asarray(img).flatten())
  train_input = np.array(train_input)


  #Train model
  click.echo('Start training')
  engine = ImprintingEngine(MODEL_PATH, keep_classes=True)
  engine.train(train_input, class_id)
  click.echo(click.style('Training finished!', fg='green'))

  engine.save_model(MODEL_PATH)
  
  #saving labels
  label_map[class_id] = newClass
  with open(LABEL_PATH , 'w') as f:
    json.dump(label_map, f, indent=4)

  #Evaluating how well the retrained model performed
  click.echo('Start evaluation')
  engine = ClassificationEngine(MODEL_PATH)
  top_k = 5
  correct = [0] * top_k
  wrong = [0] * top_k
  for img_name in test_set:
    img = Image.open(os.path.join(newClass_dir, img_name))
    candidates = engine.classify_with_image(img, threshold=0.1, top_k=top_k)
    recognized = False
    for i in range(top_k):
        if i < len(candidates) and newClass in label_map[candidates[i][0]]:
          recognized = True
        if recognized:
          correct[i] = correct[i] + 1
        else:
          wrong[i] = wrong[i] + 1
  click.echo('Evaluation Results:')
  for i in range(top_k):
    click.echo('Top {} : {:.0%}'.format(i+1, correct[i] / (correct[i] + wrong[i])))
  #  TODO  highlight with colors how well it perforemed

def run_classification():
  engine = ClassificationEngine(MODEL_PATH)

  #get labels
  with open(LABEL_PATH) as f:
    label_map = json.load(f)

  cap = cv2.VideoCapture(0)

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2_im = frame

    pil_im = Image.fromarray(cv2_im)

    start_time = time.monotonic()
    results = engine.classify_with_image(pil_im,threshold=0.1,top_k=3)   
    end_time = time.monotonic()
    text_lines = [
          'Inference: %.2f ms' %((end_time - start_time) * 1000),
    ]
    for index, score in results:
      text_lines.append('score=%.2f: %s' % (score, label_map[str(index)]))
      print(' '.join(text_lines))

    for y, line in enumerate(text_lines):
      cv2.putText(cv2_im,line,(11,y*20+20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale =0.5, color=(255, 255, 255))

    cv2.imshow('frame', cv2_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()
