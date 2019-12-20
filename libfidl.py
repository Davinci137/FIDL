__author__ = "David Taschjian"

import click
import hashlib
import os
import json
import time
import RPi.GPIO as GPIO
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine
from edgetpu.detection.engine import DetectionEngine
import cv2
import numpy as np
from PIL import Image
import threading

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
    props['classification']['path'] = './Models/model{}.tflite'.format(''.join(['_' + u for u in labels_map.values()]))
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

def facial_recogntion(props, smartLock):
    """
        Processing camera pictures with multiple AI's and executing background jobs.
        This is the main loop for the Door Lock.
        AI stages: Movement detection -> Face detection -> Face classification

        :param props: Necessery predefined properties 
        :type props: Dictonary
        :param smartLock: object used to access unlocking functions and door status
        :type smartLock: SmartLock object
    """
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
        # flip picture
        cv2_im = cv2.flip(cv2_im,-1)
        # Skip classification if door is open
        if not smartLock.is_door_closed():
            #shuffle all the pixels and make the picture unrecoginzable
            cv2_im = cv2.randShuffle(cv2_im)
            #TODO Indicate that door is open
            cv2.imshow('FIDL', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        pil_im = Image.fromarray(cv2_im)
        # TODO check first if the there is movement in the picture before utilizing edge tpu (power saving)
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
            best_result = (0,0) # index, score
            text_lines = []
            for index, score in results:
                if score > best_result[1]:
                    best_result = (index, score)
                text_lines.append('score=%.2f: %s' % (score, labels_map[str(index)])) #TODO decide to show all scores or only best score
            
            access_granted = props['user'][labels_map[str(best_result[0])]]['access'] and score >= 0.9 
            #open the door
            if access_granted and smartLock.unlocking == False:
                threading.Thread(target = smartLock.unlock, daemon = True).start()
            # TODO indicate unlocking
            #Coloring green if access was granted 
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0) if access_granted else (0,0,255), 2)
            for y, line in enumerate(text_lines):
                cv2.putText(cv2_im,line,(x0 + 10,y0 + y*20+20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale =0.5, color=(255, 255, 255)) 
        cv2.imshow('FIDL', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class SmartLock(object):
    """
    An object for easliy interacting the SmartLock which consists of an ultra sonic sensor, a relais, a door sensor and a servo motor.
    Functonality:
        Fetching distance from Ultra sonics sensor
            Approximating speed
        Changing state of relais
        Fetching status of the door (closed/open)
        Manually setting angle of servo motor
            unlocking the lock (predefined sequence of servo and relais statements)
        Automatic unlocking when approaching the door
    """
    def __init__(self, trig_pin = 18, echo_pin = 24, relais_pin = 23, door_sensor_pin = 25, servo_pin = 13, max_distance = 100):
        """
            :param trig_pin: The pin attached to the trigger pin of the ultra sonic sensor (BCM)
            :type trig_pin: int
            :param echo_pin: The pin attached to the echo pin of the ultra sonic sensor (BCM)
            :type echo_pin: int
            :param relais_pin: The pin attached to the relais (BCM)
            :type relais_pin: int
            :param door_sensor_pin: The pin attached to the door sensor (BCM)
            :type door_sensor_pin: int
            :param servo_pin: The pin attached to the servo motor (BCM)
            :type servo_pin: int
            :param max_distance: The maximum distance the ultrasonic sensor should consider when gethering the distance
            :type max_distance: int
        """
        self.trig_pin = trig_pin 
        self.echo_pin = echo_pin 
        self.relais_pin = relais_pin
        self.door = door_sensor_pin
        self.servo_pin = servo_pin
        self.max_distance = max_distance
        #initlizing the data from the ultra sonic sensor as None
        self.measurements = []
        self._reset_measurements()
        self.unlocking = False

    def _init_GPIO(self):
        """
            Intilizing all the GPIO setting
        """
        GPIO.setmode(GPIO.BCM)
        #setting up GPIO 
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.setup(self.relais_pin, GPIO.OUT)
        GPIO.setup(self.door,GPIO.IN, pull_up_down = GPIO.PUD_UP)
        GPIO.setup(self.servo_pin,GPIO.OUT)

        #Turn relais off
        GPIO.output(self.relais_pin, GPIO.LOW)

        #set servo frequency and start pulse 
        # you may edit the frequency depending on the frequency of your servo
        self.pulse = GPIO.PWM(self.servo_pin, 50)
        self.pulse.start(0)

    def is_door_closed(self):
        """
            Returning the status of the door by reading the door sensor
            return True if door closed
            :return type: boolean
        """
        return GPIO.input(self.door)

   ### Ultrosonic sensor and related algortihms ### 
    def _reset_measurements(self):
        """
            setting all measurements to None
        """
        self.measurements = [(None, None) for i in range(10)]

    def get_distance(self):
        """
            Fetching distance from ultrosonic sensor within the defined range
            returning None if distance is greater the predefinec max_measurment
            measurements in centimeter
            :return type: float or None

        """
        #Triggering the ultrasonic sensor to play sound.
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, False)
            
        #initilzing time variables
        StartTime = time.time()
        StopTime = time.time()
    
        #Fetching the start time until the ourtuut of the echo pin of the sensor changes from Low to high 
        while GPIO.input(self.echo_pin) == 0:
            StartTime = time.time()

        #Fetching end time 
        while GPIO.input(self.echo_pin) == 1:
            StopTime = time.time()
    
        # time difference between start and arrival
        TimeElapsed = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s) in air and take account double travel
        distance = round((TimeElapsed * 34300) / 2, 2)

        #ignoring all values which are not in range
        if distance > self.max_distance:
            return None
        return distance

    def approach_detected(self):
        """
            Function to determin if something is approaching the Sensor
            This use linear regression to approximate the approaching speed

            :return type: boolean
        """
        if any([None in pt for pt in self.measurements]): #TODO fix that somehow
            #Not enough critical data was gethered
            return False

        #using linear regression to determine an approach
        x = [pt[0] for pt in self.measurements]
        x_mean = round(sum(x) / len(x), 2)
        y = [pt[1] for pt in self.measurements]
        y_mean = round(sum(y) / len(y), 2)

        #slope is equivalent to the approaching speed
        speed = round(sum([(pt[0] - x_mean)*(pt[1] - y_mean) for pt in self.measurements])/sum([(x-x_mean)**2 for x in x]),2)

        #threshould is an approaching speed of 30cm per second 
        if speed <= -30:
            return True

        #no approach
        return False

    #####################################################

    ####         Servo and related algorithms        ####

    def _set_servo_angle(self,angle):
        """
            Moving the Servo motor to a predefined valid angle
            
            :param angle: an angle between 0 and 180
            :type angle: int
        """
        if angle >= 0 and angle <= 180:
            # you may change this depending on your servo
            duty = float(angle) /12 
            self.pulse.ChangeDutyCycle(duty)
            time.sleep(1)
            # 0 pulse means it is inactive
            self.pulse.ChangeDutyCycle(0)

    def _set_servo_power(self, state):
        """
            Powering on/off the servo motor with relais

            :param state: True for High (on), False for Low (off)
            :type state: boolean
        """
        GPIO.output(self.relais_pin, state)

    def unlock(self):
        """
            Sequence to unlock the door by using the servo motor
            You may edit the angles to adjust it your lock
        """
        self.unlocking = True
        #power servo on
        self._set_servo_power(True)
        #moving servo to unlocking position
        self._set_servo_angle(30)
        time.sleep(5)
        #moving the servo back to a resting postion
        self._set_servo_angle(72)
        #giving the servo enough time to move there before turning it off
        time.sleep(2)
        self._set_servo_power(False)
        self.unlocking = False

    ##################################################

    def run(self):
        """
            Constantly checking wheather someone tries to leave the room, to unlock the door
        """
        try:
            self._init_GPIO()
            #move servo to resting postion
            self._set_servo_power(True)
            self._set_servo_angle(72)
            time.sleep(2)
            self._set_servo_power(False)

            #taking measueremnts and gethering approaching status
            while True:
                #unlocking is only relevant if the door is closed
                if self.is_door_closed():
                    for i in range(len(self.measurements)):
                        self.measurements[i] = (time.time(),self.get_distance())
                        time.sleep(0.1)
                        if not self.is_door_closed():
                            self._reset_measurements()
                            break
                        elif self.approach_detected() and self.unlocking == False: 
                            #unlocking the door
                            self.unlock()
                            self._reset_measurements()

        except KeyboardInterrupt:
            #move servo to resting postion
            self._set_servo_power(True)
            self._set_servo_angle(72)
            time.sleep(2)
            self.pulse.stop()
            self._set_servo_power(False)

            GPIO.cleanup()
#TODO turn realais way in advance on (maybe servo as well) because there might be issues with turning it right before moving the servo
