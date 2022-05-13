import cv2

#TODO: Create a classes.txt, duplicate 0th frame for further static objects labels propogation
#Load a pretrained tiny yolo network
#YOLO-608 works best
#Link: https://pjreddie.com/darknet/yolo/
model = cv2.dnn.readNetFromDarknet("cfg.cfg", "yolov3.weights")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#Path to the directory with video file in it
#NOTE: It gets really messy
dir_path = "C:\\Users\\ilkute\\Downloads\\aaa\\video_data\\"
#Name of the video file
video_name = "v_4.avi"


#Array of COCO classes that we are interested in 
#Person, car, truck, boat
interested_classes = [1, 3, 8, 9]
#Counter of processed frames. Used for file naming
frame_number_file = 0

#Define thresholds for the subframe level
probability_threshold_subframe = 0.01
nms_threshold_subframe = 0.2

#Define thresholds for the frame level
probability_threshold_whole = 0.02
nms_threshold_whole = 0.2

#Returns 9 subframes of a frame, left-mid-right, top-mid-bot
def getNineSubframes(frame):
    #Find frame's dimensions
    frame_height, frame_width, frame_channels = frame.shape
    #Array for storing the subframes
    subframes = []
    #Now we need to get the parts of the image. I am pretty sure that it is possible to do it automatically(same as with scaling on line 106), but i am too lazy to do it for now.
    #Top left
    subframes.append(frame[0:int(frame_height/2), 0:int(frame_width/2)])
    #Top middle
    subframes.append(frame[0:int(frame_height/2), int(frame_width/4):int(frame_width/2 + frame_width/4)])
    #Top right
    subframes.append(frame[0:int(frame_height/2), int(frame_width/2):frame_width])
    #Middle left
    subframes.append(frame[int(frame_height/4):int(frame_height/2 + frame_height/4), 0:int(frame_width/2)])
    #Middle middle
    subframes.append(frame[int(frame_height/4):int(frame_height/2 + frame_height/4), int(frame_width/4):int(frame_width/2 + frame_width/4)])
    #Middle right
    subframes.append(frame[int(frame_height/4):int(frame_height/2 + frame_height/4), int(frame_width/2):frame_width])
    #Bottom left
    subframes.append(frame[int(frame_height/2):frame_height, 0:int(frame_width/2)])
    #Bottom middle
    subframes.append(frame[int(frame_height/2):frame_height, int(frame_width/4):int(frame_width/2 + frame_width/4)])
    #Bottom right
    subframes.append(frame[int(frame_height/2):frame_height, int(frame_width/2):frame_width])
    return subframes

#Function for processing the whole frame
def process_frame(frame, model):
    #New plan: 
    #1. Divide each frame into 4 same-sized parts (or 9, since the objects are going to be also present on the sides of the subframes)
    #2. Process each subframe
    #3. Mathematically process each bbox to be relative to the whole frame
    #4. Apply NMS on the resulting data

    #Frame counter for saving the results
    global frame_number_file
    #Output layers of the network
    global output_layers
    #Save the frame itself 
    cv2.imwrite(dir_path + str(frame_number_file) + ".jpg", frame)

    #String for storing the output values
    frame_boxes = ""

    #Find frame's dimensions
    frame_height, frame_width, frame_channels = frame.shape
    #Array for storing the subframes
    subframes = getNineSubframes(frame)

    #Conver all subframes into blobs
    for idx, subframe in enumerate(subframes):
        subframes[idx] = cv2.dnn.blobFromImage(subframe, 1/255, (416, 416), [0,0,0], 1, crop=False) 

    #A holder for the subframes's bbox outputs(same order)
    bboxes_from_subframes = []
    #A holder for the subframes's class outputs(same order)
    classes_from_subframes = []
    #A holder for the subframes's confidences outputs(same order)
    confidences_from_subframes = []

    #Process each subframe, e.g. forward each subframe through YOLO and store the results
    for subframe in subframes:
        #Feed the blob to the neural network
        model.setInput(subframe)
    
        #Run forward the neural network and get the outputs from the output layers
        out = model.forward(output_layers)

        #Process the outputs and store classes, bboxes and confidences arrays for the subframe
        subframe_classes, subframe_bboxes, subframe_confidences = process_subframe(out)

        #Everything is stored in the original format: Left-mid-right, top-mid-bot
        bboxes_from_subframes.append(subframe_bboxes)
        classes_from_subframes.append(subframe_classes)
        confidences_from_subframes.append(subframe_confidences)
    
    #Now, we have to process bboxes, since they are relative to the subframe, not the whole frame
    #There are 3 subframes in each row. The width relative offsets are: 0, 0.25, 0.5
    #There are 3 rows of subframes. The height relative offsets are: 0, 0.25, 0.5
    #Each subframe's width and height is 1/2 of the whole frame. Thus, we only need to scale the bbox's width and height by 1/2
    #Initialize offsets
    cx = 0
    cy = 0
    #Go through each subframe
    for idx, bboxes in enumerate(bboxes_from_subframes):
        #Go through each subframe's bbox
        for idx1, bbox in enumerate(bboxes):
            #Bbox is : x,y,width,height
            #Scale bbox to be relative to the whole frame
            bboxes[idx1][0] = bbox[0] * 0.5 + cx
            bboxes[idx1][1] = bbox[1] * 0.5 + cy
            bboxes[idx1][2] = bbox[2] * 0.5 
            bboxes[idx1][3] = bbox[3] * 0.5 
        #No need to save new bbox since we were working with addresses
        #Shift the width offset
        cx += 0.25
        #If the width offset is bigger than 0.5, change to the next row
        if cx > 0.5:
            #Restart at the beginning
            cx = 0
            #Change the height offset
            cy += 0.25
    
    #Now it's time to apply non-maxima surpression. It will delete bboxes of objects that were detected on multiple subframes
    #But before that, we need to flatten the bboxes, classes and confidences arrays to not to have a subframe dimension
    #This may be converted to the beautiful numpy code, but let it be like that for now
    global_bboxes = []
    #Go through each subframe
    for subframe_bboxes in bboxes_from_subframes:
        #Go through each bbox
        for bbox in subframe_bboxes:
            #Append bbox to the global bbox array
            global_bboxes.append(bbox)

    global_classes = []
    #Go throug each subframe
    for subframe_classes in classes_from_subframes:
        #Go through each class
        for classs in subframe_classes:
            #Append class to the global class array
            global_classes.append(classs)

    confidencess = []
    #Go through each subframe
    for subframe_confidencess in confidences_from_subframes:
        #Go throug each confidence
        for confidence in subframe_confidencess:
            #Append confidences to the global confidences array
            confidencess.append(confidence)

    #Apply the non maxima surpression on all of the bounding boxes.
    #This will remove all the excessive bboxes that were foung on the sides of subframes
    nms_indexes = cv2.dnn.NMSBoxes(global_bboxes, confidencess, probability_threshold_whole, nms_threshold_whole)


    #Go through each index that survived NMS annihilation
    for i in nms_indexes:
        #Get the bounding box
        box = global_bboxes[i]
        #Get the class
        classs = global_classes[i]

        #Add new entry to the string that will be written as an annotation to the frame
        frame_boxes += str(classs) + " "
        frame_boxes += str(box[0]) + " "
        frame_boxes += str(box[1]) + " "
        frame_boxes += str(box[2]) + " "
        frame_boxes += str(box[3]) + " "
        frame_boxes += '\n'
        #I also want to show the bounding boxes for visual inspection during processing
        #Scale them(values are in 0-1 format, need to multiply by original w and h)
        #Reformat center coordinates from relative format to pixel format
        cx = int(box[0] * frame_width)
        cy = int(box[1] * frame_height)
        #Reformat width and height from relative format to pixel format
        w = int(box[2] * frame_width)
        h = int(box[3] * frame_height)
        #Get top left corner
        #To get to the left, we need to substract width/2 from the center coordinate
        x = int(cx - w/2)
        #To get to the top, we need to substract height/2 from the center coordinate
        y = int(cy - h/2)
        #Get bottom right corner
        #To get to the right, we need to add width/2 to the center coordinate
        x1 = int(cx + w/2)
        #To get to the bottom, we need to add height/2 to the center coordinate
        y1 = int(cy + h/2)
        #Draw a bounding box using resulting coordinates
        cv2.rectangle(frame, (x,y), (x1, y1), (0,255,0), 2)

    #Now i want to save the file that contains generated annotations
    #Open/create a new txt file with a processed frame's number as name
    f = open(dir_path + str(frame_number_file) + ".txt", 'w')
    #Write all the (class, cx, cy, w, h) to the file
    f.write(frame_boxes)
    #Close the file
    f.close
    #Increment processed frame's counter
    frame_number_file += 1

    #Show the image with all the resulting bounding boxes
    cv2.imshow("test", frame)

#A function for processing subframes. Returns bboxes, corresponding classes and confidences
def process_subframe(out):
    #Array of subframe's bboxes
    bboxes = []
    #Array of subframe's bboxes' confidences
    confidences = []
    #Array of subframe's bboxes' classes
    classes = []

    #Go through each yolo's output layer
    for layerOut in out:
        #Go through each detection
        for detection in layerOut:
            #Find the most probable class
            max_prob = 0
            max_class = 0
            #We only want to know certain classes
            for clss in interested_classes:
                #We need to add 4 to the index, since first 5 values are the bounding box properties(cx,cy,w,h,probability that there is a object at all)
                #If confused: Indexsing of arrays start from 0, and indexing of COCO classes starts from 1, which gives index of 5 for the first class(person)
                if(detection[clss + 4] > max_prob):
                    #Update if new max
                    max_prob = detection[clss]
                    max_class = clss

            #If certain enough
            if max_prob > probability_threshold_subframe:
                #Append new class entry
                classes.append(max_class)
                #Append new bbox entry(in the subframe-relative format)
                bboxes.append([detection[0], detection[1], detection[2], detection[3]])
                #Append confidence new entry
                confidences.append(max_prob)
    #Apply NMS on the subframe level
    nms_indexes = cv2.dnn.NMSBoxes(bboxes, confidences, probability_threshold_subframe, nms_threshold_subframe)
    #Holders for the remaining bboxes, classes and their confidences
    nms_bboxes = []
    nms_classes = []
    nms_confidences = []

    #Pick only the ones that have passed NMS
    for i in nms_indexes:
        nms_bboxes.append(bboxes[i])
        nms_classes.append(classes[i])
        nms_confidences.append(confidences[i])

    #Return resulting arrays
    #We also need to return the confidences for further NMS process
    return nms_classes,nms_bboxes, nms_confidences

#Get a video capture
cap = cv2.VideoCapture(dir_path + video_name)

#Get output layer names. They may differ from model to model.
layers = model.getLayerNames()
output_layers = []

#Get output layers' indexes from the yolo's built-in method
for i in model.getUnconnectedOutLayers():
    #Append the output layer. Built-in method's indexing starts from one, so need to substract 1
    output_layers.append(layers[i-1])

while cap.isOpened():
    #Read a new frame
    ret, frame = cap.read()

    #Check if file hasn't ended
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    #Process the frame
    process_frame(frame, model)

    #Check if interrupted
    if cv2.waitKey(1) == ord('q'):
        break

#End the job
cap.release()
cv2.destroyAllWindows()