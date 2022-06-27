from os import listdir
#USAGE: Save the static bboxes as dir_path/static_labels.txt

#Path to the frames/labels directory
dir_path = "./frames/"
#Load the static labels
f = open(dir_path + "static_labels.txt", "r")
#Read the contents
bboxes = f.read()
f.close()

#Get all file names in the directory
all_files = listdir(dir_path)
#Array of label text files
text_files = []
#Go through each filename in the directory
for filename in all_files:
    #If file has a name ending with .txt
    if filename.endswith(".txt"):
        #Add filename to the text files array
        text_files.append(filename)

#Go through each file
for frame_labels in text_files:
    #We don't need to overwrite classes file
    if frame_labels == "classes.txt":
        continue
    #Path to the frame label file
    fname = dir_path + frame_labels
    #Append static labels at the end of the file
    f = open(fname, "a")
    f.write(bboxes)
    f.close()