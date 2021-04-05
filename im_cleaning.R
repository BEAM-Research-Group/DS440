# Front Matter
rm(list=ls())
gc()
# set seed for reproducibility 
set.seed(440)

# NOTES:
# Should pre-damaged building image be included as input?
# Yes, could use them to help predict no damage -> just make sure to label them as no damage
# However currently not included

# Import libraries
library(magick)
library(EBImage)
library("readxl")
library(hash)
library(stringr)

read_convert_save = function(type, im_list)
{
  # INPUT: 
  # type: type of set i.e. 'train', 'test', 'val'
  # im_list: list of the image names
  # OUTPUT: 
  # [optional] images: list of images (converted to .jpg if needed)
  # NOTE: 
  # Could potentially add transformations here, but
  # I'm pretty sure there's a parameter in keras library for R
  #images = vector(mode = "list", length = im_length)
  if (length(im_list) > 0)  # checks if there are images
  {
    print(type) # check the input
    #c = 0  # count used for debugging
    for (im_name in im_list)  # go through each image
    {
      # get image file
      im_file = paste0(im_hash[[im_name]], '/', im_name)
      print(im_name)
      # get true label
      # checks most common label first to increase speed 
      n = unlist(str_split(im_name, ' ', n=2))[1]  # get image name only
      im_label = 99
      if (n %in% labels_2$"GlobalID") {
        im_label = 2
      } else if (n %in% labels_1$"GlobalID") {
        im_label = 1
      } else if (n %in% labels_3$"GlobalID") {
        im_label = 3
      } else if (n %in% labels_4$"GlobalID") {
        im_label = 4
      } else if (n %in% labels_0$"GlobalID") {
        im_label = 0
      }  # images with NA labels are excluded
      print(n)
      print(im_label)
      if (im_label != 99)
        {
        # read image into r environment 
        im = readImage(im_file) 
        #print(im)
        #print(typeof(im))
        # convert to jpg and save
        writeImage(x = im, 
                   files = paste0('../data/clean_im3/', type, '/', im_label, '/', im_name), 
                   type = 'jpeg')
        }
    #c = c + 1
    #print(c)
    }  # end for
  } # end if
  else {print('No images provided for input.')}
} # end function

# save directory 
im_dir = "../data/PRJ-2301/D1. Building Assessments/Photographs/"
# there are folders named 1 through 233
folder_names = 1:233

# read in xlsx file with image name and their label
tab_data = read_excel("../data/PRJ-2301/D1. Building Assessments/HMichael_Data.xlsx")
# check all values of damage state
#unique(tab_data[,"Damage State"])
# get list of image names with damage state = 4
labels_4 = tab_data[tab_data$"Damage State"==4,]
labels_3 = tab_data[tab_data$"Damage State"==3,]
labels_2 = tab_data[tab_data$"Damage State"==2,]
labels_1 = tab_data[tab_data$"Damage State"==1,]
labels_0 = tab_data[tab_data$"Damage State"==0,]
# exclude NAs

# get images
# go through each folder
im_hash = hash()
for (folder in folder_names)
{
  # get image file names in current folder
  cur_path = paste0(im_dir, folder)  # current path
  # im_PNG = list.files(path = cur_path, pattern = '*.PNG')
  im_jpg = list.files(path = cur_path, pattern = '*.jpg')
  im_JPG = list.files(path = cur_path, pattern = '*.JPG')
  im_names = c(im_jpg, im_JPG) # c(im_PNG, im_jpg, im_JPG)
  for (im in im_names) {
    im_hash[[im]] = cur_path
  }
}

# split into train, validate, and test sets
# get count of all images
len_im = length(im_hash)
# portion of data that's for testing
test_per = 0.2
# take ceiling since can't have fractional count of data in a set
len_test = ceiling(test_per * len_im)  
# portion of data that's for training
train_per = 0.6  
len_train = ceiling(train_per * len_im)
# that leaves 20% for validation
# sample
# permute image names randomly
im_names_shuffled = sample(keys(im_hash))  # does not overwrite im_names
# the first randomly placed names are used in testing
test_im = im_names_shuffled[1:len_test]
# the next set randomly placed names are used in training
train_im = im_names_shuffled[(len_test + 1):(len_test + len_train)]
# the rest of the data are used in validation 
val_im = im_names_shuffled[(len_test + len_train + 1):len_im]

read_convert_save('test', test_im)
read_convert_save('train', train_im)
read_convert_save('val', val_im)
