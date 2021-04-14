##################### MODIFICATIONS ############################################
# - Decreased epochs to 10
#     - not much change was occurring after 10
#     - will save on training time
# - Added another layer so there would be an odd number of layers in the model (3 layers total)

##################### CLEAN ENVIRONMENT ########################################
#.rs.restartR()  # to restart R session (primarily for debugging purposes)
#rm(list = ls()) # removes environment variables 
#gc()            # garbage collection
##################### INSTALATIONS #############################################
#install.packages("BiocManager") 
#BiocManager::install("EBImage")

##################### LIBRARIES ################################################ 
library(reticulate)
library(keras)  
library(tensorflow)
library(tidyverse)
library(EBImage)

##################### ERROR HANDELING ##########################################
# library(rgee)
# use_python('C:/Users/Markus/AppData/Local/r-miniconda/envs/r-reticulate/python.exe')

# install_tensorflow()

#install_keras()
# *run after calling library(keras) in a fresh R*
# helps configure keras to your machine 
# if taking long with lazy initialization errors, restart R
# only need to run once, or as needed to handle errors
# notably, it can help resolve issue where PIL import is not found since running
# this line is supposed to install pillow. However, this error may still show up
# regarding system dependencies 
# Other ways o install pillow include...
# set up reticulate anaconda environment 
#   since keras depends on it for python packages 
#   specifically, fit_generator() from keras requires PIL 
#   from python pillow library 
#   my environment location: 'C:/Users/Markus/AppData/Local/r-miniconda/envs/r-reticulate/python.exe'
#   reticulate::py_install("pillow", env=tf)
#conda_list()[[1]][1] %>% 
#  use_condaenv(required = TRUE)
# conda install -c conda-forge pillow
# You can also do it directly through Anaconda's GUI Interface for the specific 
#    conda environment 
# pip3 install pillow
# if the above still does not work, activate the reticulate anaconda environment 
# in the terminal and then install pillow. Also make sure conda is up to date
#   conda update -n base -c defaults conda
#   conda info --envs
#   conda activate C:\Users\Markus\AppData\Local\r-miniconda\envs\r-reticulate
#   conda install -c anaconda pillow

# check that other keras dependancies are installed
# pip install --upgrade scipy h5py pyyaml requests Pillow

# Info on configuring conda environment  
# https://stackoverflow.com/questions/57612703/could-not-import-pil-image-even-if-pillow-already-installed 

# make sure Rtools is installed 

# image_data_generator transformations 
# https://towardsdatascience.com/exploring-image-data-augmentation-with-keras-and-tensorflow-a8162d89b844
# https://keras.rstudio.com/reference/image_data_generator.html

# Basic example of image cnn in R 
# https://www.shirin-glander.de/2018/06/keras_fruits/

# make sure used python imports are up to date
# pip install --upgrade tensorflow keras pandas sklearn pillow 
# note tensorflow 2.4.1 requires numpy~=1.19.2, which is not the most recent numpy
# python -m pip install numpy==1.19.2

##################### SET UP CONSTANTS, TRANSFORMATIONS, AND DATA FOR MODEL ####
batch_size  = 32
epochs      = 10
num_labels  = 5
img_width   = 20
img_height  = 20
target_size = c(img_width, img_height) # scale down images
channels    = 1 # color images were converted to black and white when reading in

data_gen= image_data_generator(rotation_range     = 90,
                               width_shift_range  = 0.5,
                               height_shift_range = 0.5,
                               # brightness_range = c(0.2, 1),
                               shear_range        = 40,
                               zoom_range         = 40,
                               horizontal_flip    = TRUE,
                               vertical_flip      = TRUE,
                               rescale            = 1/255)

# make sure sub directories are set up 
# (directory points to folder that contains a folder that has images)

# read in the data
train_image_array_gen  = flow_images_from_directory(directory   = '../data/clean_im3/train',
                                                    generator   = data_gen,
                                                    # subset      = 'training',
                                                    target_size = target_size,
                                                    class_mode  = "categorical",
                                                    batch_size  = batch_size,
                                                    seed        = 42,
                                                    color_mode  = 'grayscale')

valid_image_array_gen  = flow_images_from_directory(directory   = '../data/clean_im3/val',
                                                    generator   = data_gen,
                                                    # subset      = 'validation',
                                                    target_size = target_size,
                                                    class_mode  = "categorical",
                                                    batch_size  = batch_size,
                                                    seed        = 42,
                                                    color_mode  = 'grayscale')

test_image_array_gen = flow_images_from_directory(directory     = '../data/clean_im3/test', 
                                                  generator     = data_gen,
                                                  target_size   = target_size,
                                                  class_mode    = "categorical",
                                                  batch_size    = batch_size,
                                                  seed          = 42,
                                                  color_mode    = 'grayscale')

# check number of images in each class (0-4)
cat("number of images per class:")                         
table(factor(train_image_array_gen$classes))

# number of training samples
train_samples = train_image_array_gen$n
# number of validation samples
valid_samples = valid_image_array_gen$n

##################### CREATE THE MODEL ######################################## 
model = keras_model_sequential()

model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Third hidden layer
  layer_conv_2d(filter = 8, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(num_labels) %>% 
  layer_activation("softmax")

# print a table summarizing the model structure 
summary(model)

##################### COMPILE THE MODEL ######################################## 
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

##################### FIT THE MODEL ############################################ 
# use fit with generator rather than fit_generator
hist = model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint("logs/checkpoints1.h5", save_best_only = TRUE),
    # only needed for visualizing with TensorBoard
    callback_tensorboard(log_dir = "logs")
  )
)

##################### VISUALIZATION ############################################ 

# visualize the history of fitting
plot(hist)

## END
