# clean environment
rm(list = ls()) # removes environment variables 
gc() # garbage collection

# install as needed
#install.packages("BiocManager") 
#BiocManager::install("EBImage")

# libraries 
library(reticulate)
library(keras)  
library(tensorflow)
library(tidyverse)
#library(pillow)
library(EBImage)
#library(kerasR)
# library(imager) # 502 status when installing 

# install_keras()
# run after calling library(keras)
# helps configure keras to your machine
# if taking long with lazy initialization errors, restart R
# only need to run once, or as needed to handle errors

# set up reticulate anaconda environment 
# since keras depends on it for python packages 
# specifically, fit_generator() from keras requires PIL 
# from python pillow library 
# reticulate::py_install("pillow",env=tf)
conda_list()[[1]][1] %>% 
  use_condaenv(required = TRUE)

# Info on configuring conda environment  
# https://stackoverflow.com/questions/57612703/could-not-import-pil-image-even-if-pillow-already-installed 

# if using kerasE
# reticulate::use_python()
# kerasR::keras_init()

# image_data_generator transformations 
# https://towardsdatascience.com/exploring-image-data-augmentation-with-keras-and-tensorflow-a8162d89b844
# https://keras.rstudio.com/reference/image_data_generator.html

# Basic example of image cnn in R 
# https://www.shirin-glander.de/2018/06/keras_fruits/

##################### MODELING #################################################
batch_size = 32
epochs = 20
num_labels = 5
img_width = 20
img_height = 20
target_size = c(img_width, img_height) # scale down images
channels = 1 # color images were converted to black and white when reading in

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

cat("number of images per class:")                         
table(factor(train_image_array_gen$classes))

# number of training samples
train_samples = train_image_array_gen$n
# number of validation samples
valid_samples = valid_image_array_gen$n

# build model
model = keras_model_sequential()

model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
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

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# fit
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
    callback_model_checkpoint("/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/keras/fruits_checkpoints.h5", save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = "/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/keras/logs")
  )
)


# check model structure (layer summary)
summary(model)

# fit the model
# use fit with generator rather than fit_generator
hist <- model %>% fit_generator(
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

# visualize the history 
plot(hist)
