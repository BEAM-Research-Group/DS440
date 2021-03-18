library(keras)
# library(tensorflow)

# image_data_generator transformations 
# https://towardsdatascience.com/exploring-image-data-augmentation-with-keras-and-tensorflow-a8162d89b844
# https://keras.rstudio.com/reference/image_data_generator.html

# Basic example of image cnn in R
# https://www.shirin-glander.de/2018/06/keras_fruits/

##################### MODELING #####################
batch_size = 32
data_gen= image_data_generator(rotation_range     = 90,
                               width_shift_range  = 0.5,
                               height_shift_range = 0.5,
                               # brightness_range = c(0.2, 1),
                               shear_range        = 40,
                               zoom_range         = 40,
                               horizontal_flip    = TRUE,
                               vertical_flip      = TRUE,
                               rescale            = 1/255)

train_image_array_gen  = flow_images_from_directory(directory   = '../data/clean_im/train',
                                                    generator   = train_data_gen,
                                                    subset      = 'training',
                                                    target_size = c(20, 20),
                                                    class_mode  = "categorical",
                                                    batch_size  = batch_size,
                                                    seed        = 42,
                                                    color_mode  = 'grayscale')

valid_image_array_gen  = flow_images_from_directory(directory   = '../data/clean_im/val',
                                                    generator   = data_gen,
                                                    subset      = 'validation',
                                                    target_size = c(20, 20),
                                                    class_mode  = "categorical",
                                                    batch_size  = batch_size,
                                                    seed        = 42,
                                                    color_mode  = 'grayscale')

test_image_array_gen = flow_images_from_directory(directory     = '../data/clean_im/test', 
                                                  generator     = data_gen,
                                                  target_size   = c(20, 20),
                                                  class_mode    = "categorical",
                                                  batch_size    = batch_size,
                                                  seed          = 42,
                                                  color_mode    = 'grayscale')

cat("number of images per class:")                         
table(factor(train_image_array_gen$classes))

# check encoding
train_image_array_gen_t = train_iamge_array_gen$class_indeces %>%
  as.tibble()
cat("/nClass label vs index mapping:\n")
train_image_array_gen_t

# number of training samples
train_samples = train_image_array_gen$n
# number of validation samples
valid_samples = valid_image_array_gen$n

epochs = 20

# build model
model = keras_model_sequential()

model%>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(128*128*3)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

model %>%
  layer_conv_2d(filter      = 32, 
                kernel_size = c(3,3), 
                padding     = "same", 
                input_shape = c(20, 20, 3)) %>%
  layer_activation("relu") 

# Second hidden layer
layer_conv_2d(filter      = 16, 
              kernel_size = c(3,3), 
              padding     = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() 
