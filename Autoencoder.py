# imports
import tensorflow as tf
import matplotlib.pyplot as plt

# function to create weights
def create_weights(input_data_column_size, layer_nuerons):
    return tf.Variable(tf.random.normal(shape=(layer_nuerons, input_data_column_size)))

# Implmentation for Forward pass function
# Transpose data and result when using w @ data else if we use data @ w then we will have to transpose all the weights 
# changing the formula ends up saving computations in case of larger models , for the current implementation saving is neglible 
def forward_pass(data , layer1_weight, layer2_weight, layer3_weight , layer4_weight , layer5_weight , final_layer_weight):
  # Relu layer with 128 nuerons
  layer1_input = layer1_weight @ tf.transpose(data)
  output_layer1 = tf.keras.activations.relu(layer1_input) 
  # Relu layer with 64 nuerons
  layer2_input = layer2_weight @ output_layer1  
  output_layer2 = tf.keras.activations.relu(layer2_input)
  # Relu layer with 32 nuerons
  layer3_input = layer3_weight @ output_layer2 
  output_layer3 = tf.keras.activations.relu(layer3_input)
  # Relu layer with 64 nuerons
  layer4_input = layer4_weight @ output_layer3 
  output_layer4 = tf.keras.activations.relu(layer4_input)
  # Relu layer with 128 nuerons
  layer5_input =  layer5_weight @ output_layer4
  output_layer5 = tf.keras.activations.relu(layer5_input)
  # sigmoid layer as final layer of 784 nuerons
  final_layer_input = final_layer_weight @ output_layer5  
  final_layer_output = tf.math.sigmoid(final_layer_input)

  return tf.transpose(final_layer_output)

# Implmentation mean_absolute_error function
# https://en.wikipedia.org/wiki/Mean_absolute_error 

def mean_absolute_error(predicted , input_data):
  return (1/predicted.shape[0])*(tf.math.reduce_sum(tf.math.abs(input_data - predicted)))

def training_model(Iterations, optimizer, train_data, test_data, weight1, weight2, weight3, weight4, weight5, weight6):
    # list is used to store loss occurred in each iteration of gradient tape
    trainingLoss = []

    # Training on the model
    for i in range(1,Iterations+1):
        # Create an instance of Gradient Tape to monitor the forward pass to calculate the gradients based on the training data
        # This provides information on how to adjust each weight and bias to minimize the loss.
        with tf.GradientTape() as tape:
            # call forward pass - acts as our model of 6 layers
            y_pred = forward_pass(train_data, weight1, weight2, weight3, weight4, weight5, weight6)
            
            # call mean_absolute_error  
            currentLoss = mean_absolute_error(y_pred, test_data)
            trainingLoss.append(currentLoss)

        # Calculate the gradients
        gradients = tape.gradient(currentLoss, [weight1, weight2, weight3, weight4, weight5, weight6])
  
        # apply optimizer to minimize loss
        optimizer.apply_gradients(zip(gradients, [weight1, weight2, weight3, weight4, weight5, weight6]))
        if i % 50 == 0 :
            print(f"Current Iteration: {i} with Training Loss: {currentLoss}")
    return trainingLoss

def show_image(original_image_set , reconstruction_image_set, title):
  # Number of images to display 
  n = 10  
  plt.figure(figsize=(20, 4))
  plt.suptitle(title.title())    
  for i in range(n):
      # Display original
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(original_image_set[i].numpy().reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # Display reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(reconstruction_image_set[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  
  plt.show()

def main():

  (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

  # Normalize train and test data
  x_train = x_train.astype('float32') / 255.0
  x_test = x_test.astype('float32') / 255.0

  # Reshape so that each instance is a linear array of 784 normalized pixel values
  x_train = x_train.reshape((len(x_train), 784))
  x_test = x_test.reshape((len(x_test), 784))

  print ("Shape of training feature data is ", x_train.shape)
  print ("Shape of testing feature data is ", x_test.shape)

  # Add random noise to the image
  noise_factor = 0.2
  x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
  x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

  # Clip the resulting values so that they don't fall outside the upper and lower normalized value of 0 and 1
  x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
  x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

  # display the images: noisy vs original   
  show_image(original_image_set= x_test_noisy  , reconstruction_image_set= x_test, title="noisy vs original")
  
  print (x_train.shape, x_test.shape)
  print (x_train_noisy.shape, x_test_noisy.shape)
  
  # define weights for layers
  weight1 = create_weights(layer_nuerons=128,input_data_column_size=x_train_noisy.shape[1])
  weight2 = create_weights(layer_nuerons=64 ,input_data_column_size=weight1.shape[0])
  weight3 = create_weights(layer_nuerons=32 ,input_data_column_size=weight2.shape[0])
  weight4 = create_weights(layer_nuerons=64 ,input_data_column_size=weight3.shape[0])
  weight5 = create_weights(layer_nuerons=128 ,input_data_column_size=weight4.shape[0])
  final_layer_weight = create_weights(layer_nuerons=784,input_data_column_size=weight5.shape[0])
  
  # define the optimizer
  adam_optimizor = tf.keras.optimizers.Adam(learning_rate=0.001)
  
  # set the limit for iterations
  max_iterations = 500

  # Call for training the model , after this function ends the weights will get updated automaticaly
  training_loss = training_model(Iterations=max_iterations ,optimizer=adam_optimizor,train_data=x_train_noisy , test_data=x_train , weight1=weight1 , weight2=weight2, weight3=weight3, weight4=weight4 , weight5=weight5 , weight6=final_layer_weight)
  
  # Final Prediction after Training model 
  y_pred =  forward_pass(x_train_noisy, weight1 , weight2 , weight3 , weight4, weight5, final_layer_weight)

  # display the images: predicted vs original   
  show_image(original_image_set= y_pred  , reconstruction_image_set= x_test, title="predicted vs original")

  # finally plot comparison as in lecture slides
  plt.plot(training_loss, label="Train Loss")
  plt.legend()
  plt.show()


main()