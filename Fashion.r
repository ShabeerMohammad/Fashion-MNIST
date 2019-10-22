#to classify the images of clothing like sneakers and shirts.
library(keras)
#load the the fashion MNIST dataset which contains images of handwritten(0,1,2,3,etc.)
fashion_mnist <- dataset_fashion_mnist()

#We will use 60k images to train and 10k images to evaluate how accurately the network learned to classfiy images.

c(train_images,train_labels)%<-%fashion_mnist$train
c(test_images,test_labels)%<-%fashion_mnist$test

#Since the class names are not included with the dataset,we will store them in a vector to use later when plotting the images

class_names <- c('T-shirt/top','Trouser','Pullover',
                 'Dress','Coat','Sandal','Shirt','Sneaker',
                 'Bag','Ankle boot')

#Explore the data there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:
dim(train_images)

#Likewise, there are 60,000 labels in the training set:
dim(train_labels)

#Each label is an integer between 0 and 9:
train_labels[1:20]

#There are 10K images in the test set. Again,each image is represented as 28*28 pixels:
dim(test_images)

#And the test set contains 10k images labels:
dim(test_labels)

#Preprocess the data
#Let's inspect the first image in the training set and see the pixel values are ranging between 0 to 255:
library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, ,])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1,"x","value",-y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1,aes(x=x,y=y,fill=value))+
  geom_tile()+
  scale_fill_gradient(low='white',high='black',na.value = NA)+
  scale_y_reverse()+
  theme_minimal()+
  theme(panel.grid=element_blank())+
  theme(aspect.ratio = 1)+
  xlab("")+
  ylab("")
  
#We scale these values to a range of 0 to 1 before feeding to the neural network. For this, we simply divide by 255.
#(both training set and testing set)

train_images <- train_images/255
test_images <- test_images/255

#Display the first 25 images from the training set and display the class name below each image. Verifying the data is in correct format and 
#we're ready to build and train the network

par(mfcol=c(5,5))
par(mar=c(0,0,1.5,0),xaxs='i',yaxs='i')
for(i in 1:25){
  img <- train_images[i, ,]
  img <- t(apply(img,2,rev))
  image(1:28,1:28,img,col=gray((0:255)/255),xaxt='n',yaxt='n',
        main=paste(class_names[train_labels[i]+1]))
}

#Build The Model
#Set up the layers

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28,28))%>%
  layer_dense(units = 128,activation = 'relu')%>%
  layer_dense(units=10,activation = 'softmax')

#Compile the model 

model %>% compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=c('accuracy')
)

#Train the model
model %>% fit(train_images,train_labels,epochs=5)


#Evaluate Accuracy
#Compare how the model performs on the test dataset

score <- model%>%evaluate(test_images,test_labels)

cat('Test loss:',score$loss,"\n")
cat('Test accuracy:',score$acc,"\n")

#Make Predictions
#With the model trained, we can use it to make predictions about some images

predictions <- model %>%predict(test_images)

#The model has predicted the label for each image in the testing set. Let's take a look at the first prediction:

predictions[1, ]

#A prediction is an array of 10 numbers. These describe the "confidence" of the model that the image corresponds to each of the 10 different articles of clothing.

#Let's see which label has the highest confidence value:

which.max(predictions[1,])

#Alternatively, we can also directly get the class prediction:

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]

#As the labels are 0-based, this actually means a predicted label of 9(to be found in class_names[9]).
#So the model is most confident that this image is an ankle boot. And we can check the test label to see this is correct.

test_labels[1]

#Let's plot the several images with their predictions.
#Correct prediction labels are green and incorrect prediction labels are red.

par(mfcol=c(5,5))
par(mar=c(0,0,1.5,0),xaxs='i',yaxs='i')
for(i in 1:25){
  img <- test_images[i, , ]
  img <- t(apply(img,2,rev))
  #subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ])-1
  true_label <- test_labels[i]
  if(predicted_label == true_label){
    color=<-'#008800'
  }else{
    color <- "#bb0000"
  }
  image(1:28,1:28,img,col=gray((0:255)/255),xaxt='n',yaxt='n',
        main=paste0(class_names[predicted_label+1],"(",
                    class_names[true_label+1],")"),
        col.main=color)
}

#Finally, use the trained model to make a prediction about a single image.
#Grab an image from the test dataset 
#take care to keep the batch dimension, as this is expected by the model
img <- test_images[1, , ,drop=FALSE]
dim(img)

#Now predict the image
predictions <- model %>% predict(img)
predictions

#subtract 1 as labels are 0-based
prediction <- predictions[1, ]-1
which.max(prediction)

#Or, directly getting the class prediction again:
class_pred <- model%>%predict_classes(img)
class_pred

#And,as before, the model predicts a label of 9.
