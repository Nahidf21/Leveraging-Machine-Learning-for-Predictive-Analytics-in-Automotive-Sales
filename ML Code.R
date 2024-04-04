library(caret) ### caret library has the confusion matrix
library(forecast) ### forecast library has the prediction errors
library(rpart) 
library(rpart.plot) ### rpart and rpart.plot libraries are for the tree analysis
library(randomForest) ### for random forest


##### classification tree

carPurchase_data<- read.csv("Simulated_CarPurchase.csv")

### Partition data first
set.seed(123) 
train_index<-sample(rownames(carPurchase_data), dim(carPurchase_data)[1]*0.6)
valid_index<-setdiff(rownames(carPurchase_data),train_index)
train_data<-carPurchase_data[train_index, ]
valid_data<-carPurchase_data[valid_index, ]

### Train the model using the training data
mytree<- rpart(Buy ~ Budget + Age + Gender + FuelEfficiency + Preferred , data = train_data, method = "class")

prp(mytree)  ## plot the tree

### Predict using the validation data
predicted_values <- predict(mytree, newdata=valid_data, type = "class")

### Confusion matrix
confusionMatrix(relevel(as.factor(predicted_values), "Sports"), 
                relevel(as.factor(valid_data$Buy), "Sports"))



### Random forest

## since we are doing a classification random forest now,
## we need to convert some of our variables to factor (i.e., categorical) type
 
data<- read.csv("Simulated_CarPurchase.csv")

data$Gender <- as.factor(data$Gender)
data$FuelEfficiency <- as.factor(data$FuelEfficiency)
data$Preferred <- as.factor(data$Preferred)
# 'Buy' is the target variable
data$Buy <- as.factor(data$Buy) 

# Step 2: Split the data into training and testing sets
set.seed(123) 
trainIndex <- createDataPartition(data$Buy, p = .8, 
                                  list = FALSE, 
                                  times = 1)
dataTrain <- data[ trainIndex,]
dataTest  <- data[-trainIndex,]

# Step 3: Train the RandomForest model
# randomForest automatically handles factor variables for classification tasks

model <- randomForest(Buy ~ ., data = dataTrain, ntree = 500)

# Step 4: Predict and evaluate the model
predictions <- predict(model, newdata = dataTest)
confusionMatrix(predictions, dataTest$Buy)


# support vector machine
svm2 <- svm(Buy ~., data = dataTrain)

prediction2<-predict(svm2, dataTest)


# confusion matrix for validation data
confusionMatrix(as.factor(prediction2), as.factor(dataTest$Buy))



## bagging, boosting

carPurchase_data<- read.csv("Simulated_CarPurchase.csv")

set.seed(123) 
# partition the data
train.index <- sample(rownames(carPurchase_data), dim(carPurchase_data)[1]*0.7)  
valid.index<-setdiff(rownames(carPurchase_data),train.index)
train.df <- carPurchase_data[train.index, ]
valid.df <- carPurchase_data[valid.index, ]

# single tree
tr <- rpart(Buy ~ ., data = train.df)
pred <- predict(tr, valid.df, type = "class")
confusionMatrix(as.factor(pred), as.factor(valid.df$Buy))

### bagging
data<- read.csv("Simulated_CarPurchase.csv")

data$Gender <- as.factor(data$Gender)
data$FuelEfficiency <- as.factor(data$FuelEfficiency)
data$Preferred <- as.factor(data$Preferred)
# 'Buy' is the target variable
data$Buy <- as.factor(data$Buy) 

# Step 2: Split the data into training and testing sets
set.seed(123) 
trainIndex <- createDataPartition(data$Buy, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train.df <- data[ trainIndex,]
valid.df  <- data[-trainIndex,]

bag <- bagging(Buy ~ ., data = train.df)
pred_bag <- predict(bag, valid.df, type = "class")

pred_bag_factor <- factor(pred_bag$class, levels = levels(valid.df$Buy))
confusionMatrix(pred_bag_factor, valid.df$Buy)

### boosting
data<- read.csv("Simulated_CarPurchase.csv")

data$Gender <- as.factor(data$Gender)
data$FuelEfficiency <- as.factor(data$FuelEfficiency)
data$Preferred <- as.factor(data$Preferred)
# 'Buy' is the target variable
data$Buy <- as.factor(data$Buy)

set.seed(123) 
trainIndex <- createDataPartition(data$Buy, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train.df <- data[ trainIndex,]
valid.df  <- data[-trainIndex,]

boost <- boosting(Buy ~ ., data = train.df)
pred_bst <- predict(boost, valid.df, type = "class")

pred_bst_factor <- factor(pred_bst$class, levels = levels(valid.df$Buy))
confusionMatrix(pred_bst_factor, valid.df$Buy)




