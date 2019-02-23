library(ggplot2)
#library(tidyr)
library(dplyr)
library(mlbench)
library(pROC)
library(caret)

wine_data <- read.csv("../Desktop/DataAnalytics/datadets/winequality-white.csv")


#-------------------------------------------EDA----------------------------------------------------

#wine_data <- wine_data%>% mutate(quality_new = ifelse(quality>=6,"Good","Bad"))
wine_data %>% group_by(quality)

wine_data$taste <- ifelse(wine_data$quality < 4, 'Bad', ifelse(wine_data$quality<7,'Average','Good'))
#wine$taste[wine$quality == 6] <- 'normal'
wine_data$taste <- as.factor(wine_data$taste)


ggplot(data=wine_data, aes(x=taste,y=alcohol))+geom_boxplot()

ggplot(data=wine_data, aes(x=taste, y=total.sulfur.dioxide))+geom_boxplot()
ggplot(data=wine_data, aes(x=taste, y=density))+geom_boxplot()
ggplot(data=wine_data, aes(x=taste, y=pH))+geom_boxplot()
ggplot(data=wine_data, aes(x=taste, y=volatile.acidity))+geom_boxplot()



ggplot(data= wine_data, aes(y=fixed.acidity, x=alcohol, color=taste)) +geom_point()



ggplot(data=wine_data, aes(x=citric.acid))+geom_bar() # above average but not good shows highest spike, meaning that it is preffered but to an extent.
cor(wine_data$quality,wine_data$citric.acid)



#apply correlation plot to see in between variables
#install.packages("ggcorrplot")
#library(ggcorrplot)

#corr <- round(cor(wine_data$fixed.acidity),1)
#ggcorrplot(corr)


#draw for alcohol, fixed acidity
#model - linear regression


#PCA
Sample.scaled <- data.frame(apply(wine_data[,1:12], 2, scale))
Sample.scaled_wine <- data.frame(t(na.omit(t(Sample.scaled))))
pr.out <- prcomp(Sample.scaled_wine, retx=TRUE)
#pr.out <- prcomp(cr.pca,scale=TRUE)
biplot(pr.out)


#----------------bivariate analysis------------------------------------
ggplot(data = wine_data, aes(x = alcohol, y = fixed.acidity)) +
  geom_point() + 
  geom_abline(slope = 1, intercept = 0) + 
  geom_smooth(method = "lm", se = FALSE)

#apply()

set.seed(123)

#split train and test data in an 80/20 proportion
wine_data[, "train"] <- ifelse(runif(nrow(wine_data))<0.8, 1, 0)

#assign training rows to data frame trainset
trainset <- wine_data[wine_data$train == 1, ]
#assign test rows to data frame testset
testset <- wine_data[wine_data$train == 0, ]

#find index of "train" column
trainColNum <- grep("train", names(wine_data))

#remove "train" column from train and test dataset
trainset <- trainset[, -trainColNum]
testset <- testset[, -trainColNum]

# -------------------------------Load the rpart package---------------------------------
# building a simple rpart classification tree
library(rpart)
#install.packages("rpart.plot")
library(rpart.plot)


prune_control <- rpart.control(maxdepth = 30, minsplit = 20) 
m <- rpart(taste ~.-quality, data = trainset, method = "class", control = prune_control)
rpart.plot(m, type = 3, box.palette = "auto", fallen.leaves = TRUE)



# Make predictions on the test dataset
pred_tree <- predict(m,testset,type="class")

# Examine the confusion matrix
table(testset$taste, pred_tree)

# Compute the accuracy on the test dataset
mean(testset$taste == pred_tree)

#cross validation

ctrl <- trainControl(method = "cv", savePred=T, classProb=T)
mod <- train(taste~.-quality, data=trainset, method = "rpart", trControl = ctrl)
head(mod$pred)
plot(mod)


#----------------------------Random Forest[--]---------------------



# Load the randomForest package
library(randomForest)

# Build a random forest model
rfmodel <- randomForest(taste ~ .-quality, data = trainset,ntree=1000)


# Compute the accuracy of the random forest
pred_rf <- predict(rfmodel, trainset)
table(pred_rf, trainset$taste)
plot(rfmodel)
varImpPlot(rfmodel)
pred_rf_test <- predict(rfmodel, testset)
table(pred_rf_test, testset$taste)
mean(pred_rf_test == testset$taste)

#m <- randomForest(quality ~ fixed.acidity + alcohol, data = wine_train, ntree = 500, # number of trees in the forest 
 #                 mtry = sqrt(p)) # number of predictors (p) per tree


#reduce the number of trees

rfmodel2 <- randomForest(taste ~ .-quality, data = trainset,ntree=500)


# Compute the accuracy of the random forest
pred_rf2 <- predict(rfmodel2, trainset)


table(pred_rf2, trainset$taste)
plot(rfmodel2)
varImpPlot(rfmodel2)
pred_rf_test2 <- predict(rfmodel2, testset)
table(pred_rf_test2, testset$taste)
mean(pred_rf_test2 == testset$taste)

#cross validation

ctrl2 <- trainControl(method = "cv", savePred=T, classProb=T)
mod2 <- train(taste~.-quality, data=trainset, method = "rf", trControl = ctrl2)
head(mod2$pred)
plot(mod2)

#library(pROC)
#print(auc(testset$taste, pred_rf_test2))
#-------------------------------------------------KNN-----------------------------------------------------------

# Use kNN to identify the test sickness probability

wine_quality <- trainset$taste

wine_pred <- knn(train = trainset[-13], test = testset[-13], cl = wine_quality,k=5)

# Create a confusion matrix of the actual versus predicted values
quality_actual <- testset$taste
table(wine_pred,quality_actual)

# Compute the accuracy
accu <- mean(quality_actual == wine_pred)
plot(wine_pred)

#change the k value
wine_pred1 <- knn(train = trainset[-13], test = testset[-13], cl = wine_quality,k=3)

# Create a confusion matrix of the actual versus predicted values
quality_actual1 <- testset$taste
table(wine_pred1,quality_actual1)

# Compute the accuracy
mean(quality_actual1 == wine_pred1)
wine_pred1

#cross validation

ctrl3 <- trainControl(method = "cv", savePred=T, classProb=T)
mod3 <- train(taste~.-quality, data=trainset, method = "knn", trControl = ctrl3)
head(mod3$pred)
plot(mod3)
