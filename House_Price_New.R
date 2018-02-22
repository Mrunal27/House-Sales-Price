library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

system("ls ../input")

# Any results you write to the current directory are saved as output.
# Fitting NN to predict House Prices

library(data.table)
library(dplyr)
library(plyr)
library(neuralnet)


#install.packages("Boruta")
library(Boruta)

#Load Train and Test Datasets
data <- data.table(read.csv("train.csv", header = T, stringsAsFactors = FALSE))
data2 <- data.table(read.csv("test.csv", header = TRUE, stringsAsFactors = FALSE))
str(data)

summary(data)
SalePrice <- data$SalePrice
data$SalePrice <- NULL
fulldata <- rbind(data,data2)

#NeuralNet with Numerical Values amongst the dataset

# Selecting numeric variables
nums <- which(sapply(fulldata, is.integer))
fulldata <- fulldata[, nums,with = FALSE]

# Checking and ReplacingNA values
apply(fulldata, 2, function(x) sum(is.na(x)))
fulldata[is.na(fulldata)] <- 0
apply(fulldata, 2, function(x) sum(is.na(x)))

# Separating datasets
nrow(data)
data.na <- fulldata[1:nrow(data),] 
data.na.te <- fulldata[(nrow(data)+1):nrow(fulldata),]

# data.na <- Saleprice column
data.na <- cbind(data.na, SalePrice)

# Splitting train data 90% train 10% validade
index <- sample(1:nrow(data.na), round(0.90*nrow(data.na)))
train.na <- data.na[index,] #train data
val.na <- data.na[-index,] #val data set

# Normalizing train and val data sets
maxs <- apply(data.na, 2, max)
mins <- apply(data.na, 2, min)
scaled <- as.data.frame(scale(data.na, center = mins, scale = maxs - mins))

train <- scaled[index,]
test <- scaled[-index,]

#------------------------------------------SalesPrice using DeepLearning---------------------------------------------------------------------------------------------------------
install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
#data <- data.table(read.csv("train.csv", header = T, stringsAsFactors = FALSE))

model = h2o.deeplearning(y = 'SalePrice',
                         training_frame = as.h2o(train),
                         activation = 'Rectifier',
                         hidden = c(2,2),
                         epochs = 100,
                         train_samples_per_iteration = -2)


gbm <- h2o.gbm(x = 1:10, y = "SalePrice", training_frame = as.h2o(train), validation_frame = as.h2o(test),
               ntrees=500, learn_rate=0.01, score_each_iteration = TRUE)
plot(gbm)
plot(gbm, timestep = "duration", metric = "deviance")
plot(gbm, timestep = "number_of_trees", metric = "deviance")
plot(gbm, timestep = "number_of_trees", metric = "rmse")
plot(gbm, timestep = "number_of_trees", metric = "mae")

summary(model)

plot(model)
text(model)


# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(test[-2]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)


summary(y_pred)
# Making the Confusion Matrix
cm = table(test[, 2], y_pred)
cm


#--------------------------------------Random Forest-------------------------------------------------------------------------#
library(randomForest)

library(party)
library(caret)

set.seed(1000)
fit<-cforest(SalePrice~., data = train, controls = cforest_unbiased(ntree=1000,mtry = 2))
summary(fit)
cforestStats(fit)

#Sindhura RF Code
datasetRF = read.csv('train.csv')
dim(datasetRF)
set.seed(1000)
trainRF=sample(1:1460,1022,replace = FALSE)
traindataRF<-datasetRF[trainRF,]
testdataRF<-datasetRF[-trainRF,]
?cforest
fit<-cforest(SalePrice~.,data = train,controls = cforest_unbiased(ntree=1000 , mtry=2))
fit
cforestStats(fit)

#For Accuracy
pred <- predict(fit, newdata=test)
plot(pred)
table(pred, test$SalePrice)

cm=table(test[,2], pred)
cm

#Extra work
classifier = randomForest(x = train,
                          y = data$SalePrice,
                          ntree = 10)

y_pred = predict(classifier, newdata = test)

# Making the Confusion Matrix
cm = table(test[, 3], y_pred)
cm

model <- randomForest(SalePrice ~ . -LotArea, data = train)
randomForest(formula = SalePrice ~ . -LotArea, data = train) 
pred <- predict(model, newdata = test)
table(pred, test$SalePrice)

# Making the Confusion Matrix
cm = table(test[, 2], pred)
cm
#----------------------------------------------Decision Tree----------------------------------------------------------------#
#Decision Tree

install.packages("rpart")
library(rpart)
tree.model <- rpart(SalePrice ~ ScreenPorch + PoolArea	+ X3SsnPorch + EnclosedPorch +	OpenPorchSF +  MiscVal + MoSold+YrSold + GarageArea + GarageCars + GarageYrBlt, data=train)

summary(tree.model)

plot(tree.model)
text(tree.model)

install.packages("tree")
library(tree)

#tree.model <- tree(log(SalePrice) ~ ScreenPorch + PoolArea	+ X3SsnPorch + EnclosedPorch +	OpenPorchSF +  MiscVal + MoSold+YrSold + GarageArea + GarageCars + GarageYrBlt, data=train)

text(tree.model, cex=.75)
traindecission=sample(1:nrow(datasetRF), nrow(datasetRF)/1.5)

fit1=tree(SalePrice~ ScreenPorch + PoolArea	+ X3SsnPorch + EnclosedPorch +	OpenPorchSF +  MiscVal + MoSold+YrSold + GarageArea + GarageCars + GarageYrBlt,datasetRF,subset=traindecission)

#fit1=tree(SalePrice~ ScreenPorch + PoolArea	+ X3SsnPorch + EnclosedPorch +	OpenPorchSF +  MiscVal + MoSold+YrSold + GarageArea + GarageCars + GarageYrBlt, traindecission, subset=train)
summary(fit1)
plot(fit1)
text(fit1,pretty = 0)
cv_fit=cv.tree(fit1)
plot(cv_fit$size,cv_fit$dev,type='b')
prune_fit=prune.tree(fit1, best=4)
plot(prune_fit)
text(prune_fit, pretty=0)
pred=predict(fit1,test=datasetRF[-train,])
pred

#------------------------------------NeuralNet with Featured Values----------------------------------------------------------
nntrain <- neuralnet(SalePrice ~ ScreenPorch + PoolArea	+ X3SsnPorch + EnclosedPorch +	OpenPorchSF +  MiscVal + MoSold +  YrSold + GarageArea + GarageCars + GarageYrBlt,train ,hidden=15,lifesign = "minimal", linear.output=FALSE, threshold = 0.1)
plot(nntrain, rep="best")

#nntrain <- neuralnet(SalePrice ~ train[boruta.train],train ,hidden=15,lifesign = "minimal", linear.output=FALSE, threshold = 0.1)
#plot(nntrain, rep="best")

pr.nn.val <- compute(nntrain, test[,2:37])

# backnormalizing SalePrice variable
pr.nn.r <- pr.nn.val$net.result
pr.nn.r <- pr.nn.r*(max(data.na$SalePrice)-min(data.na$SalePrice))+min(data.na$SalePrice)
pr.nn.r <- round(pr.nn.r, digits = 0)
test.r <- (test$SalePrice)*(max(data.na$SalePrice)-min(data.na$SalePrice))+min(data.na$SalePrice)

pred <- predict(model, newdata = test)
table(pred, test$SalePrice)

# Making the Confusion Matrix
cm = table(test[, 2], pred)
cm


# MSE NN model
MSE.nn <- sum((test.r-pr.nn.r)^2)/nrow(test)
MSE.nn

# Visual plot NN computed values vs val dataset
plot(test.r,pr.nn.r,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')
text(100000,500000, "MSE =  11169560070", pch = 18)

#NeuralNet with important test data values
nntest <- neuralnet(SalePrice ~ ScreenPorch + PoolArea	+ X3SsnPorch + EnclosedPorch +	OpenPorchSF +  MiscVal + MoSold +  YrSold + GarageArea + GarageCars + GarageYrBlt,test ,hidden=20,lifesign = "minimal", linear.output=FALSE, threshold = 0.1)
plot(nntest, rep="best")

#-----------------------------------------Multi Linear Regressor-----------------------------------------------------------#
regressor = lm(formula = SalePrice ~  ScreenPorch + PoolArea	+ X3SsnPorch + EnclosedPorch +	OpenPorchSF +  MiscVal + MoSold +  YrSold + GarageArea + GarageCars + GarageYrBlt,
               data = train)
summary(regressor)

plot(regressor)
y_pred = predict(regressor, newdata = test[-2])
plot(y_pred)
#y_pred


# Making the Confusion Matrix
cm = table(test[, 2], y_pred)
cm

#---------------------------------------BackNormalizing--------------------------------------------------------------------#
# Normalizing test data set
maxs.te <- apply(data.na.te, 2, max)
mins.te <- apply(data.na.te, 2, min)
scaled.te <- as.data.frame(scale(data.na.te, center = mins.te, scale = maxs.te - mins.te))


# writing the formula since neuralnet library doesn't accept y ~ x1 + x2
feats <- names(train)
f <- as.formula(paste("SalePrice ~", paste(feats[!feats %in% c("SalePrice", "Id")], collapse = " + ")))
f

# Creating NN model using 36 numeric variables (excluding Id)
nn <- neuralnet(f, data = train, hidden = c(50,50), linear.output = TRUE,lifesign = "minimal", threshold = 0.1)
plot(nn, rep="best")

# predicting SalePrice using NN

pr.nn.val <- compute(nn, test[,2:37])

# backnormalizing SalePrice variable
pr.nn.r <- pr.nn.val$net.result
pr.nn.r <- pr.nn.r*(max(data.na$SalePrice)-min(data.na$SalePrice))+min(data.na$SalePrice)
pr.nn.r <- round(pr.nn.r, digits = 0)

test.r <- (test$SalePrice)*(max(data.na$SalePrice)-min(data.na$SalePrice))+min(data.na$SalePrice)

# MSE NN model
MSE.nn <- sum((test.r-pr.nn.r)^2)/nrow(test)
MSE.nn

# Visual plot NN computed values vs val dataset
plot(test.r,pr.nn.r,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')
text(100000,500000, "MSE =  1622003183", pch = 18)

# Computing value for test dataset
# predicting SalePrice using NN

pr.nn.te <- compute(nn, scaled.te[,2:37])

# backnormalizing SalePrice variable

pr.nn.tes <- pr.nn.te$net.result
pr.nn.tes <- pr.nn.tes*(max(data.na$SalePrice)-min(data.na$SalePrice))+min(data.na$SalePrice)
pr.nn.tes <- round(pr.nn.tes, digits = 0)

# writing output
submit <- data.frame(cbind(data.na.te$Id,pr.nn.tes))
colnames(submit) <- c("Id", "SalePrice")

write.csv(submit, file ="Predicted House SalesPrice(Train).csv", row.names = FALSE)





