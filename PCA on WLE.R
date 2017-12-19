library(caret); library(kernlab); data(spam)

pml <- read.csv("pml-training.csv", header = TRUE, stringsAsFactors = FALSE)

## Split data into training and testing set
inTrain <- createDataPartition(y = pml$classe, p=0.75, list = FALSE)

training = pml[inTrain, ]
testing <- pml[-inTrain, ]

dim(training)

set.seed(32343)
modelFit <- train(classe ~., data=training, method = "glm")
modelFit
modelFit$finalModel

predictions <- predict(modelFit, newdata = testing)

predictions

confusionMatrix(predictions, testing$type)

## Use cross-validation to split training set, e.g. k-folds
set.seed(32343)

folds <- createFolds(y=spam$type, k=10, list = TRUE, returnTrain = TRUE) # return training set

sapply(folds, length)

folds[[1]][1:10]

## Resampling
set.seed(32343)

folds <- createResample(y=spam$type, times = 10, list = TRUE)

sapply(folds, length)

folds[[1]][1:10]

##

args(trainControl)

## Methods:
# boot = bootstrapping
# boot632 = bootstrapping with adjustment
# cv = cross validation
# repeatedcv = repeated cross validation
# LOOCV = leave one out cross validation

## Preprocessing with PCA

# Check for highly correlated quantitative vriables
M <- abs(cor(training[,-58]))
M == 1
diag(M) <- 0
which(M > 0.8, arr.ind = T)
names(spam)[c(32,34,40)]
plot(spam[,34], spam[,32])
plot(spam[,40], spam[,32])

smallSpam <- spam[,c(34, 32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[,1], prComp$x[,2])

prComp$rotation

typeColor <- ((spam$type == "spam")*1 + 1)  # black if not spam(ham), red if spam
prComp <- prcomp(log10(spam[, -58]+1))      #log transformation + 1 to make skewed data look more gaussian
plot(prComp$x[,1], prComp$x[,2], col = typeColor, xlab = "PC1", ylab = "PC2")

# PCA done with caret
preProc <- preProcess(log10(spam[,-58]+1), method = "pca", pcaComp = 2)
spamPC <- predict(preProc, log10(spam[,-58]+1))
plot(spamPC[,1], spamPC[,2], col = typeColor)

# Preprocessing with PCA
preProc <- preProcess(log10(training[,-58]+1), method = "pca", pcaComp = 2)
trainPC <- predict(preProc, log10(training[,-58]+1))
modelFit <- train(training$type ~., method = "glm", data = trainPC)
# error?

## After all that...
modelFit <- train(type ~., method = "glm", preProcess = "pca", data=training)

confusionMatrix(testing$type, predict(modelFit, testing))
 
## PCA works best with linear models
