library(caret);
library(dplyr); library(rattle)

set.seed(1234)

training <- read.csv("pml-training.csv", header = TRUE, stringsAsFactors = FALSE)
dim(training)
str(training)

testing <- read.csv("pml-testing.csv", header = TRUE, stringsAsFactors = FALSE)
dim(testing)

training$classe <- as.factor(training$classe)
training$new_window <- as.factor(training$new_window)

table(training$classe)

str(training)

summary(training$new_window)

tail(subset(training, new_window == "yes"))[, c(6:15, 160)]

str(training$kurtosis_yaw_belt)

## remove all columns starting with min_, max_ kurtosis_, skewness_, amplitude
names(training)

drop <- c("^avg_", "^min_", "^max_")

## ??
##training[, -(grep("^avg_", x))]


training <- subset(training, new_window == "no")[, c(6:15, 160)]

setdiff(names(training), names(testing))

features <- names(training[,-160])

trainSmall <- training[,c(8:11, 37:48, 60:68, 84:86, 113:124, 151:160)]

beltSmall <- training[,c(2:45, 160)]

names(beltSmall)

beltSmall <- beltSmall[, -c(1:6, 11:35)]

corBeltMatrix <- cor(beltSmall[, -14])

highlyCorrelated <- findCorrelation(corBeltMatrix, cutoff=0.5, names = TRUE)

beltSmall <- beltSmall[, !(names(beltSmall) %in% highlyCorrelated)]

print(highlyCorrelated)



diag(corBeltMatrix) <- 0

corBeltdf <- as.data.frame(as.table(corBeltMatrix))

subset(corBeltdf, abs(Freq) > 0.5)

plot(beltSmall$accel_belt_x, beltSmall$accel_belt_y)
plot(beltSmall$roll_belt, beltSmall$accel_belt_z, col = "beltSmall$classe")
plot(beltSmall$total_accel_belt, beltSmall$roll_belt)

## Split data into training and testing set
inTrain <- createDataPartition(y = trainSmall$classe, p=0.75, list = FALSE)

trainBelt <- beltSmall[inTrain, ]
testBelt <- beltSmall[-inTrain, ]

train <- trainSmall[inTrain, ]
test <- trainSmall[-inTrain,]

trainAll <- training[inTrain, ]

dim(trainBelt); dim(testBelt)
dim(train); dim(test)

library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

control <- trainControl(method = "cv", number = 3) #5
system.time(model1 <- train(classe ~ ., data = train, preProcess = "pca", method = "rpart", trControl = control))

system.time(model1 <- train(classe ~ ., data = train, method = "rpart", trControl = control))

stopCluster(cluster)
registerDoSEQ()

model1
model1$finalModel
fancyRpartPlot(model1$finalModel)

predictions <- predict(model1, test)
predictions
confusionMatrix(predictions, test$classe)

plot(varImp(model1, scale = FALSE))

predictors(model1)

system.time(model2 <- train(classe ~., data = train, preProcess = "pca", method = "rf"))
predictions <- predict(model2, testBelt)
predictions
confusionMatrix(predictions, testBelt$classe)
