library(mlbench); library(caret)

data("Sonar")

set.seed(95014)

#training & testing data sets

inTraining <- createDataPartition(Sonar$Class, p = .75, list = FALSE)
training <- Sonar[inTraining,]
testing <- Sonar[-inTraining,]

#use x / y syntax
x <- training[, -61]
y <- training[, 61]

#configure parallel processing
library(parallel); library(doParallel)
cluster <- makeCluster(detectCores() - 1) # leave 1 core for OS
registerDoParallel(cluster)

#configure trainControl object
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

#develop training model
system.time(fit <- train(x, y, method = "rf", data = Sonar, trControl = fitControl))

#stop the cluster nad return R to single threaded processing
stopCluster(cluster)
registerDoSEQ()

fit
fit$resample
fit$finalModel
confusionMatrix.train(fit)

#predict using test data set
predictions <- predict(fit, testing)
predictions
confusionMatrix(predictions, testing$Class)
