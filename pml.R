library(mlbench); library(caret); library(dplyr); library(e1071); library(MASS); library(parallel); library(doParallel)

set.seed(1234)

## download files from web
linkTraining <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
linkTesting <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

download.file(url = linkTraining, destfile = "pml-training.csv")
download.file(url = linkTesting, destfile = "pml-testing.csv")

## read downloaded CSV files into R dataframes
qar_wle <- read.csv("pml-training.csv", header = TRUE, stringsAsFactors = FALSE)

dim(qar_wle)
str(qar_wle)

qar_testing <- read.csv("pml-testing.csv", header = TRUE, stringsAsFactors = FALSE)
dim(qar_testing)
str(qar_testing)

qar_wle$classe <- as.factor(qar_wle$classe)
qar_wle$user_name <- as.factor(qar_wle$user_name)
qar_wle$new_window <- as.factor(qar_wle$new_window)

for (i in c(8:159)) {
    qar_wle[,i] <- as.numeric(qar_wle[,i])
}

qar_wle <- qar_wle[qar_wle$new_window == "yes",]

#use 75% of data set for this example
inTraining <- createDataPartition(qar_wle$classe, p = .20, list = FALSE)
qar_wle <- qar_wle[inTraining, ]

qar_wle <- qar_wle[qar_wle$new_window == "yes", ]


classe <- qar_wle$classe
table(classe)

#only belt measurements (cols) for this example
include <- c("belt", "arm", "dumbbell")
qar_wle <- qar_wle[grep(paste(include, collapse = "|"), names(qar_wle))]
#qar_wle <- qar_wle[grep("+belt", names(qar_wle))]
qar_wle$classe <- classe

#also let's remove the calculated vars for each complete window of the exercise
#kurtosis, skewness, max, min, amplitude, var, avg, std_dev
remove <- c("kurtosis", "skewness", "max", "min", "amplitude", "var", "avg", "stddev")
qar_wle <- qar_wle[-grep(paste(remove, collapse = "|"), names(qar_wle))]

#build the rf model

#training and testing data sets
inTraining <- createDataPartition(qar_wle$classe, p = .75, list = FALSE)
training <- qar_wle[inTraining,]
testing <- qar_wle[-inTraining,]

dim(training); dim(testing)

#use x / y syntax
x <- training[, -53]
y <- training[, 53]

#configure trainControl object
rfModelControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

#configure parallel processing
cluster <- makeCluster(detectCores() - 1) # leave 1 core for OS
registerDoParallel(cluster)

#develop training model
system.time(rfModel <- train(x, y, method = "rf", data = qar_wle, trControl = rfModelControl))

#svm model
system.time(svmModel <- svm(training$classe ~ ., training))

#LDA model
system.time(ldaModel <- lda(training$classe ~., training))

#stop the cluster nad return R to single threaded processing
stopCluster(cluster)
registerDoSEQ()

rfModel
rfModel$resample
rfModel$finalModel
confusionMatrix.train(rfModel)

#predict using test data set
rfPredictions <- predict(rfModel, testing)
rfPredictions
confusionMatrix(rfPredictions, testing$classe)

svmPredictions <- predict(svmModel, testing)
svmPredictions
confusionMatrix(svmPredictions, testing$classe)

ldaPredictions <- predict(ldaModel, testing)
ldaPredictions
confusionMatrix(ldaPredictions$class, testing$classe)

# combine models using simple majority vote
predictions <- data.frame(rfPredictions, svmPredictions, ldaPredictions$class)

# Quiz
finalPredict <- predict(rfModel, wle_testing)
finalPredict
