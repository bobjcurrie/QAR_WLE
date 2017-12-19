library(mlbench); library(caret); library(dplyr); library(e1071); library(MASS); library(parallel); library(doParallel)

set.seed(1234)

## download files from web
linkTraining <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
linkTesting <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(url = linkTraining, destfile = "pml-training.csv")
download.file(url = linkTesting, destfile = "pml-testing.csv")

## read downloaded CSV files into R dataframes
qarwle <- read.csv("pml-training.csv", header = TRUE, stringsAsFactors = FALSE)

dim(qarwle)
str(qarwle)

testing <- read.csv("pml-testing.csv", header = TRUE, stringsAsFactors = FALSE)
dim(testing)
str(testing)

qarwle$classe <- as.factor(qarwle$classe)
qarwle$user_name <- as.factor(qarwle$user_name)
qarwle$new_window <- as.factor(qarwle$new_window)

for (i in c(8:159)) {
    qarwle[,i] <- as.numeric(qarwle[,i])
}

#use 75% of data set for this example
inTraining <- createDataPartition(qarwle$classe, p = .20, list = FALSE)
qarwle <- qarwle[inTraining, ]

qarwle <- qarwle[qarwle$new_window == "yes", ]


classe <- qarwle$classe
table(classe)

table(qarwle$user_name, qarwle$classe)

#only belt measurements (cols) for this example
include <- c("belt", "arm", "dumbbell")
qarwle <- qarwle[grep(paste(include, collapse = "|"), names(qarwle))]
#qarwle <- qarwle[grep("+belt", names(qarwle))]

qarwle <- qarwle[,-c(1:7)]

qarwle$classe <- classe

#also let's remove the calculated vars for each complete window of the exercise
#kurtosis, skewness, max, min, amplitude, var, avg, std_dev
remove <- c("kurtosis", "skewness", "max", "min", "amplitude", "var", "avg", "stddev")
qarwle <- qarwle[-grep(paste(remove, collapse = "|"), names(qarwle))]

#build the rf model

#training and testing data sets
inTraining <- createDataPartition(qarwle$classe, p = .75, list = FALSE)
training <- qarwle[inTraining,]
testing <- qarwle[-inTraining,]

dim(training); dim(testing)

#use x / y syntax
x <- training[, -53] #-53
y <- training[, 53] #53


corMatrix <- cor(qarwle[, -53])
print(corMatrix)
highlyCorrelated <- findCorrelation(corMatrix, cutoff = 0.8)
CorrelatedVars <- names(qarwle)[highlyCorrelated]
print(CorrelatedVars)

qarwle <- qarwle[, -which(names(qarwle) %in% CorrelatedVars)]

#configure trainControl object
rfModelControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

#configure parallel processing
cluster <- makeCluster(detectCores() - 1) # leave 1 core for OS
registerDoParallel(cluster)

#develop training model
system.time(rfModel <- train(x, y, method = "rf", data = qarwle, trControl = rfModelControl))
svm model
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


varImp(rfModel, scale = FALSE)
predictors(rfModel)
plot(rfModel)

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
