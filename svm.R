library( 'e1071' ); 
library(caret);
library(MASS)

set.seed(1234)

data( iris )
model <- svm( iris$Species~., iris )
res <- predict( model, newdata=iris )
confusionMatrix(res, iris$Species)

x <- iris[, -5]
y <- iris[, 5]

rfModel <- train(x, y, method = "rf", data = iris)

rfModel
rfModel$resample
rfModel$finalModel
confusionMatrix.train(rfModel)

res2 <- predict(rfModel, newdata = iris)
confusionMatrix(res2, iris$Species)

## LDA
LDAModel <- lda(formula = iris$Species ~., data = iris)
resLDA <- predict(LDAModel, newdata = iris)
confusionMatrix(res, iris$Species)

