
setwd("C:\\Users\\Padmini\\Google Drive\\Predict 422\\Final Project")
charity <- read.csv("charity.csv")

#EDA
summary(charity)
str(charity)



#check if there are missing values
count.na = sum(is.na(charity[charity$part=="train",]))
count.na 

count.na = sum(is.na(charity[charity$part=="valid",]))
count.na 

#distributions
#histograms of of continous predictors
for( p in c("damt","avhv","incm","inca","plow","npro","tgif","lgif","rgif","tdon","tlag","agif")){
  hist(charity[[p]],main = paste("Histogram of" , p),xlab = p)
}

#boxplots of donor amount by categorical variables
for( p in c("home","chld","hinc","genf","wrat")){
  boxplot(damt~get(p),charity,main = paste("BoxPlot of damt vs" , p))
}

#frequency table of donors vs categorical variables
for( p in c("home","chld","hinc","genf","wrat")){
  print(p)
  with(charity,print(table(donr,get(p))))
}

#scatter plot of continuous variables
pairs(damt~avhv+incm+inca+plow,data=charity[charity$part=="train",])
pairs(damt~tdon+tlag+agif+npro+tgif+lgif,data=charity[charity$part=="train",])

#correlation
cor(charity[charity$part=="train",c(11:21,23)])

#predictor transformations
charity.t <- charity
for( p in c("avhv","incm","inca","tgif","lgif","rgif","agif")){
  charity.t[[p]] <- log(charity.t[[p]])
  par(mfrow=c(1,2))
  hist(charity[[p]],main = paste("Histogram of " , p),xlab = p)
  hist(charity.t[[p]],main = paste("Histogram of log" , p),xlab = p)
  par(mfrow=c(1,1))
}

#charity.t$avhv <- log(charity.t$avhv)
#charity.t$tgif = log(charity.t$tgif)
#charity.t$lgif = log(charity.t$lgif)
#charity.t$rgif = log(charity.t$rgif)
#charity.t$incm = log(charity.t$incm)

# Make columns factors
#for( p in c("home","chld","hinc","genf","wrat")){
#  charity.t[[p]] <- as.factor(charity.t[[p]])
#}
#str(charity.t)

# set up data for analysis
data.train <- charity.t[charity$part=="train",]
x.train <- data.train[,2:21]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[,2:21]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,2:21]

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)

data.train.std.c$donr = as.factor(data.train.std.c$donr)
data.valid.std.c$donr = as.factor(data.valid.std.c$donr)

# Part2 : Classification Model
######## logistic regression


library(MASS)
library(class)
library(ISLR)

#using model from sample code
#AIC 2177.8
model.log1 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc 
                  + I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + tgif 
                  + lgif + rgif + tdon + tlag + agif, data.train.std.c, family=binomial("logit"))
summary(model.log1)

#Include only variable found significant in model.log1
#AIC AIC: 2170.8
model.log2 <- glm(donr ~ reg1 + reg2 + home + chld  
                  + I(hinc^2) + wrat  + incm + plow + tgif 
                  + tdon + tlag , data.train.std.c, family=binomial("logit"))
summary(model.log2)

#model validation and confusion matrix
glm.probs.log2 <- predict(model.log2, data.valid.std.c, type="response") # n.valid post probs
glm.pred.log2 = rep("0" ,2018)
glm.pred.log2[glm.probs.log2 > .5] = "1"
table(glm.pred.log2,c.valid)
glm.err.log2 = mean(glm.pred.log2 != c.valid)
glm.err.log2 #0.1199207

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log2 <- cumsum(14.5*c.valid[order(glm.probs.log2 , decreasing=T)]-2)
# see how profits change as more mailings are made
plot(profit.log2, main = "Maximum Profit - Logistic") 
# number of mailings that maximizes profits
n.mail.log2 <- which.max(profit.log2) 
# report number of mailings and maximum profit
c(n.mail.log2, max(profit.log2)) # 1348 11630

#cutoffs and check optimized mailing for max profit
cutoff.log2 <- sort(glm.probs.log2, decreasing=T)[n.mail.log2+1] # set cutoff based on number of mailings for max profit
chat.log2 <- ifelse(glm.probs.log2  > cutoff.log2, 1, 0) # mail to everyone above the cutoff
table(chat.log2, c.valid) # classification table
# correct predictions - 659+988/2018
log.donors = 360+988 #1348
log.profit = 988*14.5 - 2*1348 #11630

#######Logistic GAM
library(gam)
#AIC 2785.111
gam.model1 <- gam(I(donr == 1) ~ reg1 + reg2 + reg3 + reg4 + home
                  + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro
                  + tgif + lgif + rgif + tdon + tlag + agif, family = binomial,
                  data = data.train.std.c)
summary(gam.model1)

#using signficant variables from gam.model1 and smoothing splines
#AIC 2818.673
gam.model2 <- gam(I(donr == 1) ~  reg2 + home
                  + chld  + wrat + s(avhv,df=4) + s(incm,df=4)  + s(plow,df=4) + s(npro, df=4)
                  + s(tgif,df=4)  + s(tdon, df=4) + s(tlag,df=4) , family = binomial,
                  data = data.train.std.c)
summary(gam.model2)

# GAM model validation and confusion matrix
gam.probs.model2 = predict (gam.model2 , newdata = data.valid.std.c)
gam.pred.model2 <- rep("0", 2018)
gam.pred.model2[ gam.probs.model2 > .5] = "1"
table(gam.pred.model2, c.valid)
gam.err.model2 <- mean(gam.pred.model2 != c.valid)
gam.err.model2 #0.1635282

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.gam2 <- cumsum(14.5*c.valid[order(gam.probs.model2 , decreasing=T)]-2)
# see how profits change as more mailings are made
plot(profit.gam2, main = "Maximum Profit - GAM") 
# number of mailings that maximizes profits
n.mail.gam2 <- which.max(profit.gam2) 
# report number of mailings and maximum profit
c(n.mail.gam2, max(profit.gam2)) # 1489.0 11420.5

#cutoffs and check optimized mailing for max profit
cutoff.gam2 <- sort(gam.probs.model2, decreasing=T)[n.mail.gam2+1] # set cutoff based on number of mailings for max profit
chat.gam2 <- ifelse(gam.probs.model2  > cutoff.gam2, 1, 0) # mail to everyone above the cutoff
table(chat.gam2, c.valid) # classification table
#correct predictions 523+993/2018
gam.donors = 496+993 # 1489
gam.profit = 14.5*993 - 2*1489 #11420.5

#LDA

library(MASS)
#LDA with all variables
lda.model1 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c) 
lda.model1

lda.pred.model1 = predict(lda.model1, data.valid.std.c)
table(lda.pred.model1$class,c.valid)
lda.err.model1 = mean(lda.pred.model1$class != c.valid) #0.1313181
lda.err.model1 #0.1313181

lda.pred.model1.posterior <- lda.pred.model1$posterior[,2]
lda.pred.model1.posterior

#LDA with significant variables 
lda.model2 <- lda(donr ~ reg1 + reg2 + home + chld + I(hinc^2) + wrat  + incm + plow + tgif 
                  + tdon + tlag , data.train.std.c)
lda.model2

lda.pred.model2 = predict(lda.model2, data.valid.std.c)
table(lda.pred.model2$class,c.valid)
lda.err.model2 = mean(lda.pred.model2$class != c.valid) 
lda.err.model2 #0.1323092

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.lda2 <- cumsum(14.5*c.valid[order(lda.pred.model2$posterior[,2] , decreasing=T)]-2)
# see how profits change as more mailings are made
plot(profit.lda2, main = "Maximum Profit - LDA") 
# number of mailings that maximizes profits
n.mail.lda2 <- which.max(profit.lda2 ) 
# report number of mailings and maximum profit
c(n.mail.lda2, max(profit.lda2)) # 1406 11601

#cutoffs
cutoff.lda2 <- sort(lda.pred.model2$posterior[,2], decreasing=T)[n.mail.lda2+1] # set cutoff based on number of mailings for max profit
chat.lda2 <- ifelse(lda.pred.model2$posterior[,2]  > cutoff.lda2, 1, 0) # mail to everyone above the cutoff
table(chat.lda2, c.valid) # classification table
#correct predictions 607+994/2018
lda.donors = 412+994 # 1406
lda.profit = 994*14.5-2*1406 # 11601

#QDA

library(MASS)
#QDA with all variables
qda.model1 <- qda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c) 
qda.model1

qda.pred.model1 = predict(qda.model1, data.valid.std.c)
table(qda.pred.model1$class,c.valid)
qda.err.model1 = mean(qda.pred.model1$class != c.valid)
qda.err.model1 # 0.1590684

qda.pred.model1.posterior <- qda.pred.model1$posterior[,2]


#QDA with significant variables 
qda.model2 <- qda(donr ~ reg1 + reg2 + home + chld + I(hinc^2) + wrat  + incm + plow + tgif 
                  + tdon + tlag , data.train.std.c)
qda.model2

qda.pred.model2 = predict(qda.model2, data.valid.std.c)
table(qda.pred.model2$class,c.valid)
qda.err.model2 = mean(qda.pred.model2$class != c.valid) #0.4340932
qda.err.model2 # 0.1590684

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.qda2 <- cumsum(14.5*c.valid[order(qda.pred.model2$posterior[,2] , decreasing=T)]-2)
# see how profits change as more mailings are made
plot(profit.qda2, main = "Maximum Profit - QDA") 
# number of mailings that maximizes profits
n.mail.qda2 <- which.max(profit.qda2 ) 
# report number of mailings and maximum profit
c(n.mail.qda2, max(profit.qda2)) # 1402.0 11275.5

#cutoffs
cutoff.qda2 <- sort(qda.pred.model2$posterior[,2], decreasing=T)[n.mail.qda2+1] # set cutoff based on number of mailings for max profit
chat.qda2 <- ifelse(qda.pred.model2$posterior[,2]  > cutoff.qda2, 1, 0) # mail to everyone above the cutoff
table(chat.qda2, c.valid) # classification table
#correct predictions 588+971/2018
qda.donors = 431+971 #1402
qda.profit = 14.5*971-2*1402 # 11275.5

#KNN
# k = 3
library(class)

set.seed(1)
knn.pred.model1 = knn(x.train, x.valid, c.train, k=3)
table(knn.pred.model1, c.valid)
knn.err.model1 = mean(knn.pred.model1 != c.valid)
knn.err.model1 #  0.3320119

# k = 5
knn.pred.model2 = knn(x.train, x.valid, c.train, k=5)
knn.pred.prop = attributes(.Last.value)
table(knn.pred.model2, c.valid)
knn.err.model2 = mean(knn.pred.model2 != c.valid)
knn.err.model2 # 0.3280476
knn.donors = 443+780 # 1223
knn.profit = 14.5*780-2*1223 #8864


#### decision tree
library(tree)
tree.model = tree(donr~.,data.train.std.c)
plot(tree.model)
text(tree.model, pretty=0)

#cross validation and pruning
tree.cv.model <- cv.tree(tree.model, FUN=prune.misclass)
plot(tree.cv.model$size, tree.cv.model$dev, type = "b")
plot(tree.cv.model$k, tree.cv.model$dev, type = "b")

#tree with 15 nodes has lowest misclassification error.
#significant reduction in error seen with 5 nodes

#prune to 5 nodes

tree.prune.model1 = prune.misclass(tree.model, best=5)
tree.pred.model1 = predict(tree.prune.model1, data.valid.std.c, type="class")
table(tree.pred.model1, c.valid) #correct predictions 794+853
tree.err.model1 <- mean(tree.pred.model1 != c.valid)
tree.err.model1 # 0.1838454

tree.prune.model2 = prune.misclass(tree.model, best=15)
tree.pred.model2 = predict(tree.prune.model2, data.valid.std.c, type="class")
table(tree.pred.model2, c.valid) #correct predictions 783+929
tree.err.model2 <- mean(tree.pred.model2 != c.valid)
tree.err.model2 # 0.1516353
tree.donors = 236+929 # 1165
tree.profit = 14.5*929-2*1165 #11140.5

##### bagging
library(randomForest)
set.seed(1) # For reproducible results
bagging.model = randomForest(donr~., data=data.train.std.c, mtry=20, 
                                  importance=TRUE, type="classification")
importance(bagging.model)

set.seed(1)
bagging.pred.model = predict(bagging.model, newdata=data.valid.std.c)
# Correct predictions 889+904/2018 .
table(bagging.pred.model, c.valid)
bagging.err.model = mean(bagging.pred.model != c.valid)
bagging.err.model #0.1114965
bagging.donors = 130+904 #1034
bagging.profit = 14.5*904-2*1034 #11040

##### random forest
library(randomForest)
set.seed(1) # For reproducible results
randomforest.model = randomForest(donr~., data=data.train.std.c, mtry=5, 
            importance=TRUE, type="classification")
importance(randomforest.model)

set.seed(1)
randomforest.pred.model = predict(randomforest.model, newdata=data.valid.std.c)
# Correct predictions 878+915/2018 donors.
table(randomforest.pred.model, c.valid)
randomforest.err.model = mean(randomforest.pred.model != c.valid)
randomforest.err.model #0.1114965
randomforest.donors = 141+915 # 1056
randomforest.profit = 14.5*915-2*1056 #11155.5


### Boosting
library(gbm)
set.seed(1)
boost.model = gbm(donr ~ reg1 + reg2 + home + chld  
                  + I(hinc^2) + wrat  + incm + plow + tgif 
                  + tdon + tlag , data = data.train, 
                  distribution = "bernoulli", n.trees = 5000, interaction.depth = 4)
#chld, hinc^2, reg2 and home are most importart variables
summary(boost.model)

set.seed(1)
boost.prob.model = predict.gbm(boost.model, newdata = data.valid,n.trees = 5000, type = "response")
boost.pred.model = rep("0", 2018)
boost.pred.model[boost.prob.model> .5] = "1"
table(boost.pred.model , c.valid) # correct prediction 877+920/2081
boost.err.model <- mean(boost.pred.model != c.valid)
boost.err.model #0.1095144

profit.boost <- cumsum(14.5*c.valid[order(boost.prob.model , decreasing=T)]-2)
# see how profits change as more mailings are made
plot(profit.boost, main = "Maximum Profit - Boosting") 
# number of mailings that maximizes profits
n.mail.boost <- which.max(profit.boost ) 
# report number of mailings and maximum profit
c(n.mail.boost, max(profit.boost)) # 1244 11838

#cutoffs
cutoff.boost <- sort(boost.prob.model, decreasing=T)[n.mail.boost+1] # set cutoff based on number of mailings for max profit
chat.boost <- ifelse(boost.prob.model  > cutoff.boost, 1, 0) # mail to everyone above the cutoff
table(chat.boost, c.valid) # classification table
#249+987/2018
boost.donors = 256+988 #1244
boost.profit = 14.5*988-2*1244 # 11838

# View Error table for best models.
error <- c(round(glm.err.log2,4), round(gam.err.model2,4), round(lda.err.model1,4), 
           round(qda.err.model2,4),round(knn.err.model2,4), round(tree.err.model2,4),
           round(bagging.err.model,4),round(randomforest.err.model,4),round(boost.err.model,4))
model = c("Logistic", "GAM", "LDA", "QDA", "KNN", "Tree", "Bagging", "RandomForest", "Boosting")
profit = c(log.profit, gam.profit, lda.profit, qda.profit, 
           knn.profit, tree.profit, bagging.profit, randomforest.profit, boost.profit)
donors = c(log.donors, gam.donors, lda.donors, qda.donors, 
           knn.donors,tree.donors,bagging.donors,randomforest.donors,boost.donors)
errtbl <- as.data.frame(cbind(model, error, profit, donors))
errtbl
which.max(profit) # max profit for boosting model $11838 with mailing to 1244 donors
which.min(error)  # min error for boosting model 0.1095


# predictions on test data set
# select model.log2 since it has maximum profit in the validation sample

post.test <- predict(model.log2, data.test.std, type="response") # post probs for test data

# Oversampling adjustment for calculating number of mailings for test set

n.mail.valid <- which.max(profit.log2)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test) #367 mailings to be sent to potential donors with test set
