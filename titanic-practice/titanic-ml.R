library(ggplot2)
library(dplyr)
library(stringr)

#load data
train<-read.csv("train.csv",header=TRUE)
test<-read.csv("test.csv",header=TRUE)

#combined both datasets to clean the whole data
test.survived<-data.frame(Survived=rep("none",nrow(test)),test[,])
combined.data<-rbind(train,test.survived)

str(combined.data)
combined.data$Pclass<-as.factor(combined.data$Pclass)
combined.data$Survived<-as.factor(combined.data$Survived)
combined.data$Name<-as.character(combined.data$Name)
combined.data$Ticket<-as.character(combined.data$Ticket)
combined.data$Cabin<-as.character(combined.data$Cabin)
str(combined.data)
head(combined.data)
table(combined.data$Survived)

#graph Pclass ~ survived
ggplot(combined.data[1:891,], aes(x=Pclass,fill=Survived)) +
  geom_bar(width=0.5) +
  xlab("Pclass") +
  ylab("Total Count") +
  labs("Survived")

#analyzing name feature
length(unique(combined.data$Name))
#find duplicates
duplicate.names<-combined.data[which(duplicated(combined.data$Name)),"Name"]
combined.data[which(combined.data$Name %in% duplicate.names),]

#Hypothesis: title could predict survivability
#extract title from name
misses<-combined.data[which(str_detect(combined.data$Name,"Miss.")),]
ggplot(misses, aes(x=Pclass, fill=Survived)) +
  geom_bar()
mrses<-combined.data[which(str_detect(combined.data$Name,"Mrs.")),]
mres<-combined.data[which(str_detect(combined.data$Name,"Mr.")),]
masters<-combined.data[which(str_detect(combined.data$Name,"Master.")),]
head(masters)

#create a new feature based on the title
extractTitle<-function(name){
  if(length(grep("Miss.", name))>0) {
    return("Miss")
  } else if (length(grep("Mrs.",name))>0) {
    return("Mrs")
  } else if (length(grep("Mr.",name))>0) {
    return("Mr")
  } else if (length(grep("Master.", name))>0) {
    return("Master")
  } else {
    return("Other")
  }
}
titles<-NULL
for (i in 1:nrow(combined.data)) {
  titles<-c(titles, extractTitle(combined.data[i,"Name"]))
}
combined.data$Title<-as.factor(titles)

#plot new feature title
ggplot(combined.data[1:391,],aes(x=Title,fill=Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) +
  xlab("Title") +
  ylab("Total count") +
  labs("Survived")

#find if title is a better feature than age and sex
ggplot(combined.data,aes(x=Age)) +
  geom_histogram() +
  facet_wrap(~Title)

summary(combined.data$Age)
summary(combined.data[1:981,]$Age)
ggplot(combined.data[1:891,],aes(x=Age, fill=Survived)) +
  geom_bar() +
  facet_wrap(~Title + Pclass) +
  xlab("Age") +
  ylab("Total count") +
  ggtitle("Age analysis")
ggplot(combined.data[1:891,],aes(x=Age, fill=Survived)) +
  geom_bar(binwidth = 10) +
  facet_wrap(~Sex + Pclass) +
  xlab("Age") +
  ylab("Total count") +
  ggtitle("Age analysis")

summary(masters$Age)
summary(misses$Age)

#detail misses feature
ggplot(misses[misses$Survived!="none",],aes(x=Age, fill=Survived))+
  geom_histogram(binwidth = 5) +
  facet_wrap(~Pclass)

#explore further misses title
misses.alone<-misses[which(misses$SibSp==0 & misses$Parch==0),]
summary(misses.alone$Age)
#how many misses alone are younger than 14.5 (the value is based on maximum value for masters)
length(which(misses.alone$Age<14.5))

#explore Sibsp feature
summary(combined.data$SibSp)

length(unique(combined.data$SibSp))
combined.data$SibSp<-as.factor(combined.data$SibSp)
combined.data$Parch<-as.factor(combined.data$Parch)
ggplot(combined.data[1:891,],aes(x=Parch, fill=Survived)) +
  geom_bar() +
  facet_wrap(~Pclass + Title)
#feature engineering family size feature
#since SibSp and Parch were made factors, we need the original vectors
temp.SibSp<-c(train$SibSp,test$SibSp)
temp.Parch<-c(train$Parch, test$Parch)
combined.data$FamilySize<-as.factor(temp.SibSp+temp.Parch+1) 
ggplot(combined.data[combined.data$Survived!="none",], aes(x=FamilySize, fill=Survived))+
  geom_bar() +
  facet_wrap(~Title + Pclass) +
  xlab("Family size") +
  ylab("Total count") +
  ggtitle("Family size by Pclass and Title")

#explore the ticket feature
str(combined.data$Ticket)
combined.data$Ticket[1:20]
ticket.first.char<-substr(combined.data$Ticket,0,1)
unique(ticket.first.char)
combined.data$ticketFirstChar<-as.factor(ticket.first.char)
ggplot(combined.data[1:891,],aes(x=ticketFirstChar, fill=Survived)) +
  geom_bar() +
  facet_wrap(~Pclass+Sex) +
  xlab("TicketChar") +
  ylab("count") +
  ggtitle("Ticket first Char by Pclass")

#explore the fare feature
ggplot(combined.data[1:891,], aes(x=Fare)) +
  geom_histogram() +
  facet_wrap(~Pclass) +
  xlim(0,300)

#explore the cabin feature
str(combined.data$Cabin)
combined.data$Cabin[1:100]
combined.data[which(combined.data$Cabin==""),"Cabin"]<-"U"
combined.data$cabinFirstChar<-substr(combined.data$Cabin,0,1)
combined.data$cabinFirstChar[1:100]
combined.data$cabinFirstChar<-as.factor(combined.data$cabinFirstChar)
levels(combined.data$cabinFirstChar)
ggplot(combined.data[1:891,], aes(x=cabinFirstChar, fill=Survived)) +
  geom_bar() +
  facet_wrap(~Pclass+Sex) +
  ggtitle("cabin by family size survivability")
combined.data$multipleCabin<-as.factor(ifelse(str_detect(combined.data$Cabin," "),"Y","N"))
summary(combined.data$multipleCabin)
ggplot(combined.data[1:891,], aes(x=multipleCabin,fill=Survived)) +
  geom_bar() +
  facet_wrap(~FamilySize) +
  ylim(0,10) +
  ggtitle("multiple cabins survivability")

#explore embarked feature
str(combined.data$Embarked)
ggplot(combined.data[1:891,], aes(x=Embarked, fill=Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) +
  ggtitle("Embarked Survivability")

library(randomForest)
#train a randomForest with the default parameters using Pclass & Title
rf.train.1<-combined.data[1:891,c("Pclass","Title")]
rf.label<-as.factor(train$Survived)
set.seed(1234) #to always have the same parameter
rf.1<-randomForest(x=rf.train.1,y=rf.label,importance = TRUE, ntree = 1000) #default ntree is 500
rf.1
varImpPlot(rf.1)
#train a randomForest using Pclass, Title, SibSp
rf.train.2<-combined.data[1:891,c("Pclass", "Title", "SibSp")]
set.seed(1234)
rf.2<-randomForest(x=rf.train.2, y=rf.label, importance = TRUE, ntree=1000)
rf.2
varImpPlot(rf.2)
#train a randomForest using Pclass, Title, Parch
rf.train.3<-combined.data[1:891,c("Pclass","Title", "Parch")]
set.seed(1234)
rf.3<-randomForest(x=rf.train.3, y=rf.label, importance = TRUE, ntree=1000)
rf.3
varImpPlot(rf.3)
#train a randomForest using Pclass, Title, SibSp, Parch
rf.train.4<-combined.data[1:891, c("Pclass", "Title", "SibSp", "Parch")]
set.seed(1234)
rf.4<-randomForest(x=rf.train.4, y=rf.label, importance = TRUE, ntree=1000)
rf.4
varImpPlot(rf.4)
#train a randomForest using Pclass, Title, FamilySize
rf.train.5<-combined.data[1:891,c("Pclass", "Title", "FamilySize")]
set.seed(1234)
rf.5<-randomForest(x=rf.train.5, y=rf.label, importance = TRUE, ntree=1000)
rf.5
varImpPlot(rf.5)
#train a randomForest using Pclasss, Title, FamilySize, SipSp
rf.train.6<-combined.data[1:891,c("Pclass", "Title", "FamilySize", "SibSp")]
set.seed(1234)
rf.6<-randomForest(x=rf.train.6, y=rf.label, importance = TRUE, ntree=1000)
rf.6
varImpPlot(rf.6)
#train a randomForest using Pclass, Title, FamilySize, Parch
rf.train.7<-combined.data[1:891,c("Pclass", "Title", "FamilySize", "Parch")]
set.seed(1234)
rf.7<-randomForest(x=rf.train.7, y=rf.label, importance = TRUE, ntree=1000)
rf.7
varImpPlot(rf.7)

#Subset the test records and features selected in the model
test.submit.df<-combined.data[892:1309,c("Pclass", "Title", "FamilySize")]

#make predictions
rf.5.preds<-predict(rf.5,test.submit.df)
table(rf.5.preds)

#write a CSV
submit.df<-data.frame(PassengerID=rep(892:1309),Survived=rf.5.preds)
write.csv(submit.df,file="titanic_sub_1.csv",row.names = FALSE)

#cross-validation
library(caret)
library(doSNOW)

#create 10 Fold CV repeated 10 times -> 100 total folds
set.seed(2348)
cv.10.folds<-createMultiFolds(rf.label,k=10,times=10)

#check stratification
table(rf.label)
table(rf.label[cv.10.folds[[33]]])
308/494
342/549 
#set up Caret's trainControl
ctrl.1<-trainControl(method = "repeatedcv", number=10, repeats=10, index=cv.10.folds)

#set doSnow package for multi-core training since we are training 100.000 trees
cl<-makeCluster(6,type = "SOCK")
registerDoSNOW(cl)

#set seed
set.seed(34324)
rf.5.cv.1<-train(x=rf.train.5, y=rf.label, method="rf", tunelength=3, 
                 ntree=1000, trControl=ctrl.1)
#shutdown cluster
stopCluster(cl)

#check results
rf.5.cv.1

#set the train model with less folds, so we use less data and prevent overfitting
set.seed(2543)
cv.5.folds<-createMultiFolds(rf.label, k=5, times=10)
ctrl.2<-trainControl(method = "repeatedcv", number=5, repeats=10, index=cv.5.folds)
cl<-makeCluster(6,type = "SOCK")
registerDoSNOW(cl)
set.seed(89472)
rf.5.cv.2<-train(x=rf.train.5, y=rf.label, method="rf", tunelength=3, 
                 ntree=1000, trControl=ctrl.2)
stopCluster(cl)
rf.5.cv.2
#set train model with 3 folds
set.seed(3423)
cv.3.folds<-createMultiFolds(rf.label, k=3, times=10)
ctrl.3<-trainControl(method = "repeatedcv", number=3, repeats=10, index=cv.3.folds)
cl<-makeCluster(6,type = "SOCK")
registerDoSNOW(cl)
set.seed(54252)
rf.5.cv.3<-train(x=rf.train.5, y=rf.label, method="rf", tunelength=3, 
                 ntree=1000, trControl=ctrl.3)
stopCluster(cl)
rf.5.cv.3

#using a single decision tree
library(rpart)
library(rpart.plot)

#create utility function
rpart.cv<-function(seed, training, labels, ctrl) {
  cl<-makeCluster(6,type = "SOCK")
  registerDoSNOW(cl)
  set.seed(seed)
  rpart.cv<-train(x=training, y=labels, method="rpart", tunelength=30, 
                   trControl=ctrl)
  stopCluster(cl)
  return(rpart.cv)
}
features<-c("Pclass", "Title", "FamilySize")
rpart.train.1<-combined.data[1:891,features]
rpart.1.cv.1<-train(x=rpart.train.1, y=rf.label, method="rpart", tunelength=3, trControl=ctrl.3)
rpart.1.cv.1<-rpart.cv(94622,rpart.train.1, rf.label,ctrl.3)
rpart.1.cv.1
#not working rpart training
