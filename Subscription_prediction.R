getwd()
setwd("/Users/harshitmehta/Desktop/ISBF_course/Machine_Learning/UoL_assignment")

# loading all the libraries

library(dplyr)  
library(ggplot2)
library(lattice)
library(glmnet)
library(ROSE)

bank=read.table("bank.csv",sep=";",header=TRUE)

head(bank)

######################### Data Cleaning & Preparation ##############################

# to check if there are any missing values
any(is.na(bank))
# Thus we have no missing values in the data set.

colnames(bank)

glimpse(bank)

summary(bank)

# The following variables need to be removed from the dataset as they are not useful 
# for analysis purpose :
# pdays : 75% of the values are -1 (not previously contacted)
# previous : 75% of the values are 0 (75% of clients never contacted before)
# poutcome : 87% of the observations fall in unknown/other category
# durartion : can be known only after making the call - not useful for prediction purposes

dropped_cols = c("pdays", "previous", "poutcome", "duration")
bank_df = bank[,! (names(bank) %in% dropped_cols)]

summary(bank_df)

########################################## EDA ####################################


plot(bank_df$y, bank_df$age, xlab = "Outcome", ylab = "Age")
#mosaicplot(job~y, data = bank_df, xlab = "Job", ylab = "Outcome")
spineplot(y~job, data = bank_df, xlab = "Job", ylab = "Outcome")
job_set1 = c("admin.","blue-collar","entrepreneur","housemaid")
job_set2 = c("self-employed","services","student","technician")
job_set3 = c("management","retired","unemployed","unknown")
spineplot(y~job, data = bank_df[(bank_df$job %in% job_set1),], xlab = "Job", ylab = "Outcome")
spineplot(y~job, data = bank_df[(bank_df$job %in% job_set2),], xlab = "Job", ylab = "Outcome")
spineplot(y~job, data = bank_df[(bank_df$job %in% job_set3),], xlab = "Job", ylab = "Outcome")
job_effect = table(bank_df$job,bank_df$y)
job_frame = as.data.frame(job_effect)
t1 = job_frame[1:12,]
t2 = job_frame[13:24,]
job_frame = merge(t1, t2, by="Var1")
names(job_frame)[names(job_frame) == "Var1"] <- "Job"
names(job_frame)[names(job_frame) == "Freq.x"] <- "no"
names(job_frame)[names(job_frame) == "Freq.y"] <- "yes"
job_frame = job_frame[,c(1,3,5)]
job_frame$percent_yes = round(job_frame$yes/(job_frame$yes+job_frame$no),4)*100
plot(job_frame$Job, job_frame$percent_yes, xlab = "Job", ylab = "Percent yes")
# Students and retired people are more likely to subscribe to the bank product.
# Blue-collar proffessionals least likely to subscribe

# martital status effect on subscription
table(bank_df$marital,bank_df$y)
spineplot(y~marital, data = bank_df, xlab = "Marital Status", ylab = "Outcome")
# married people less likely to invest in long-term deposits

# education effect on subscription
table(bank_df$education,bank_df$y)
spineplot(y~education, data = bank_df, xlab = "Education Level", ylab = "Outcome")
# Customers with tertiary level of education most likely to invest in long-term deposits

# default effect on subscription
table(bank_df$default,bank_df$y)
spineplot(y~default, data = bank_df, xlab = "Default", ylab = "Outcome")
# default does not look like an important variable

# bank balance effect on subscription
plot(bank_df$y, log10(bank_df$balance), xlab = "Outcome", ylab = "Balance")
summary(bank_df[bank_df$y=="yes",]$balance)
summary(bank_df[bank_df$y=="no",]$balance) 
# people subscribing to long term generally have more bank balance

# housing loan effect on subscription
summary(bank_df$housing)
table(bank_df$housing,bank_df$y)
spineplot(y~housing, data = bank_df, xlab = "Hosuing Loan", ylab = "Outcome")
# people who do not have housing loan are more likely to invest in long-term loans

# personal loan effect on subscription
summary(bank_df$loan)
table(bank_df$loan,bank_df$y)
spineplot(y~loan, data = bank_df, xlab = "Personal Loan", ylab = "Subscription Outcome")
# people who do not have personal loan are more likely to invest in long-term loans

# number of contacts persormed's effect on subscription
summary(bank_df$campaign)
plot(bank_df$y, log2(bank_df$campaign), xlab = "Outcome", ylab = "Campaign contacts (log2)")
summary(bank_df[bank_df$y=="yes",]$campaign)
summary(bank_df[bank_df$y=="no",]$campaign) 
# 50% of the customers who subscribed did so after 2 calls.


###################################### Test/Train Split ##############################

set.seed(1)
train = sample(1:nrow(bank_df), 3164)

########################################## Modelling #################################


# Logistic Regression 1
glm.fit <- glm(y~., data = bank_df, subset = train, family = binomial)
glm.probs = predict(glm.fit, newdata = bank_df[-train,], type="response")
glm.pred = ifelse(glm.probs>0.5, "yes","no")
actual = bank_df[-train,]$y
# predicting almost all as "no"
mean(glm.pred==actual)
# 87% accuracy - but it is basically labelling everything as "no"
confusion_matrix1 <- table(glm.pred, actual)
confusion_matrix1
cat("Accuracy of Logistic Regression : ",((confusion_matrix1[1,"no"] + confusion_matrix1[2,"yes"])/1357),"\n" )

roc.curve(bank_df[-train,]$y, glm.pred, plotit = TRUE)

# Classification Trees

library(rpart)
library(rpart.plot)
tree_fit1 <- rpart(y~., method = "class", data = bank_df, subset = train, control = rpart.control(maxdepth = 20, cp=0.0018727)) 
summary(tree_fit1)
printcp(tree_fit1)
plot(tree_fit1, uniform = TRUE)
text(tree_fit1, all=TRUE, cex=0.75, splits=TRUE, use.n=TRUE, xpd = TRUE)
library(maptree)
tree_pred = predict(tree_fit1, bank_df[-train,], type="class")
confusion_matrix2 <- table(tree_pred, actual = bank_df[-train,]$y)
confusion_matrix2
cat("Accuracy of CT : ",((confusion_matrix2[1,"no"] + confusion_matrix2[2,"yes"])/1357) )

roc.curve(bank_df[-train,]$y, tree_pred, plotit = TRUE)

######## Random Forests

library(randomForest)
rf_fit <- randomForest(y~., data = bank_df, subset = train)
rf_fit
confusion_matrix3 <- table( predicted = predict(rf_fit, newdata = bank_df[-train,], type = "class"),
                            actual = bank_df[-train,]$y)
confusion_matrix3
cat("Accuracy of RF : ",((confusion_matrix3[1,"no"] + confusion_matrix3[2,"yes"])/1357) )

roc.curve(bank_df[-train,]$y, predict(rf_fit, newdata = bank_df[-train,], type = "class"), plotit = TRUE)


################################ Over Sampling #####################################

#over_sampled_data <- ovun.sample(y~., data = bank_df[train,], method = "both", N=4000,
#                                 p=0.5, seed = 1)$data
#table(over_sampled_data$y)

rose_data <- ROSE(y~., data = bank_df[train,], seed = 1)$data
table(rose_data$y)

# Logistic Regression 2
glm.fit_2 <- glm(y~., data = rose_data, family = binomial)
glm.probs_2 = predict(glm.fit_2, newdata = bank_df[-train,], type="response")
glm.pred_2 = ifelse(glm.probs_2>0.5, "yes","no")
actual = bank_df[-train,]$y
mean(glm.pred_2==actual)
confusion_matrix4 <- table(glm.pred_2, actual)
confusion_matrix4
cat("Accuracy of Logistic Regression 2 : ",((confusion_matrix4[1,"no"] + confusion_matrix4[2,"yes"])/1357) )

r1 = roc.curve(bank_df[-train,]$y, glm.pred_2, plotit = TRUE)
r1

# Classification Tree 2

tree_fit2 <- rpart(y~., method = "class", data = rose_data, control = rpart.control(maxdepth = 20, cp=0.0026281)) 
#summary(tree_fit2)
printcp(tree_fit2)
plot(tree_fit2, uniform = TRUE)
text(tree_fit2, all=TRUE, cex=0.75, splits=TRUE, use.n=TRUE, xpd = TRUE)
library(maptree)
tree_pred_2 = predict(tree_fit2, bank_df[-train,], type="class")
confusion_matrix5 <- table(tree_pred_2, actual = bank_df[-train,]$y)
confusion_matrix5
cat("Accuracy of CT 2 : ",((confusion_matrix5[1,"no"] + confusion_matrix5[2,"yes"])/1357),"\n" )

r2 =roc.curve(bank_df[-train,]$y, tree_pred_2, plotit = TRUE)
r2

# Random Forests 2

library(randomForest)
set.seed(1)
rf_fit2 <- randomForest(y~., data = rose_data, ntree = 500)
rf_fit2
confusion_matrix6 <- table( predicted = predict(rf_fit2, newdata = bank_df[-train,], type = "class"),
                            actual = bank_df[-train,]$y)
confusion_matrix6
cat("Accuracy of RF 2 : ",((confusion_matrix6[1,"no"] + confusion_matrix6[2,"yes"])/1357),"\n" )

r3 = roc.curve(bank_df[-train,]$y, predict(rf_fit2, newdata = bank_df[-train,], type = "class"), plotit = TRUE)
r3
##################################  RESULTS  ##########################################

cat("\n\n Model Performance : \n\n")
cat("AUC of Logistic Regression : ", r1$auc,"\n")
cat("AUC of Classification Tree : ", r2$auc,"\n")
cat("AUC of RF : ", r3$auc,"\n")

