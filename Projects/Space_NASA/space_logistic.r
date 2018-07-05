
## Libraries Used
library(car)
library(Rcpp)   #required package for Amelia
library(gplots) #required package for Amelia
library(Amelia) #data visdualization 
library(gplots) #required package for ROCR
library(ROCR)
library(ggplot2) #required package for caret
library(lattice) #required package for caret
library(caret)

##Objective
#The project aims to use *logistic regression* to predict the binary outcome of Challenger space mission given the temperature at launch.

##Data Exploration
#The data collected from 23 previous successful shuttle launches prior to the actual Challenger space shuttle launch on January 28, 1986. 
#There are four features in the dataset with `Distress_ct` indicating number of distress events such as shuttle disintegration as independent predictor variable.
#Temperature, field_check_pressure, and flight_number are independent variables.

#Load the dataset
launch <- read.csv("ml10/challenger.csv")

#data preview
#The shuttle had 6 O-rings responsible for sealing joints of the rocket booster and cold temperatures could make the component more brittle and susceptible to improper sealing.

#The correlation between `distress_ct` and `temperature` is tested where -0.51 meaning there is a moderately 
#strong negative linear association. In addition, the negative correlation implies that an increase in temperature are related to decrease in the number of distressed events. 

cor<-cor(launch$distress_ct,launch$temperature); cor
cor2<-cov(launch$temperature,launch$distress_ct)/(sd(launch$temperature)*sd(launch$distress_ct)) #manually caluation for Pearson's correlation. Alternatively, R's correlation function is cor()
cor2


#Next, check if there are any missing values in dataset. Amelia package provides a visual image of missing data per features.  

missmap(launch, main = "Missing values vs observed")

#'NA' is sometimes used in a dataset instead of null to represent missing values. Sapply() is used to check for 'NA'. 
#'#It also can be used to check for number of unique values in each feature. Here, 3 different values are representing distress_ct, 16 different values for temperature field, etc.

# gives number of missing values in each column
sapply(launch,function(x) sum(is.na(x)))

# gives number of unique values in each column
sapply(launch, function(x) length(unique(x)))


## Logistic Regression

#`distress_ct` is ranging from value 0 to 3. Hence, it needs to be converted into a binary outcome where 0 means there is no distress event and 1 as detection of a distress event for the logistic regression model.
#Any values that are greater than 1 would be converted to 1 to imply a distree event was happended, and everything else would be converted to 0.

launch$distress_ct = ifelse(launch$distress_ct<1,0,1)
launch$distress_ct


#This is a small dataset (only 23 observations) so 90% of overall observations is used for training data and 10% is used for testing data.

# set up trainning and test data sets
set.seed(123)
indx = sample(1:nrow(launch), as.integer(0.9*nrow(launch)))
indx

launch_train = launch[indx,]
launch_test = launch[-indx,]

launch_train_labels = launch[indx,1]
launch_test_labels = launch[-indx,1]   

#The generalized linear models (GLMs) are a broad class of model that includes linear regression, ANOVA, logistic etc. Therefore, family parameter is used in glm() to define which model to use. 
#The coefficients estimators obtained from the model is a *log odds* of response variable, distress_ct. Because the generalized linear model for logistic is log odds of the dependent variable (class)
#and it's a linear combination of the independent variables(features). 


#fit the logistic regression model, with all predictor variables

model <- glm(distress_ct ~.,family=binomial(link='logit'),data=launch_train)
summary(model)



#In order to better estimate the significance of each predictors, anova() with chi-square test parameter is used to see how does the model fit by adding each predictors.
#The best model is one that has the *least residuals deviance*. 

#Here, having the temperature as a predictor in the model has the greatest decrease in residual deviance compared to NULL model. 
#And having field_check_pressure and flight_num do not have significant decrease in residual deviance compared to temperature. 
#The p-value of temperatuer also confirms that temperature predctor is significant at the 0.05 level so its the best predictor for the model.

anova(model, test="Chisq")

#summary(model)

#anova(model, test="Wald")


Here, evaluate the model again with just temperature as predictor.

# drop the insignificant predictors, alpha = 0.10
model <- glm(distress_ct ~ temperature,family=binomial(link='logit'),data=launch_train)
summary(model)
anova(model, test="Chisq")


## Evaluating Model Performance
#Type='response' sets the prediction for the model to probability values of each tested observation between 0 and 1.
#To get a meaningful binary prediction of `distress_ct` as to whether shuttle would be distress or not, it needs be converted the probability values into 0 or 1.
#Hence, if probability of distress_ct is >0.5, it would be counted as 1 but 0 otherwise.

# check Accuracy
fitted.results <- predict(model,newdata=launch_test,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
head(fitted.results)

#Calculating the error rate of the testing data on distress_ct from the model. Accuracy rate is 100%.

misClasificError <- mean(fitted.results != launch_test$distress_ct)
print(paste('Accuracy',1-misClasificError))

#Another way to evaluate model performance is using *Receiver Operating Characteristic (ROC)*. ROC curve is a graph that simultaneously displays true positive (sensitivity) and false positive rate (1- specificity). 
#The overall performance of classifier is summarized by area under the curve (AUC) and larger the AUC represents the better classifier.
#For our model, the ROC curve is showing a straight line that does not convey meaningful message of the model accuracy. This is probably due to size of training and testing dataset being too small.


p <- predict(model, newdata=launch_test, type="response")
pr <- prediction(p, launch_test$distress_ct)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
abline(0,1, lwd=2, lty=2)


#AUC value can be extracted using performance(). 1 is equivalent to perfect classifier. 

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


##Model Improvement using Boostrapping.
#Since the size of this dataset is too small, bootstrapping is a reasonable way to improve the model by re-sampling the number of observations in this dataset multiple times with replacement. 


start.time <- Sys.time()
ctrl <- trainControl(method = "boot632", number = 23, savePredictions = TRUE)
mod_fit <- train(distress_ct ~ ., data=launch,method="glm", family="binomial", trControl = ctrl, tuneLength = 5)


#The prediction values from the testing dataset are probability between 0 and 1 so it needed to be converted into binomial value. 
#3 out of 3 observations in the testing dataset are coxrrectly predicted.

pred <- predict(mod_fit, newdata=launch_test)
pred = ifelse(pred > 0.5,1,0)
confusionMatrix(data=pred, launch_test_labels)


#The time that takes to run boostraping is 0.5 secs.

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken


#Conclusion: Logistic regression is used to predict distress event for the space shuttle incident. 
#A simple model was first built and found temperature was the best independent variable for the dependent outome. 
#Sine the dataset is very small with only 23 observations, boostrapping method was used for model improvement. However, model alone or with bootstrap method are equally good model for this dataset.