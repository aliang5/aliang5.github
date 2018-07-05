#The data, credit.csv, is archived in the UCI Machine Learning Data Repository (http://archive.ics.uci.edu/ml). 
#It was collected and donated from University of Hamburg. The dataset contains information on loans obtained from
#a credit agency in Germany and so #the currency is recorded in Deutsche Marks(DM). 

#Goal of this analysis is identifying applicants that are at high risk to default, allowing the bank to refuse credit 
#requests.Default indicates whether the loan applicant is able to pay back the amount of money that they borrow plus all interest. 

## Exploring and prepraring the data
#Credit.csv file includes 1000 observations and 17 features on loan information and loan applicants such as checking and saving account balance, credit history, the amount of loan they plan to borrow, and loan duration, etc. The target feature is located at the very last column as applicant's default status ( Yes or No). 

#Note `sringsAsFacors` option is ignore that all categorical variables are imported as factor. 

#Summary of the data structure:
credit <- read.csv("ml7/credit.csv")
str(credit)

#The loan amounts ranged from 250 DM to 18,240 DM between loan period of 4 to 72 months with median duration of 18 months and amount of 2,320 DM.
summary(credit$months_loan_duration)
summary(credit$amount)

#Here, number of default and not default applicants in this data set. A total of 30% of loans in this dataset went into default.
table(credit$default)


#Create random training and test datasets
#Since this is a relatively small dataset, 90 percent of the data is decided to used for training and 10 percent for testing.
#It's undesirable to simply divide the dataset into two portion as it would create basis in both training and testing processes 
#so a sample randomization technique is used to select 900 observations at random and store as `credit_train` for training dataset and 
#the remaining "unseen" observations store as `credit_test` as test dataset.
#make sure the proportion between the training and test dataset are approximately similar.
#Otherwise, any error generated at the end of the analysis may be cause from the inequality in class proportion. 


# create a random sample for training and test data
set.seed(123)
train_sample <- sample(1000, 900)
credit_train <- credit[train_sample, ]
credit_test  <- credit[-train_sample, ]
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))


##Training a model on the data
#The theory behinds classification decision tree is using entropy to measure information gain. 
#The value of entropy ranges from 0 to 1 for two class levels and from 0 to $\log_2(n)$ for n class levels. 
#$$ Entropy=\sum_{i}^c -P_{i}{log}{2}(P_{i})$$ where c refers to the number of class level and P_i refers to
#the proportion of value falling into class level i.

#Calculating *before split* and *after split* entropy values gives measurement on information gain. 

#So goal of decision tree algorithm is finding a split on attribute that would reduce entropy that ultimately
#increases homogeneity within the branches. The split stops when entropy reaches minimal and that also means the split 
#results in homogeneous branches of division.

#Use the default C5.0 configuration for the first iteration of credit approval model. 

# build the simplest decision tree
library(C50)
credit_model <- C5.0(credit_train[-17], credit_train$default)

#*Train Model Result*
#Using 900 observations and 16 features to train the model, a tree with 57 branches is produced. 
#*Checking Balance* is the *root node* that C5.0 algorithm determined to be the attribute that returned the highest 
#information gain. This also makes intuitive sense that having money in the bank account would allow applicants to pay off proportion of loans or interest. 

#Looking at checking balance feature alone- with 462 applicants who have checking balance that are unknown or greater than 200DM,
#412 examples were correctly classified and granted loan and 50 were incorrectly classified as not likely to default. 
#The applicants who have a checking balance between 0 - 200 DM, it's further split into "perfect,very good" and "critical" credit history.

#The algorithm also provide a confusion matrix of the train data- 35 observations were misclassified as default that the bank granted 
#loans when were actually not(false positive) and 98 observations were misclassified as not default when the bank actually should have
#( false negative). The accuracy for the decision tree train model is 85.2%.

credit_model
# display detailed information about the tree
summary(credit_model)
```

## Evaluating model performance 
#Using `credit_model` as the first parameter and `credit_test` second parameter, it predicts the default status of the remaining 100 applicants 
#from the test datatest. *CrosTable()* function is used to compare the *actual default status* vs the *predicted default status*.

#*First Model Result*
#8 observations are misclassified as default and 19 observations are misclassified as not default. 
#This gives 27 percent error which is 73 percent accuracy. This error is considered large.

credit_pred <- predict(credit_model, credit_test)

library(gmodels)
CrossTable(credit_test$default, credit_pred,
prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
dnn = c('actual default', 'predicted default'))

library(caret)
library(lattice)
library(ggplot2)
confusionMatrix(credit_pred, credit_test$default, dnn = c("Predicted", "Actual"))


##Improving model performance 

#We can improve the decision tree classification model using two method- 
#1. Adaptive Boosting/ AdaBoost  
#2. Error Matrix

#The idea of using Adaptive Boosting is generating not just one but many decision trees and from there, reweighing misclassified events and "vote"" 
#on the best class for each example. Here, adding an additional `trials` parameter indicating the number of separate decision trees to use for the vote. 
#10 trials is the de facto standard. 

#*Second Model Result*
#Average tree size has shrink after adding the `trials` parameter. An accuracy improvement is achieved on the training model that went from 85.2% to 96.2%.

#The confusion matrix of actual vs predicted default in the test dataset is also improved from 73% to 82%. Even though this model predicts better than the original model, 
#we should still use another metric to improve the model performance and to reduce false negative as it's associating with financial lost on the bank.


credit_boost10 <- C5.0(credit_train[-17], credit_train$default,
                       trials = 10)
credit_boost10
#summary(credit_boost10)
```


credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
confusionMatrix(credit_boost_pred10, credit_test$default)


#Another way to improve model performance is using *Error Matrix*. 
#It allows user to discourage tree from making more costly mistaken by assigning a penalty to different types of error. 
#For example, reducing server financial lost rather than losing opportunities is a goal for a bank, we can put on more weight (penalty) for false negative, 
#misclasification as no default, than false positive.

#Here, creating `error_cost` matrix to take different weight for the errors. Then apply this error matrix on model improve using `cost` parameter.

matrix_dimensions <- list(c("no", "yes"), c("no", "yes"))
names(matrix_dimensions) <- c("predicted", "actual")
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2, dimnames = matrix_dimensions)
error_cost


#*Third Model Result*
#The accuracy achieved by this model is 63 percent and the false negative incidents is reduced from 19 to 7.
#Undoubtedly, this has the least accuracy among the three models but this model is a better model as it minimizes the prediction on false negative default status 
#in which may be a better gain for a bank.


credit_cost <- C5.0(credit_train[-17], credit_train$default,
                    costs = error_cost)
credit_cost_pred <- predict(credit_cost, credit_test)
CrossTable(credit_test$default, credit_cost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,

#Final Mark: Decision Tree classification uses entropy to measurement information gain and decides which split of observations to 
#take in order to achieve minimal entropy that ultimately increases homogeneity within the branches. The model and test dataset accuracy and error percentages 
#can be calculated using confusion matrix. AdaBoost and Error Matrix are two way to improve the model performance. The idea of AdaBoost is by generating many more trees
#to reweigh misclassified events. Error Matrix discourages tree from making costly mistaken by assigning a penalty to different types of error. 
#The Error Matrix model is considered to be the best model this banking dataset. 
