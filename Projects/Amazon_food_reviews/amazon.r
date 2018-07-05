#The dataset is provided by Stanford Network Analysis Project (SNAP). It contains Amazon fine #foods reviews from October 1999 to October 2012.
#Goal of this project is using Naive Bayes #Algorithm to perform sentiment classification for reviews scores based on the reviews. 

# Features
#**Target Variable** : Score - Rating between 1 and 5 
#**Other Features**
#1. Id
#2. Product Id - Unique identifier for the product
#3. UserId - Unqiue identifier for the user
#4. ProfileName - Profile names of the customers who left the reviews
#5. HelpfulnessNumerator - The number of customers who have voted and found the review helpful
#6. HelpfulnessDenominator - The total number of customers who indicated whether or not the the #review was helpful
#7. Score - Rating between 1 and 5
#8. Time - Timestamp for the review
#9. Summary - A summary of the review
#10.Text - The review


Reviews <- read.csv('/Users/Annie/Documents/Reviews2.csv', stringsAsFactors = FALSE)
index <- sample(1:40000)
Reviews1 <- Reviews[index,]
rm(Reviews)

#Data Structure 
str(Reviews1)

#Using `table()` to get the count of scores. 
table(Reviews1$Score, useNA = "ifany")


head(Reviews1[,7:9])


#Assign 4 or higher scores as positive reviews and otherwise negative.

Reviews1$Score<- ifelse(Reviews1$Score >= 4, "positive", "negative")
head(Reviews1[,7:9])


#Convert character, positive and negative to factor.
Reviews1$Score <- factor(Reviews1$Score)

table(Reviews1$Score)

head(Reviews1[,10])


#Use tm package to clean up review texts, create a bag of words and frequency with which those words were used.  

library(tm)
library(NLP)
Reviews_corpus1<- VCorpus(VectorSource(Reviews1$Text))

#View the actual review text using `as.character()`.

lapply(Reviews_corpus1[1:2], as.character)

# Data Cleaning
#* Convert texts in corpus to lowercases
#* Stop words removal such as to, and, but and or
#* Pruning (numbers and punctuation)
#* Stemming
#* Tokenization


reviews_corpus_clean1 <- tm_map(Reviews_corpus1, content_transformer(tolower))


#Took 34 secs to remove stop words.

start.time <- Sys.time()
reviews_corpus_clean1 <- tm_map(reviews_corpus_clean1, removeWords, stopwords())
end.time<- Sys.time()
time.taken_stopwords<- end.time - start.time
time.taken_stopwords


reviews_corpus_clean1 <- tm_map(reviews_corpus_clean1, removePunctuation)
reviews_corpus_clean1 <- tm_map(reviews_corpus_clean1, removeNumbers)
library(SnowballC)
wordStem(c("amazed", "amazing", "amazingly"))
wordStem(c("artifical", "artificial", "artificially"))
wordStem(c("assume", "assumed", "assuming","assumption"))
wordStem(c("assume", "assumed", "assuming","assumption"))
wordStem(c("bake", "baked", "baker","bakery","bakes"))
wordStem(c("begin", "beggin", "began","beginning","begins","begun"))
wordStem(c("blueberries", "blueberry"))
wordStem(c("coffee", "coffe", "coffeebr","coffeemaker","coffees"))
wordStem(c("fall", "fallen", "falling","falls"))
wordStem(c("favorite", "favorable", "favorite","favoritebr"))
wordStem(c("fresh", "freshbr", "freshen","fresher","freshly","freshness"))
wordStem(c("guess", "guessed", "guessing"))
wordStem(c("improves", "improved", "improvement"))
wordStem(c("interest", "interested", "interesting","interestingly"))
wordStem(c("recommendations", "recommend", "recommendation","recommended","recommending","recommendedbr"))
wordStem(c("smooth", "smoother","smoothies","smoothly","smoothness"))
wordStem(c("sweetened", "sweetener", "sweeteners","sweetening","sweeter","sweetner","sweetness", "sweets"))

reviews_corpus_clean1 <- tm_map(reviews_corpus_clean1, stemDocument)
review_corpus_clean1 <- tm_map(reviews_corpus_clean1, stripWhitespace)


#Review before data cleaning

lapply(Reviews_corpus1[1:2], as.character)

#After data cleaning
lapply(reviews_corpus_clean1[1:2], as.character)

#Create DTM. It converts corpus into a data structure that has documents as the rows, terms/words as the columns,
#frequency of the term in the document as the entries.

start.time1 <- Sys.time()
Reviews_dtm <- DocumentTermMatrix(reviews_corpus_clean1)
end.time1<- Sys.time()
time.taken_dtm<- end.time1- start.time1
time.taken_dtm

dim(Reviews_dtm)


#Remove features that aren't applicable for the training. 

Reviews3<-Reviews1[-c(1:4)]
Reviews4<-Reviews3[,-4]


# Create Training and Test dasets
index <- sample(1:nrow(Reviews_dtm), as.integer(0.7*nrow(Reviews_dtm)))

Reviews_dtm_train<- Reviews_dtm[index, ]
Reviews_dtm_test<-  Reviews_dtm[-index,  ]

Reviews_train_labels<-Reviews4[index,3]
Reviews_test_labels<-Reviews4[-index,3]

#Proportion for train labels
prop.table(table(Reviews_train_labels))
#Proportion for test labels
prop.table(table(Reviews_test_labels))



#Data Visualization
library(wordcloud)
library(RColorBrewer)
wordcloud(reviews_corpus_clean1,min.freq = 50, random.order = FALSE, colors=brewer.pal(8, "Dark2"), max.words = 40)


#Creating Indicator Feature
#Transform *sparse matrix* into data structure that Naive Bayes classifier can train. Use `findFreqTerms()` to takes DTM 
#and return as *character vector* contains words that appear for 30 times.

reviews_freq_words <- findFreqTerms(Reviews_dtm_train, 30)
str(reviews_freq_words)


#Filter the DTM sparse matrix to only contain words with at least 30 occurrences.

reviews_dtm_freq_train <- Reviews_dtm_train[ , reviews_freq_words]
reviews_dtm_freq_test <- Reviews_dtm_test[ , reviews_freq_words]

reviews_dtm_freq_test


#Since Naive Bayes classifier trains on categorical features, the numerical data must be converted to categorical data. 
#Using `couvert_counts()` to convert counts to *Yes/No* strings.

convert_counts <- function(x){
x <- ifelse(x > 0, "Yes", "No")
}


#Apply the reduced DTM to train and test dataset per columns.

#MARGIN parameter specify either rows or columnes. MARGIN=1 is used for rows.
Reviews_train <- apply(reviews_dtm_freq_train, MARGIN = 2, convert_counts)
Reviews_test <- apply(reviews_dtm_freq_test, MARGIN = 2, convert_counts)

library(e1071)
system.time( reviews_classifier <- naiveBayes(Reviews_train,Reviews_train_labels) )


#Evaluate model performance

system.time( reviews_test_pred <- predict(reviews_classifier, Reviews_test) )
head(reviews_test_pred)

library(caret)
conf.mat <- confusionMatrix(reviews_test_pred, Reviews_test_labels)
conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy']
##########Accuracy is 81% ############



#Improving model performance
#Laplacian smoothing is used to improve the model performance.

library(e1071)
system.time( review_classifier2 <- naiveBayes(Reviews_train,Reviews_train_labels, laplace = 1) )

system.time( reviews_test_pred2 <- predict(review_classifier2 , Reviews_test ))
head(reviews_test_pred2)

library(gmodels)
CrossTable(reviews_test_pred2 ,Reviews_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))

#Accuracy is 81%.


conf.mat <- confusionMatrix(reviews_test_pred2, Reviews_test_labels); conf.mat
conf.mat$byClass; conf.mat$overall; conf.mat$overall['Accuracy']

#Conclusion: The prediction accuracy of a classification model is given by the proportion of the total number of correct predictions. 
#The accuracy for this model turns out to be 81%. Accuracy for using laplacian smoothing is same. Naive Bayes algorithm does pretty well
#at predicting the correct sentiment reviews.