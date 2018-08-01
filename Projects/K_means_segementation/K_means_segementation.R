library(readxl)
retail_dat <- read_excel("Downloads/retail_dat.xlsx")

#Remove observiation with missing ID#
length(unique(retail_dat$CustomerID))
sum(is.na(retail_dat$CustomerID))
dat <- subset(retail_dat, !is.na(retail_dat$CustomerID))
unique(dat$Country)

#customer clusters can be varied by geography so I only pick UK in this cluster analysis
dat <-subset(retail_dat, Country == "United Kingdom")
length(unique(dat$InvoiceNo))
length(unique(dat$CustomerID))

#We now have 3,8191 unique customers.

#Invoices with purchases is differ than invoices with returns. Here Id-ing returns -

dat$item.return <- grepl("C", dat$InvoiceNo, fixed=TRUE)
dat$purchase.invoice <- ifelse(dat$item.return=="TRUE", 0, 1)

# Creating RFM varibles
#Recency   = # of days since the customer's last purchased ( smaller values indicate recent activitiy)
#Frequency = # of invoices with purchase
#Monetary  = # of customer spent (there can be negative monetary values) 

#first, lets aggregate the data to create recency variable
customers <- as.data.frame(unique(dat$CustomerID))
names(customers) <- "CustomerID"

dat$recency <- as.Date("2011-12-31") - as.Date(dat$InvoiceDate)
# remove returns so only consider the data of most recent purchase
temp <- subset(dat, purchase.invoice == 1)
## of days since most recent purchase
recency <- aggregate(recency ~ CustomerID, data=temp, FUN=min, na.rm=TRUE)
remove(temp)

# Add recency to customer data
customers <- merge(customers, recency, by="CustomerID", all=TRUE, sort=TRUE)
remove(recency)

customers$recency <- as.numeric(customers$recency)

  
#Get the frequency of customers spends by
#1. remove duplicate invoices
#2. aggregate # of invoices per customer ID
#3. remove customers who have not made any purchases
customer.invoice <- subset(dat, select = c("CustomerID","InvoiceNo", "purchase.invoice"))
customer.invoice <- customer.invoice[!duplicated(customer.invoice), ]
customer.invoice <- customer.invoice[order(customer.invoice$CustomerID),]
row.names(customer.invoice) <- NULL
head(customer.invoice )
# number of purchased invoices per year
annual.invoice <- aggregate(purchase.invoice ~ CustomerID, data=customer.invoice, FUN=sum, na.rm=TRUE)
names(annual.invoice)[names(annual.invoice)=="purchase.invoice"] <- "frequency"

# merge # of invoices to customers data
customers <- merge(customers, annual.invoice, by="CustomerID", all=TRUE, sort=TRUE)
table(customers$frequency)


customers <- subset(customers, frequency > 0)

#Next, on monetary value of customers
#1. Total spent on each item
#2. aggegate the total sales

dat$Amount <- dat$Quantity * dat$UnitPrice

annual.sale <- aggregate(Amount ~ CustomerID, data=dat, FUN=sum, na.rm=TRUE)
names(annual.sale)[names(annual.sale)=="Amount"] <- "monetary"

# add monetary value to customers dataset
customers <- merge(customers, annual.sale, by="CustomerID", all.x=TRUE, sort=TRUE)
remove(annual.sale)

#Identify customers with negative monetary value, as they were presumably returning purchases from the #preceding year
hist(customers$monetary)
customers$monetary <- ifelse(customers$monetary < 0, 0, customers$monetary) # reset negative numbers to zero
hist(customers$monetary)

#Next transforming positive-skewed variables and then standarize them as z-scores
customers$recency_log <- log(customers$recency)
customers$frequency_log <- log(customers$frequency)
customers$monetary_log <- customers$monetary + 0.1 # so add a small value to remove zeros
customers$monetary_log <- log(customers$monetary_log)

customers$recency_z <- scale(customers$recency_log, center=TRUE, scale=TRUE)
customers$frequency_z <- scale(customers$frequency_log, center=TRUE, scale=TRUE)
customers$monetary_z <- scale(customers$monetary_log, center=TRUE, scale=TRUE)

#High scattering of high-value and frequency customers in the top and right area. Points are dark meaning #they recently purchased someething. Most importatnt to note is that data are fairly #continuously-distributed meaning there are no clear clusters.
#So the clusters that are about to created will not reflect the underlying group.

#Also that data did not suggest obvious choice for the number of clusters.

library(ggplot2)
gp <- ggplot(customers, aes(x = frequency_log, y = monetary_log))
gp <- gp + geom_point(aes(colour = recency_log)) +  xlab("Log-transformed Frequency") + ylab("Log-transformed Monetary Value of Customer")
gp

summary(customers)

#Using NbClust metric to find the optimal cluster number

preprocessed <- customers[,8:10]

library(NbClust)
set.seed(11111)
nc <- NbClust(preprocessed, min.nc=2, max.nc=7, method="kmeans")
table(nc$Best.n[1,])

nc$All.index # estimates for each number of clusters on 26 different metrics of model fit

barplot(table(nc$Best.n[1,]),
        xlab="Number of Clusters", ylab="Number of Criteria",
        main="Number of Clusters Chosen by Criteria")

#Build the model - 
output <- kmeans(preprocessed, centers =2, nstart = 20)
var.name <- paste("cluster", 2, sep="_")
customers[,(var.name)] <- output$cluster
customers[,(var.name)] <- factor(customers[,(var.name)])


library(plyr)
print(title)
cluster_centers <- ddply(customers, .(customers[,(var.name)]), summarize,
                         monetary=round(median(monetary),2),# use median b/c this is the raw, heavily-skewed data
                         frequency=round(median(frequency),1),
                         recency=round(median(recency), 0))
names(cluster_centers)[names(cluster_centers)=="customers[, (var.name)]"] <- "Cluster"
print(cluster_centers)

output <- kmeans(preprocessed, centers =6, nstart = 20)
var.name <- paste("cluster", 6, sep="_")
customers[,(var.name)] <- output$cluster
customers[,(var.name)] <- factor(customers[,(var.name)])


library(plyr)
cluster_centers <- ddply(customers, .(customers[,(var.name)]), summarize,
                         monetary=round(median(monetary),2),# use median b/c this is the raw, heavily-skewed data
                         frequency=round(median(frequency),1),
                         recency=round(median(recency), 0))
names(cluster_centers)[names(cluster_centers)=="customers[, (var.name)]"] <- "Cluster"
print(cluster_centers)