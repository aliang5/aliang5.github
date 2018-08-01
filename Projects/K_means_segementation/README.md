# Customer Segmentation using K-means

This repo contains customer segmentation analysis done to an Online Retail data from UCI Machine Learning Repository.

The Approach: Identify high- and low-value customers using calculated RFM (recency, frequency and monetary) variables for clustering.

## Solution:
A 2-cluster produces one group of high monetary value (median= $1685.57) with high-frequency purchase (median = 5) customers who have recently purchased with median of 39 days (since their most recent purchase). The other group has lower value (median = $315.96), and lower frequency (median =1) with median of 96 days since their last purchase.

However, 2-clusters may simplify the customer’s behavior a bit, so I find 6-cluster gives the best insight into subtle distinction between customers. We see:
Cluster 1: Medium- value, medium-purchase frequency but purchase earlier in the year 
Cluster 2: High – value, medium-frequency and also relatively-recent purchase
Cluster 3: High – value, high-frequency and recently purchase group
Cluster 4: monitory value to the business is $0.00
Cluster 5: Low-value, low-frequency and relatively short recency 
Cluster 6: Lowest among the clusters on all variables

## Recommendation: 
I’d recommend the 6-cluster solution to business who wants to understand a range of customer behavior from high-medium-low value customers. And with the distinction in the no-value group of customers, business probably want to eliminate or work out a marketing strategy to bring those customers back on more regular spending.

Date source
https://archive.ics.uci.edu/ml/datasets/Online+Retail 