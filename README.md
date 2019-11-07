# HR ANALYTICS and MODELING :

## Problem Statement :
HR analytics is revolutionising the way human resources departments operate, leading to higher efficiency and better results overall. Human resources has been using analytics for years. However, the collection, processing and analysis of data has been largely manual, and given the nature of human resources dynamics and HR KPIs, the approach has been constraining HR. Therefore, it is surprising that HR departments woke up to the utility of machine learning so late in the game. Here is an opportunity to try predictive analytics in identifying the employees most likely to get promoted.

## Dataset Description
|Variable|Definition|
|--------|:---------:|
|employee_id|	Unique ID for employee
|department	|Department of employee
|region	|Region of employment (unordered)
|education|	Education Level
|gender|	Gender of Employee
|recruitment_channel|	Channel of recruitment for employee
|no_of_trainings|	no of other trainings completed in previous year on soft skills, technical skills etc.
|age|	Age of Employee
|previous_year_rating|	Employee Rating for the previous year
|length_of_service|	Length of service in years
|KPIs_met >80%	|if Percent of KPIs(Key performance Indicators) >80% then 1 else 0
|awards_won?|	if awards won during previous year then 1 else 0
|avg_training_score	|Average score in current training evaluations
|is_promoted|	(Target) Recommended for promotion

## Solution :
### Problem Understanding :

1. HR are mostly following manual approach for collecting data,processing data and analysis of data which is really difficult for any big organisation to do all these things manually and come up with correct result.
2. HR team also want to make the things automated to save the time as well correctness just like many organisation is currently using chatbot to help their customers(saving customer care member time for basic queries and definitely chatbot works 24/7).
3. From last line of problem,I am able to observe that it is classification problem because employee will be either promoted or not promoted.

### Sequential steps followed during problem solving 
1. Exploring the data
2. Variables Identification
3. Missing Value Treatment
4. Univariate Analysis (4.1 Univariate analysis 4.2 Bivariate Analysis)
5. Bivariate Analysis (5.1 Continuous-Continuous variables 5.2 Continuous-Categorical Variables 5.3 Categorical-Categorical Variables)
6. Outliers Treatment (No requirement for our dataset)
7. Feature Engineering
8. Modeling and Evaluation (8.1 Directly applying algorithms 8.2 Downsampling 8.3 Upsampling 8.4 SMOTE)
9. Saving the models


### What are the challenges and how to overcome ? 

1. How to do preprocessing part if I have separate train and test data? (merge or concatenate or treat independently).
2. Missing value treatment for columns education and previous_year_rating (not directly imputed mode value or dropping these column will be fine).
3. Separating of continuous and categorical variables (sometimes it is hard when categorical variable is in numerical form and is present in large numbers like in this problem 'no_of_trainings', 'avg_training_score').
4. Target column,'is_promoted' has imbalanced(91:9) data (so to deal such type of data advance technique required but still it is better to first go through brute force approach).
5. For imbalance data,accuracy metrics(biased towards majority class value) is not the good metrics to check the performance of model (precision,recall,f1_score metrics will work for such data).
6. Feature engineering is challenging for some variables if there is ambiguity that, is this variable is nominal or ordinal in nature.
7. After simply applying four algorithms accuracy goes approx,92% (in all four cases as expected due to unbalanced class) while f1 score are 34% (in case of Logistic Regression) and 41% (in case of Random Forest).
8. To overcome from above problem (point 7),three new techniques tried that are Downsampling, Upsampling and SMOTE .
9. Downsampling, Upsampling works well compare to SMOTE and Logistic Regression and KNN consistantly perform well (accuracy,precision,recall and f1 score are constant around 75%) for  Downsampling, Upsampling techniques.

### Observations/Insights during preprocessing part :
1. Train dataset consist of approx 55k observations and 14 features while test dataset has 23.5k observations and 13 features.
2. education and previous_year_rating columns have missing values but very less in number so dropping is not good option.
3. Some of the graph is little bit skewed but still fine.
4. The data in train and test is almost similar,we can observe each column corresponding graph for both dataset.
5. For maximum organisation, sales and marketing department play huge role(this department is bringing customers).In our data also 35%(highest) people belong to this department only.
6. 70% have Bachelor's degree in our data and It is true for almost many organisation
7. Again we know that Male number is higher in maximum industry and same our data is also giving.70% people are male.
8. There are only few people(4%) who come in industry through reference.
9. Approx 65% people have KPI<80% and which is really correct if we see data of any company.
10. Only 2% people won the award.
11. Approx 85% people are not promoted and only 15% people are promoted.From this data we can observe class is imbalance and type 1 or type 2 error occur.We can't use accuracy here.
12.  The person with age 27-35(binning option) has higher chance of promotion.
