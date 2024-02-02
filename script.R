
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE)
set.seed(20231205)

library(dplyr)
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(tidyverse)
library(caret)
library(yardstick)
library(lime)
library(pROC)
library(rsample)
library(recipes)
library(purrr)
library(corrplot)
library(rpart.plot) 
library(ggthemes)
library(tidyquant)
library(randomForestExplainer)
library(randomForest)
```

## Load Data

```{r churn}
churn <- read.csv('churn.csv',sep = ',')

churn %>% 
  glimpse()
```

## Churn Distribution
```{r}
table(churn$Churn)
```

```{r}
prop.table(table(churn$Churn))
```

## Dataset Description
```{r, echo=FALSE}
summary(churn)

```

## Data Preprocessing
### Data features
+----------------------+------------------------------------------------------------------------+
| Field                | Description                                                            |
+----------------------+------------------------------------------------------------------------+
| customerID           | Customer ID                                                            |
| genderCustomer       | Gender (female, male)                                                  |
| SeniorCitizen        | Whether the customer is a senior citizen or not (1, 0)                 |
| Partner              | Whether the customer has a partner or not (Yes, No)                    |
| Dependents           | Whether the customer has dependents or not (Yes, No)                   |
| tenure               | Number of months the customer has stayed with the company              |
| PhoneService         | Whether the customer has a phone service or not (Yes, No)              |
| MultipleLines        | Whether the customer has multiple lines or not (Yes, No, No phone service) |
| InternetService      | Customer’s internet service provider (DSL, Fiber optic, No)           |
| OnlineSecurity       | Whether the customer has online security or not (Yes, No, No internet service) |
| OnlineBackup         | Whether the customer has online backup or not (Yes, No, No internet service) |
| DeviceProtection     | Whether the customer has device protection or not (Yes, No, No internet service) |
| TechSupport          | Whether the customer has tech support or not (Yes, No, No internet service) |
| StreamingTV          | Whether the customer has streaming TV or not (Yes, No, No internet service) |
| StreamingMovies      | Whether the customer has streaming movies or not (Yes, No, No internet service) |
| Contract             | The contract term of the customer (Month-to-month, One year, Two year) |
| PaperlessBilling     | Whether the customer has paperless billing or not (Yes, No)             |
| PaymentMethod        | The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) |
| MonthlyCharges       | The amount charged to the customer monthly                             |
| TotalCharges         | The total amount charged to the customer                               |
| Churn                | Whether the customer churned or not (Yes or No)                       |
+----------------------+------------------------------------------------------------------------+

```{r}
churn %>%
  str()
```
### Check duplicates
```{r}
churn %>% 
  duplicated() %>% 
  any()
```

### Check missing Values
```{r}
churn %>%
  anyNA()
```

```{r}
sapply(churn, function(x) sum(is.na(x)))
```
It is evident that there are missing values in the dataset. The following five techniques can be employed to deal with missing data, and they include - Deletion, Mean Imputation, Median Imputation, Hot Deck Imputation, Multiple Imputation.

 **Deletion**: This method includes removing from the dataset any observations or variables having missing values. Although this is a simple method, if the missing data isn't entirely random, it may lead to bias and knowledge loss.
 
**Mean Imputation**: This method replaces missing values for a variable with the observed value mean. It's an easy procedure, but if the missing data isn't random, it might distort the variable's distribution.

 **Median Imputation**: This technique replaces missing values with the median of the observed values for that variable. Because it is less affected by outliers than mean imputation, it is a more reliable method.

**Hot Deck Imputation**: This method uses the value of a comparable observation to fill in the missing data. If there are significant correlations between the variables, it could be useful, but it can also be difficult to use and lead to bias.

**Multiple Imputation**: In this approach, several datasets are imputed and the missing values are filled in for each dataset using a distinct imputation technique. Next, an average of the analysis's findings is applied to all of the imputed datasets. Although it might be computationally demanding, this is a more sophisticated method.


```{r}
churn %>%
  filter_all(any_vars(is.na(.))) %>%
  nrow()
```
```{r}
sum(is.na(churn$TotalCharges))/nrow(churn)
```

The are 11 eleven rows with the missing is data - this subset is 0.16% of our data and is quite small. Statistically, this number is too negligible to impact any further analysis if they are removed, considering that the are a total of 7043 rows in the original dataset. Therefore, to handle this missing values, the deletion option of dealing with the missing values in the dataset is preferred.


```{r}
# Eliminate the rows with missing values
churn <- na.omit(churn)
dim(churn)
```

It is also important to ensure that the dataset contains no unnecessary duplicates, for this reason, the possible duplicates are removed before any further analysis are carried out. After checking the dataset, there were no duplicated rows which proved beneficial for the subsequent analysis.

```{r}
churn %>%
  filter(duplicated(.)) %>%
  nrow()
```

### Dropping unnecessary features

The column customerID is a unique identifier for each record in the dataset, it has no effect in the further analysis and as such, it is removed from the dataset. This ensures that only relevant features are used in the further analysis.

```{r}
churn <- churn %>%
    select(everything(),-c(customerID))
```

```{r}
colnames(churn)
```

### Convert variables to factors

All the categorical variables in the dataset have been appropriately designated as factors, signifying distinct features. The character features that underwent conversion into factor variables encompass attributes such as gender, SeniorCitizen status, partnership status, presence of dependents, tenure, phone service availability, multiple lines availability, internet service provider, online security presence, online backup availability, device protection status, tech support availability, streaming TV service, streaming movie service, contract terms, paperless billing preference, payment method, and churn status.

```{r}
churn$SeniorCitizen <- as.factor(churn$SeniorCitizen)
churn <- churn %>%
  mutate(across(where(is.character), factor))
churn %>%
  glimpse()
```
The SeniorCitizen variable is coded 0 or 1 rather than Yes or No. This needs to be re-coded  to ease further interpretations of the implemented graphs and models. Also, the MultipleLines variable is dependent on the PhoneService variable - in which a "No" value for the MultipleLines variable generally means a No. To ease the implementatio and interpretation of the graphics and modeling, the values held by the MultipleLines variable is re-coded from "No phone service" response to "No" for the MultipleLines variable.
Similiarly, the OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, and StreamingMovies variables are all dependent on the OnlineService variable where they have values with "No internet service" which simply means "No". These values are all re-coded from "No internet service" to "No".

```{r}
# Recode "SeniorCitizen" column
churn <- churn %>%
  mutate(SeniorCitizen = fct_recode(SeniorCitizen, "No" = "0", "Yes" = "1"))

# Recode "MultipleLines" column
churn <- churn %>%
  mutate(MultipleLines = fct_recode(MultipleLines, "No phone service" = "No"))

# Recode a range of columns (9 to 14)
for (i in 9:14) {
  churn[[i]] <- as.factor(fct_recode(churn[[i]], "No" = "No internet service"))
}

churn$Tenure <- cut(churn$tenure, breaks = c(0, 12, 24, 36, 48, Inf), labels = c("0-1", "1-2", "2-3", "3-4", "4+"))

# Verify the changes
str(churn)

```
```{r}
# The final clean dataset ready for analysis and exploration
df <- churn
```
## Exploratory Data Analysis

Exploratory Data Analysis (EDA), also known as Data Exploration, is a step in the Data Analysis Process, where a number of techniques are used to better understand the dataset being used. EDA is a crucial step in the data analysis process where the study examined and summarized key characteristics, patterns, and relationships within a dataset.

**Summary Statistics**

The study used R functions like summary() and describe() to get basic statistics for numerical variables.
For categorical variables, the table() function was used to get frequency distributions.

**Data Visualization**

This stage involved a number of sub-processes such as:
- Creating histograms for numerical variables to understand their distributions.

- Use bar plots for categorical variables to visualize frequencies.

- Use of Box plots to provide insights into the spread and central tendency of numerical data.

- Use of Scatter plots to help visualize relationships between two numerical variables.

**Correlation Analysis**

This stage involved computing the correlation coefficients (e.g., Pearson, Spearman) to measure relationships between pairs of numerical variables as well as visualizing the correlations using a correlation matrix and/or a heatmap.


### Summary Statistics
```{r}
summary(df)
```
### Visualization

```{r}
# Function to demographic plots
create_plot <- function(data, variable) {
  ggplot(data, aes(x = .data[[variable]], fill = Churn)) +
    geom_bar(position = "dodge") +
    geom_text(aes(y = ..count.. - 200, 
                  label = paste0(round(prop.table(..count..), 4) * 100, '%')), 
              stat = 'count', 
              position = position_dodge(0.9),
              size = 3) +
    ylab("Count")
}

```

```{r}
# List of demographic variables
demographic_variables <- c("gender", "SeniorCitizen", "Partner", "Dependents")

# Create and arrange plots
plots <- lapply(demographic_variables, function(var) create_plot(df, var))
grid.arrange(grobs = plots, ncol = 2)
```


```{r}
# List of offered services variables
offered_service_variables <- c("PhoneService", "MultipleLines", "InternetService", "OnlineSecurity")

# Create and arrange plots
plots <- lapply(offered_service_variables, function(var) create_plot(df, var))
grid.arrange(grobs = plots, ncol = 2)
```


```{r}
# List of offered services variables
offered_service_variables <- c("OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")
plots <- lapply(offered_service_variables, function(var) create_plot(df, var))
grid.arrange(grobs = plots, ncol = 2)

```

```{r}
# List of offered services variables
offered_service_variables <- c("Contract", "PaperlessBilling", "PaymentMethod")
plots <- lapply(offered_service_variables, function(var) create_plot(df, var))
grid.arrange(grobs = plots, ncol = 2)
```

The analysis of the dataset reveals prominent patterns in customer services utilization. A significant majority of the sampled customers are subscribed to phone services, with a preference for a single phone line over multiple lines. This trend underscores the prevalent choice for a more streamlined and singular communication channel, potentially reflecting cost-conscious consumer behavior or individual lifestyle preferences.

Furthermore, the data emphasizes a clear dominance of Fiber optic internet connections over DSL services. This suggests a growing preference for high-speed and reliable internet connectivity, possibly driven by the increasing demand for bandwidth-intensive activities such as video streaming, online gaming, and remote work.

Additionally, the examination of various online services, including online security, backup, device protection, tech support, streaming TV, and streaming movies, indicates that these features collectively cater to a niche audience. The majority of customers seem to opt for basic service packages, with a smaller subset choosing to enhance their plans with additional online offerings. This distribution underscores the diverse needs and priorities of the customer base, with the majority finding satisfaction in standard service packages while a minority opts for more specialized and comprehensive online services.

```{r}
df %>% ggplot(aes(x=TotalCharges,fill=Churn)) +
    geom_density(alpha=0.8) +
    scale_fill_manual(values=c('red','yellow')) +
  labs(title='Total Charges split churn vs non churn' )
```
```{r}
df %>% ggplot(aes(x=MonthlyCharges,fill=Churn)) +
    geom_density(alpha=0.8) +
  labs(title='Monthly Charges split churn vs non churn' )
```
```{r}
df %>% ggplot(aes(x=tenure,fill=Churn)) +
    geom_density(alpha=0.8) +
    scale_fill_manual(values=c('blue','pink'))+
  labs(title='Tenure split churn vs non churn' )
```

```{r}
# Function to create histograms for quantitative variables
create_histogram <- function(data, variable, binwidth) {
  # Check if the variable is numeric
  if (!is.numeric(data[[variable]])) {
    # Convert the variable to numeric
    data[[variable]] <- as.numeric(data[[variable]])
  }
  
  ggplot(data, aes(.data[[variable]], color = Churn)) +
    geom_freqpoly(binwidth = binwidth, size = 1) 
}

# List of quantitative variables and corresponding binwidths
quantitative_variables <- list(
  list(variable = "tenure", binwidth = 5),
  list(variable = "MonthlyCharges", binwidth = 5),
  list(variable = "TotalCharges", binwidth = 200)
)

# Create and arrange histograms
plots <- lapply(quantitative_variables, function(vars) create_histogram(df, vars$variable, vars$binwidth))
grid.arrange(grobs = plots, ncol = 2)

```

The analysis of the quantitative variables in the dataset reveals distinctive patterns in customer tenure, monthly charges, and total charges. The "tenure" variable exhibits a bimodal distribution, with a significant proportion of customers having either the shortest tenure of 1 month or the longest tenure of 72 months. This suggests a segmentation in customer behavior, possibly indicating a substantial number of new customers and a loyal customer base that has stayed with the service provider for an extended period. The bimodal nature of the distribution highlights the importance of understanding and catering to the needs of both these customer segments.

Examining the "MonthlyCharges" variable reveals a roughly normal distribution centered around $80 per month, with a notable concentration near the lower rates. This distribution implies that a substantial portion of customers opts for lower-cost plans, highlighting the importance of competitive pricing and the demand for more budget-friendly options.

Moreover, the "TotalCharges" variable displays a positively skewed distribution, indicating that a considerable number of customers have lower total charges. This skewness could be attributed to a significant portion of the customer base choosing lower-priced plans or recently subscribing to the service. Understanding the distribution of total charges is crucial for revenue forecasting and tailoring marketing strategies to different customer segments.


### Correlation
```{r}
df %>%
  select (TotalCharges, MonthlyCharges, tenure) %>%
  cor() %>%
  corrplot.mixed(upper = "circle", tl.col = "black", number.cex = 0.7)
```
```{r}
df %>%
  dplyr::select(TotalCharges, MonthlyCharges, tenure) %>%
  cor()
```
```{r}
# Create the correlation plot
correlation_matrix <- df %>%
  dplyr::select(TotalCharges, MonthlyCharges, tenure) %>%
  cor()
corrplot(correlation_matrix, method = "circle")
```

The correlation matrix analysis reveals valuable insights into the relationships among key variables in the dataset. One notable observation is the moderate positive correlation (0.6511) between "TotalCharges" and "MonthlyCharges." This finding suggests that customers incurring higher monthly charges tend to accumulate higher total charges over time. The positive correlation signifies a coherent influence of monthly charges on the overall amount charged to a customer, emphasizing the importance of understanding and managing monthly subscription costs.

Furthermore, a robust positive correlation (0.8259) emerges between "TotalCharges" and "Tenure," indicating that customers who have been with the company for a longer duration tend to accumulate higher total charges. This correlation underscores the role of customer loyalty and tenure in shaping the overall financial contribution of each customer. Longer-term relationships are associated with increased total charges, emphasizing the significance of customer retention strategies and the potential for revenue growth over time.

Conversely, the correlation coefficient of 0.2469 between "MonthlyCharges" and "Tenure" reveals a comparatively weaker positive relationship. While there is some discernible correlation, it is not as pronounced as the correlation between "TotalCharges" and "Tenure." This implies that monthly charges may have a less substantial dependence on the tenure of a customer. Understanding these nuanced relationships is crucial for tailoring business strategies, as it allows for a more precise focus on factors influencing customer charges and longevity with the service. Due to the high correlation between the MonthlyCharges and TotalCharges variables, it is considered appropriate to eliminate the TotalCharges variable from the dataset.

```{r}
df <- df %>%
    select(everything(),-c(TotalCharges, tenure))
```

## Modeling

In this project, the application of three machine learning models, namely Naive Bayes, Decision Tree, and Random Forest, stands as a pivotal strategy to derive predictive insights from the dataset. To objectively evaluate the efficacy of these models, a fundamental step involves partitioning the available data into distinct subsets for training and testing purposes. This division enables a comprehensive assessment of model performance by allowing the models to learn from a designated training set and subsequently apply their acquired parameters to make predictions on an independent test set.

To establish these subsets, a randomized sampling approach will be adopted from the complete dataset. The set.seed() function, a crucial component for reproducibility, can be adjusted to reset the random number generator utilized in the sampling process. The training subset is designated to encompass approximately 70% of the original dataset, while the remaining 30% forms the test subset. This ratio strikes a balance between providing a sufficiently large training set for robust model learning and ensuring a sizable test set for rigorous performance evaluation.

This division strategy facilitates a robust evaluation of the models' generalization capabilities by exposing them to data they have not encountered during training. Ultimately, this meticulous approach to model assessment ensures a comprehensive understanding of their predictive power and reliability when applied to unseen data, paving the way for informed decision-making in the context of customer churn prediction.


```{r}
split_train_test <- createDataPartition(df$Churn,p=0.7,list=FALSE)
train_df <- df[split_train_test,]
test_df  <- df[-split_train_test,]
```

```{r}
dim(train_df)
```

```{r}
dim(test_df)
```


After splitting the data, the test dataset comprises 2108 observations and 20 variables, while the training dataset consists of 4924 observations and 20 variables. These dimensions signify the number of instances and features, respectively, in each dataset. The distinct sizes of the training and testing datasets are pivotal for machine learning model evaluation, with the former used to train models and the latter serving as an independent set for assessing model generalization to unseen data.

### Decision Tree Classifier

A Decision Tree Classifier is a simple yet powerful Machine Learning algorithm. It follows a set of if-then rules which lead to a decision. The tree structure consists of nodes representing feature splits, branches representing rule conditions, and leaves representing outcomes or decisions. It's intuitive and easy to interpret, making it great for visual understanding of data. It can handle both categorical and numerical data. However, it can easily overfit or underfit the data, so tuning its parameters like tree depth is crucial. It's sensitive to data changes, meaning small data variations can result in different trees. Despite these challenges, it's widely used in various fields due to its simplicity and interpretability.

```{r}
dt_fit <- rpart(Churn ~., data = train_df, method="class")
rpart.plot(dt_fit)
```

The decision tree model reveals some key insights: Firstly, the type of contract a customer has is the most significant factor. Those on month-to-month contracts are more prone to churn. Secondly, customers with DSL internet service are less likely to churn, indicating satisfaction with this service. Thirdly, customer loyalty plays a role; those who have been with the company for over 15 months are less likely to churn. To evaluate the predictive accuracy of this model, the study applied it to the test data subset. The confusion matrix is used, a valuable tool for visualizing classification accuracy, to assess how well the model predicts customer churn. This provided a clear picture of the model's performance. Confusion matrix provides a more detailed breakdown of correct and incorrect classifications made by the model:

- True Positives (TP): These are the cases where the model predicted that the customer would churn (Yes), and they did indeed churn.

- True Negatives (TN): These are the cases where the model predicted that the customer would not churn (No), and they didn’t churn.

- False Positives (FP): These are the cases where the model predicted that the customer would churn (Yes), but they didn’t churn. This is also known as a “Type I error.”

- False Negatives (FN): These are the cases where the model predicted that the customer would not churn (No), but they did churn. This is also known as a “Type II error.”

```{r}
dt_prob1 <- predict(dt_fit, test_df)
dt_pred1 <- ifelse(dt_prob1[,2] > 0.5,"Yes","No")
table(Predicted = dt_pred1, Actual = test_df$Churn)
```
Based on the result of the test data, True Negatives (TN): The model correctly predicted ‘No’ 1410 times, which means it correctly identified 1410 customers who did not churn. False Positives (FP): The model incorrectly predicted ‘Yes’ 295 times, meaning it mistakenly identified 295 customers as churned when they did not. False Negatives (FN): The model incorrectly predicted ‘No’ 138 times, meaning it failed to identify 138 customers who did churn. True Positives (TP): The model correctly predicted ‘Yes’ 265 times, which means it correctly identified 265 customers who did churn.

```{r}
dt_prob2 <- predict(dt_fit, train_df)
dt_pred2 <- ifelse(dt_prob2[,2] > 0.5,"Yes","No")
dt_tab1 <- table(Predicted = dt_pred2, Actual = train_df$Churn)
dt_tab2 <- table(Predicted = dt_pred1, Actual = test_df$Churn)
```

```{r}
# Train
confusionMatrix(
  as.factor(dt_pred2),
  as.factor(train_df$Churn),
  positive = "Yes" 
)

```

```{r}
confusionMatrix(
  as.factor(dt_pred1),
  as.factor(test_df$Churn),
  positive = "Yes" 
)
```

The model’s accuracy is 0.7955, which means it correctly predicted the churn status for about 79.55% of the customers in your test set. The Kappa statistic is 0.4212, which measures the agreement between the predicted and actual values, corrected for chance. A Kappa of 1 indicates perfect agreement, while a Kappa of 0 indicates agreement equivalent to chance. Your model’s Kappa of 0.4212 suggests moderate agreement.

The sensitivity (or recall) of the model is 0.4668, indicating that it correctly identified 46.68% of the customers who churned. The specificity is 0.9145, meaning it correctly identified 91.45% of the customers who did not churn.

The positive predictive value (or precision) is 0.6641, indicating that 66.41% of the customers the model predicted would churn actually did churn. The negative predictive value is 0.8257, meaning that 82.57% of the customers the model predicted would not churn actually did not churn.

The balanced accuracy, which is the average of sensitivity and specificity, is 0.6906. This metric gives a more balanced measure of the model’s performance when the classes are imbalanced.

### Random Forest Classifier

A Random Forest Classifier is a robust machine learning algorithm that leverages the power of multiple decision trees for making predictions. It operates by creating a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random forests correct for decision trees' habit of overfitting to their training set. They are versatile, capable of handling both regression and classification tasks, and they work well with both categorical and numerical data. One of the significant advantages of a random forest is its feature importance functionality, which ranks the influence of different features in the prediction process. Despite being slightly more complex and computationally intensive than simple decision trees, Random Forests are a potent tool in predictive analytics due to their flexibility, accuracy, and ease of use.

```{r}
#Set control parameters for random forest model selection
ctrl <- trainControl(method = "cv", number=5, 
                     classProbs = TRUE, summaryFunction = twoClassSummary)

#Exploratory random forest model selection
rf_fit1 <- train(Churn ~., data = train_df, 
                 method = "rf",
                 ntree = 75,
                 tuneLength = 5,
                 metric = "ROC",
                 trControl = ctrl)

```

```{r}
rf_fit1

```

```{r}
rf_fit2 <- randomForest(Churn ~., data = train_df, 
                        ntree = 75, mtry = 2, 
                        importance = TRUE, proximity = TRUE)

#Display variable importance from random tree
varImpPlot(rf_fit2, sort=T, n.var = 10, 
           main = 'Top 10 important variables')
```

Just like the decision tree model, the random forest model also identifies contract status and tenure length as significant predictors of customer churn. However, there are some differences. In the random forest model, the importance of internet service status is less emphasized. Interestingly, the total charges variable, which was not as important in the decision tree model, is now highly emphasized in the random forest model. This suggests that the total amount a customer is charged could be a strong predictor of churn in the random forest model. These differences highlight the unique ways in which these two models process and interpret the data, leading to different sets of identified important features. This underlines the importance of using multiple models to gain a comprehensive understanding of the factors influencing customer churn.

```{r}
rf_fit2
```
```{r}
rf_pred1 <- predict(rf_fit2, test_df)
table(Predicted = rf_pred1, Actual = test_df$Churn)
```

From the resulting confusion matrix, the True Negatives (TN): indicates that the model correctly predicted ‘No’ 1431 times, which means it correctly identified 1431 customers who did not churn.False Positives (FP): The model incorrectly predicted ‘Yes’ 287 times, meaning it mistakenly identified 287 customers as churned when they did not. False Negatives (FN): The model incorrectly predicted ‘No’ 117 times, meaning it failed to identify 117 customers who did churn. True Positives (TP): The model correctly predicted ‘Yes’ 273 times, which means it correctly identified 273 customers who did churn.

```{r}
# Create an importance plot
# Assuming you've already fitted the random forest model (rf_fit2)

# Plot variable importance
var_importance <- importance(rf_fit2)
var_importance_plot <- barplot(var_importance[, 3], names.arg = rownames(var_importance),
                               main = "Variable Importance in Random Forest Model",
                               col = "skyblue", las = 2, cex.names = 0.8)

# Adding labels
text(var_importance_plot, par("usr")[3], srt = 45, adj = c(1, 1), labels = rownames(var_importance), xpd = TRUE, cex = 0.7)

```

```{r}
plot(rf_fit2)
```

```{r}
confusionMatrix(
  as.factor(rf_pred1),
  as.factor(test_df$Churn),
  positive = "Yes" 
)
```

The model’s accuracy is 0.8083, which means it correctly predicted the churn status for about 80.83% of the customers in your test set. The Kappa statistic is 0.4561, which measures the agreement between the predicted and actual values, corrected for chance. A Kappa of 1 indicates perfect agreement, while a Kappa of 0 indicates agreement equivalent to chance. Your model’s Kappa of 0.4561 suggests moderate agreement.

The sensitivity (or recall) of the model is 0.4875, indicating that it correctly identified 48.75% of the customers who churned. The specificity is 0.9244, meaning it correctly identified 92.44% of the customers who did not churn.

The positive predictive value (or precision) is 0.7000, indicating that 70.00% of the customers the model predicted would churn actually did churn. The negative predictive value is 0.8329, meaning that 83.29% of the customers the model predicted would not churn actually did not churn.

The balanced accuracy, which is the average of sensitivity and specificity, is 0.7060. This metric gives a more balanced measure of the model’s performance when the classes are imbalanced.

### Logistic Regression Classifier
A Logistic Regression Classifier is a statistical model used in machine learning for binary classification problems. It uses the logistic function to model the probability of a certain class or event, such as pass/fail, win/lose, alive/dead, or healthy/sick. The output of logistic regression is a probability that the given input point belongs to a certain class. The central premise of Logistic Regression is the assumption that your input space can be separated into two nice 'regions', one for each class, by a linear boundary. It's simple, fast, and provides good performance for problems with relatively low-dimensional feature space. However, it may not perform well when feature space is large or if the decision boundary is not linear. Despite these limitations, it's a popular choice due to its interpretability and ease of implementation.

```{r}
lr_fit <- glm(Churn ~., data = train_df,
          family=binomial(link='logit'))
summary(lr_fit)
```

From the Logistic Regression Classifier model summary above, it can be seen that the ignificant predictors of churn include contract status and tenure length. Customers with one-year or two-year contracts are less likely to churn, as indicated by their negative coefficients. Similarly, customers with longer tenure are less likely to churn. However, customers who pay via electronic check are more likely to churn, as suggested by the positive coefficient for ‘PaymentMethodElectronic check’.

The accuracy of the model can be assessed using the deviance and the Akaike Information Criterion (AIC). The lower these values, the better the model fits the data. The model’s AIC is 4198.8, which can be used for model comparison. The model has gone through 6 iterations of Fisher Scoring to arrive at the final estimates.

```{r}
# Assuming you have a logistic regression model 'model' and a test dataset 'test'
# Get predicted probabilities
probabilities <- predict(lr_fit, newdata = test_df, type = "response")

# Use pROC package to generate ROC curve
roc_obj <- roc(test_df$Churn, probabilities)

# Plot ROC curve
plot(roc_obj, main="ROC Curve")
abline(a=0, b=1, lty=2, col="gray")  # Adds a line for reference
```
```{r}
lr_prob1 <- predict(lr_fit, test_df, type="response")
lr_pred1 <- ifelse(lr_prob1 > 0.5,"Yes","No")
table(Predicted = lr_pred1, Actual = test_df$Churn)
```

Similar to the machine learning algorithms, the false negative rate is relatively low, with 1397 correct predictions and 151 incorrect predictions. Although not as minimal, this still indicates a proficient identification of instances that should have been classified positively. Conversely, the false positive rate, with 264 correct predictions and 296 incorrect predictions, surpasses 50%. Despite exceeding this threshold, the model's performance in minimizing false positives outshines the machine learning algorithms, suggesting a more conservative approach in erroneously classifying instances negatively, which can be advantageous in scenarios prioritizing precision over recall.

```{r}
# Test
confusionMatrix(
  as.factor(lr_pred1),
  as.factor(test_df$Churn),
  positive = "Yes" 
)

```

The Logistic Regression Classifier model correctly predicted 'No' 1397 times and 'Yes' 296 times, which are the True Negatives (TN) and True Positives (TP) respectively. However, the model incorrectly predicted 'Yes' 264 times and 'No' 151 times, which are the False Positives (FP) and False Negatives (FN) respectively. The model's accuracy is 0.8031, indicating it correctly predicted the churn status for about 80.31% of the customers. The Kappa statistic is 0.4607, suggesting moderate agreement between the predicted and actual values. The sensitivity (or recall) of the model is 0.5286, and the specificity is 0.9025. The positive predictive value (or precision) is 0.6622, and the negative predictive value is 0.8411. The balanced accuracy is 0.7155.

## Discusion 
Referencing from the study results, the churn rate among customers with month-to-month contracts is significantly higher compared to those with longer contracts. This observation aligns with the general expectation that customers who are willing to commit to longer contracts exhibit a lower likelihood of churn. The commitment to a long-term contract could be indicative of the customer's satisfaction with the service, their unwillingness to go through the process of switching providers, or their belief in the value they receive from the service. Therefore, strategies aimed at encouraging customers to opt for longer contracts could potentially be effective in reducing churn rates. However, it's important to ensure that these strategies are paired with efforts to maintain high service quality and customer satisfaction, as these are key factors in a customer's decision to stay with a service provider.

```{r}
ggplot(df, aes(x = Contract, fill = Churn)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) +
  labs(title="Churn rate by contract status")+
   ylab("Count")
```


```{r}
ggplot(df, aes(x = Tenure, fill = Churn)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) +
  labs(title="Churn rate by tenure")+
  ylab("Count")
```
It is also found that the likelihood of churn tends to decrease as the length of time a customer stays with a service increases. However, the notable spike at one month suggests a significant portion of customers decide to leave after just one month of service. This could be due to various reasons such as dissatisfaction with the service, better offers from competitors, or other factors. It’s crucial for service providers to understand and address these early churn triggers to improve customer retention, especially during the initial stages of the customer journey.


```{r}
ggplot(df, aes(x = InternetService, fill = Churn)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) +
  labs(title="Churn rate by internet service status")+
  ylab("Count")
```
It is found that customers with internet service, particularly those with fiber optic internet service, appear to have a higher likelihood of churn compared to those without internet service. This could be due to various factors such as the quality of the internet service, pricing, or customer service experiences. It’s crucial for service providers to understand these dynamics and address the underlying issues to reduce churn rates, especially among fiber optic internet service users. They could focus on improving service quality, offering competitive pricing, or enhancing customer service to retain these customers.

It is also observed that the Customers who have spent more with the company are generally less likely to churn. This could be a reflection of the tenure effect, as longer-tenured customers tend to have higher total spending. Alternatively, it could be indicative of the customers’ financial characteristics. Customers who are more financially stable might be less sensitive to price changes and therefore less likely to churn. Understanding these dynamics can help service providers tailor their customer retention strategies more effectively.

```{r}
ggplot(df, aes(x = MonthlyCharges, fill = Churn)) +
  geom_histogram(binwidth = 100) +
  labs(x = "Dollars (binwidth=100)",
       title = "Churn rate by tenure")
```

## Conclusion

In the course of our study, we embarked on several preparatory steps that included the loading of data and libraries, as well as preprocessing. This was a crucial phase that set the stage for the subsequent analysis. Our focus was on churn analysis, a critical aspect in understanding customer behavior and loyalty. To this end, we employed three statistical classification methods that are commonly used in this field. These methods provided us with a robust framework to dissect the churn phenomenon and understand its underlying factors.

From these models, we were able to identify several important variables that act as predictors of churn. These variables are key indicators that can signal whether a customer is likely to cease using a service or product, thus helping businesses to preemptively address these issues and retain their customer base.

In addition to identifying these churn predictors, we also compared the models on their accuracy measures. This comparative analysis allowed us to gauge the effectiveness of each model, providing valuable insights into their respective strengths and weaknesses.

Our findings from this comprehensive study can be summarized as follows:

- Customers who are on month-to-month contracts are less likely to churn. This suggests that the flexibility of short-term contracts may contribute to customer retention.
- Interestingly, we found that customers with internet service, particularly those with fiber optic service, are more likely to churn. This could be due to various factors such as service quality, pricing, or competition in the market.
- Lastly, our analysis revealed that customers who have been with the company for a longer period or have made higher total payments are less likely to churn. This underscores the value of fostering long-term relationships with customers and providing them with satisfactory service.

These findings provide a valuable foundation for developing effective strategies to reduce customer churn and enhance customer loyalty.





















































