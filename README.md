# Project Telco

Learn to discern what turns customers to churn

### Project Description

Telco is a telecommunications company that offers many services to a wide range of customers. I have decided to look into the different elements of a customer churning to determine if any of them increase or decrease the chance of a customer to churn.

### Project Goal

* Discover drivers of churn of Telco customers
* Use drivers to develop a machine learning model to classify churn as a customer ending their contract or not ending (renewing) their contract with Telco
* This information could be used to further our understanding of which customer elements contribute to or detract from a customer churning.

### Initial Thoughts

My initial hypothesis is that drivers of churn will be elements that have dissatisfied customers. Some service/package or lack of one might be pushing customers to churn.

## The Plan

* Acquire data from Codeup MySQL DB
* Prepare data
    * Create Engineered columns from existing data
* Explore data in search of drivers of churn
    * Answer the following initial questions
    * Is Churn independent from payment type?
    * Is Churn independent from internet Service type?
    * Is Churn independent from tech support?
    * Is Churn independent from a customer having at least one internet package?
* Develop a Model to predict if a customer will churn
    * Use drivers identified in explore to build predictive models of different types
    * Evaluate models on train and validate data
    * Select the best model based on highest accuracy
    * Evaluate the best model on test data
* Draw conclusions

## Data Dictionary

| Feature               | Values                      | Definition                                                |
| :-------------------- | --------------------------- | :-------------------------------------------------------- |
| customer_id           | Alpha-numeric               | Unique ID for each customer                               |
| gender                | Female=1/Male=0             | Gender of customer                                        |
| senior_citizen        | True=1/False=0              | Whether customer is a senior citizen or not               |
| partner               | True=1/False=0              | True=1/False=0, whether customer has a partner or not     |
| dependents            | True=1/False=0              | True=1/False=0, whether customer has dependents or not    |
| phone_service         | True=1/False=0              | True=1/False=0, whether customer has phone service or not |
| multiple_lines        | Yes/No/No phone service     | Whether customer has multiple lines or not                |
| internet_service_type | None/DSL/Fiber Optic        | Which internet service customer has                       |
| online_security       | Yes/No/No internet service | Whether customer has online_security                      |
| online_backup         | Yes/No/No internet service | Whether customer has online_backup                        |
| device_protection     | Yes/No/No internet service | Whether customer has device_protection                    |
| tech_support          | Yes/No/No internet service | Whether customer has tech_support                         |
| streaming_tv          | Yes/No/No internet service | Whether customer has streaming_tv                         |
| streaming_movies      | Yes/No/No internet service | Whether customer has streaming_movies                     |
| contract_type         | 3 options                   | Month-to-Month/One-year/Two-year, term of contract        |
| payment_type          | 4 options (2 auto)          | Customer payment method                                   |
| paperless_billing     | True=1/False=0              | Whether a customer has paperless billing enabled          |
| monthly_charges       | Numeric USD                 | Amount customer is charged monthly                        |
| total_charges         | Numeric USD                 | Total amount customer has been charged                    |
| tenure                | Numeric                     | Number of months customer has stayed                      |
| churn (target)        | True=1/False=0              | Whether or not the customer has churned                   |
| Additional Features   | True=1/False=0              | Encoded values for categorical data                       |
| has_internet_packages | True=1/False=0              | Whether customer has at least 1 internet package          |

## Steps to Reproduce

1) Clone this repo
2) If you have access to the Codeup MySQL DB:
    - Save **env.py** in the repo w/ `user`, `password`, and `host` variables
    - Run notebook
3) If you don't have access:
    - Request access from Codeup
    - Do step 2

# Conclusions

### Takeaways and Key Findings

* Payment type was found to be a driver of churn
    - Electronic check being the most common among churn
* Internet service type was found to be a driver of churn
    - Fiber optic being the most common among churn
* Tech support was found to be a driver of churn
    - No tech support being the most common among churn
    - With more time I could check if not having tech support increases churn for certain internet services and packages
* Having an internet package was found to be a driver of churn
    - If given more time I could investigate why without making assumptions
* Gender and phone service were found to not be a driver of churn

### Recommendations
* Check with engineers to see if there are frequent issues with Fiber optic internet
* Check with tech support technicians and see what can be done for customers to choose tech support as an internet package