# Project Telco

Learn to discern what turns the churn burn

### Project Description

Telco is a telecommunications company that offers many services to a wide range of customers. I have decided to look into the different elements of a customer churning to determine if any of them increase or decrease the chance of a customer to churn.

### Project Goal

* Discover drivers of churn of Telco customers
* Use drivers to develop a machine learning model to classify churn as a customer ending their contract or not ending (renewing) their contract with Telco
* This information could be used to further our understanding of which customer elements contribute to or detract from a customer churning.

### Initial Thoughts

My initial hypothesis is that ...

## The Plan

* Acquire data from Codeup MySQL DB
* Prepare data

  * Create Engineered columns from existing data
* Explore data in search of drivers of churn

  * Answer the following initial questions
    * How often does churn occur?
    * 
* Develop a Model to predict if a customer will churn

  * Use drivers identified in explore to build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on highest accuracy
  * Evaluate the best model on test data
* Draw conclusions

## Data Dictionary

| Feature               | Definition                                                          |
| :-------------------- | :------------------------------------------------------------------ |
| customer_id           | Alpha-numeric, unique ID for each customer                          |
| gender                | Female=1/Male=0, gender of customer                                 |
| senior_citizen        | True=1/False=0, whether customer is a senior citizen or not         |
| partner               | True=1/False=0, whether customer has a partner or not               |
| dependents            | True=1/False=0, whether customer has dependents or not              |
| phone_service         | True=1/False=0, whether customer has phone service or not           |
| multiple_lines        | Yes/No/No phone service, whether customer has multiple lines or not |
| internet_service_type | None/DSL/Fiber Optic, which internet service customer has           |
| online_security       | Yes/No/No internet service, whether customer has online_security   |
| online_backup         | Yes/No/No internet service, whether customer has online_backup     |
| device_protection     | Yes/No/No internet service, whether customer has device_protection |
| tech_support          | Yes/No/No internet service, whether customer has tech_support      |
| streaming_tv          | Yes/No/No internet service, whether customer has streaming_tv      |
| streaming_movies      | Yes/No/No internet service, whether customer has streaming_movies  |
| contract_type         | Month-to-Month/One-year/Two-year, term of contract                  |
| payment_type          | 4 options (2 automatic), customer payment method                    |
| paperless_billing     | True=1/False=0, whether a customer has paperless billing enabled    |
| monthly_charges       | Numeric USD, amount customer is charged monthly                     |
| total_charges         | Numeric USD, total amount customer has been charged                 |
| tenure                | Numeric, number of months customer has stayed                       |
| churn (target)        | True=1/False=0, whether or not the customer has churned             |
| Additional Features   | Encoded values for categorical data                                 |
| tenure_bin            | numeric, tenure months binned to years (1 'year' is <=12 'months)   |
| internet_packages     | numeric, number of internet services added together                 |

## Steps to Reproduce

1) Clone this repo
2) If you have access to the Codeup MySQL DB:
   - Save **env.py** in the repo w/ `user`, `password`, and `host` variables
   - Run notebook
3) If you don't have access:
   - Acquire the data somewhere else
   - Save as **telco.csv** in the repo
   - Run notebook

# Conclusions

### Takeaways and Key Findings

### Recommendations
