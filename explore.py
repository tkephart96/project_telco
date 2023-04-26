import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats

### focused exploration ###

def cat_chi(train, target, cat_var):
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'Chi2: {chi2}, p-value: {p}')

def explore_cat(train, target, cat_var):
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'Chi2: {chi2}, p-value: {p}')
    p = sns.barplot(target, cat_var, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[target].mean()
    p = plt.axvline(overall_rate, ls='--', color='gray')
    plt.show(p)

def explore_int(train, target, cat_var):
    observed = pd.crosstab(train[target], train[cat_var])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'Chi2: {chi2}, p-value: {p}')
    p = sns.barplot(cat_var, target, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[target].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    plt.show(p)

def baseline(target):
    print(f'Baseline: {round(((target==target.value_counts().idxmax()).mean())*100,2)}% Accuracy')

### broad exploration ###

def explore_univariate(train, cat_vars, quant_vars):
    """
    The function explores univariate categorical and quantitative variables in a given dataset.
    
    :param train: This parameter is likely a pandas DataFrame containing the training data for a machine
    learning model
    :param cat_vars: A list of categorical variables in the dataset
    :param quant_vars: Quantitative variables, also known as numerical variables, are variables that
    represent a numeric value, such as age, income, or height. These variables can be measured and
    analyzed using mathematical and statistical methods
    """
    for var in cat_vars:
        explore_univariate_categorical(train, var)
        print('_________________________________________________________________')
    for col in quant_vars:
        p, descriptive_stats = explore_univariate_quant(train, col)
        plt.show(p)
        print(descriptive_stats)

def explore_bivariate(train, target, cat_vars, quant_vars):
    """
    The function explores bivariate relationships between categorical and quantitative variables in a
    given dataset.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable that we are trying to predict or explain in our
    analysis. It is usually a categorical or numerical variable that we want to model or understand its
    relationship with other variables in the dataset
    :param cat_vars: a list of categorical variables in the dataset
    :param quant_vars: A list of quantitative variables (numeric variables) in the dataset that you want
    to explore in relation to the target variable
    """
    for cat in cat_vars:
        explore_bivariate_categorical(train, target, cat)
    for quant in quant_vars:
        explore_bivariate_quant(train, target, quant)

def explore_multivariate(train, target, cat_vars, quant_vars):
    """
    The function explores multivariate relationships between variables in a dataset using various
    visualization techniques.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable that we are trying to predict or model. It is
    usually a categorical variable or a continuous variable that we want to predict based on the other
    variables in the dataset
    :param cat_vars: A list of categorical variables in the dataset
    :param quant_vars: Quantitative variables are variables that have numerical values and can be
    measured on a continuous or discrete scale. Examples of quantitative variables include age, income,
    height, weight, and temperature
    """
    plot_swarm_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    violin = plot_violin_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    pair = sns.pairplot(data=train, vars=quant_vars, hue=target)
    plt.show()
    plot_all_continuous_vars(train, target, quant_vars)
    plt.show()    


### Univariate

def explore_univariate_categorical(train, cat_var):
    """
    This function creates a frequency table and a bar plot for a categorical variable in a given
    dataset.
    
    :param train: The training dataset containing the categorical variable to be explored
    :param cat_var: The categorical variable that we want to explore
    """
    frequency_table = freq_table(train, cat_var)
    plt.figure(figsize=(2,2))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()
    print(frequency_table)

def explore_univariate_quant(train, quant_var):
    """
    The function explores a univariate quantitative variable by creating a histogram and box plot and
    returning descriptive statistics.
    
    :param train: a pandas DataFrame containing the training data
    :param quant_var: The variable name of the quantitative variable that we want to explore
    :return: two values: the plots for the histogram and boxplot of the specified quantitative variable,
    and the descriptive statistics of that variable.
    """
    descriptive_stats = train[quant_var].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant_var])
    p = plt.title(quant_var)
    return p, descriptive_stats

def freq_table(train, cat_var):
    """
    The function creates a frequency table for a categorical variable in a given dataset.
    
    :param train: The training dataset that contains the categorical variable for which we want to
    create a frequency table
    :param cat_var: The categorical variable for which we want to create a frequency table
    :return: The function `freq_table` returns a pandas DataFrame that contains the unique values of a
    categorical variable in a given dataset, along with the count and percentage of each value in the
    dataset.
    """
    class_labels = list(train[cat_var].unique())

    return pd.DataFrame(
        {
            cat_var: class_labels,
            'Count': train[cat_var].value_counts(normalize=False),
            'Percent': round(
                train[cat_var].value_counts(normalize=True) * 100, 2
            ),
        }
    )


#### Bivariate

def explore_bivariate_categorical(train, target, cat_var):
    """
    The function explores the relationship between a categorical variable and a target variable using
    chi-squared test and plots the results.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable we are trying to predict or explain in our
    analysis. It is usually a categorical variable that we want to classify or predict based on the
    values of other variables
    :param cat_var: a categorical variable in the dataset that we want to explore in relation to the
    target variable
    """
    print(cat_var, "\n_____________________\n")
    ct = pd.crosstab(train[cat_var], train[target], margins=True)
    chi2_summary, observed, expected = run_chi2(train, cat_var, target)
    p = plot_cat_by_target(train, target, cat_var)

    print(chi2_summary)
    print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")

def explore_bivariate_quant(train, target, quant_var):
    """
    The function explores the relationship between a quantitative variable and a target variable using
    descriptive statistics, a boxen plot, a swarm plot, and a Mann-Whitney test.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable that we are trying to predict or understand in
    our analysis. It is the dependent variable in a regression or classification problem. In this
    function, it is used to group the data and compare the descriptive statistics and means between
    different groups
    :param quant_var: The quantitative variable that we want to explore in relation to the target
    variable
    """
    print(quant_var, "\n____________________\n")
    descriptive_stats = train.groupby(target)[quant_var].describe()
    average = train[quant_var].mean()
    mann_whitney = compare_means(train, target, quant_var)
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, target, quant_var)
    swarm = plot_swarm(train, target, quant_var)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nMann-Whitney Test:\n", mann_whitney)
    print("\n____________________\n")

## Bivariate Categorical

def run_chi2(train, cat_var, target):
    """
    The function calculates the chi-squared test statistic, p-value, degrees of freedom, and expected
    values for a given categorical variable and target variable in a training dataset.
    
    :param train: This parameter is likely a pandas DataFrame containing the training data for a machine
    learning model. It could include features, labels, and other relevant information for the model
    :param cat_var: The categorical variable for which we want to calculate the chi-squared test
    statistic and p-value
    :param target: The target variable is the variable that we are trying to predict or explain using
    the categorical variable. It is the dependent variable in our analysis
    :return: The function `run_chi2` returns three objects:
    1. `chi2_summary`: a pandas DataFrame containing the chi-square statistic, p-value, and degrees of
    freedom.
    2. `observed`: a pandas DataFrame containing the observed frequencies of the contingency table.
    3. `expected`: a pandas DataFrame containing the expected frequencies of the contingency table.
    """
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected

def plot_cat_by_target(train, target, cat_var):
    """
    This function plots a bar chart of a categorical variable against a target variable and adds a
    horizontal line representing the overall rate of the target variable.
    
    :param train: The training dataset containing the variables to be plotted
    :param target: The target variable is the variable we are trying to predict or explain in our
    analysis. It is usually a binary variable (0 or 1) in a classification problem or a continuous
    variable in a regression problem
    :param cat_var: categorical variable that you want to plot
    :return: a bar plot with the specified categorical variable on the x-axis and the target variable on
    the y-axis. It also includes a horizontal line representing the overall rate of the target variable
    in the dataset.
    """
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(cat_var, target, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[target].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p


## Bivariate Quant

def plot_swarm(train, target, quant_var):
    """
    The function plots a swarmplot with a horizontal line indicating the mean value of a quantitative
    variable for a given target variable.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable that we want to predict or explain using the
    other variables in the dataset. It is usually represented on the x-axis of a plot
    :param quant_var: The quantitative variable that we want to plot on the y-axis of the swarm plot
    :return: the plot object `p`.
    """
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_boxen(train, target, quant_var):
    """
    This function plots a boxenplot with a horizontal line representing the mean of a quantitative
    variable for each category of a target variable in a given dataset.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the categorical variable that we want to compare the
    distribution of the quantitative variable across. For example, if we are analyzing the relationship
    between income and education level, the target variable would be education level (e.g. high school,
    college, graduate degree)
    :param quant_var: The quantitative variable that we want to plot on the y-axis of the boxen plot
    :return: a boxenplot with a horizontal line representing the mean value of the quantitative
    variable, and a title indicating the name of the variable being plotted. The variable `p` is being
    returned, which is the result of the last plotted object (in this case, the title). However, it is
    not necessary to return `p` since it is not being used outside of the function
    """
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

# alt_hyp = ‘two-sided’, ‘less’, ‘greater’

def compare_means(train, target, quant_var, alt_hyp='two-sided'):
    """
    The function compares the means of two groups using the Mann-Whitney U test and returns the test
    statistic and p-value.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is a binary variable that indicates the outcome of interest. In
    this function, it is used to split the data into two groups based on the value of the target
    variable (0 or 1)
    :param quant_var: The quantitative variable that we want to compare the means of between two groups
    :param alt_hyp: The alternative hypothesis for the Mann-Whitney U test. It specifies the direction
    of the test and can be either "two-sided" (default), "less" or "greater". "two-sided" means that the
    test is two-tailed, "less" means that the test is one, defaults to two-sided (optional)
    :return: the result of a Mann-Whitney U test comparing the means of two groups (x and y) based on a
    quantitative variable (quant_var) in a training dataset (train) with a binary target variable
    (target). The alternative hypothesis (alt_hyp) can be specified as either 'two-sided' (default),
    'less', or 'greater'.
    """
    x = train[train[target]==0][quant_var]
    y = train[train[target]==1][quant_var]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)


### Multivariate

def plot_all_continuous_vars(train, target, quant_vars):
    """
    This function plots boxenplots of continuous variables with color representing the target variable.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable that we are trying to predict or model. It is
    usually the dependent variable in a regression analysis or the class label in a classification
    problem
    :param quant_vars: a list of column names in the training dataset that contain continuous variables
    (numeric data that can take on any value within a range)
    """
    my_vars = [item for sublist in [quant_vars, [target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = train[my_vars].melt(id_vars=target, var_name="measurement")
    plt.figure(figsize=(8,6))
    p = sns.boxenplot(x="measurement", y="value", hue=target, data=melt)
    p.set(yscale="log", xlabel='')    
    plt.show()

def plot_violin_grid_with_color(train, target, cat_vars, quant_vars):
    """
    This function plots a grid of violin plots for categorical and quantitative variables with
    color-coded target variable.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable that we are trying to predict or model. It is
    usually the dependent variable in a machine learning problem. In this function, it is used to color
    the violin plots based on the target variable
    :param cat_vars: A list of categorical variables to be plotted on the x-axis of the violin plots
    :param quant_vars: quant_vars are the quantitative variables (numerical variables) that we want to
    plot on the y-axis of the violin plots
    """
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.violinplot(x=cat, y=quant, data=train, split=True, 
                            ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()

def plot_swarm_grid_with_color(train, target, cat_vars, quant_vars):
    """
    This function plots a grid of swarmplots for categorical and quantitative variables with color-coded
    target variable.
    
    :param train: This parameter is likely a pandas DataFrame containing the training data for a machine
    learning model
    :param target: The target variable is a categorical variable that we want to predict or analyze. It
    is used to color-code the swarm plots in the grid
    :param cat_vars: A list of categorical variables to be plotted on the x-axis of the swarm plots
    :param quant_vars: quant_vars are the quantitative variables (numeric variables) that we want to
    plot in the swarm plot
    """
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.swarmplot(x=cat, y=quant, data=train, ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()