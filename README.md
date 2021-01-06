### SyriaTel Churn Data
By: Eduardo Osorio

## Objective:
The objective of this report is to find any correlations that can help us
accurately predict when a customer is about to Churn. With this information we
should be able to help create recommendations to help increase customer
retention. For the purpose of this study, I will be focusing on increasing the
recall score.

## Sources Used:
- SyriaTel Customer Churn Datasheet
- https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
- https://xgboost.readthedocs.io/en/latest/
- scikit-learn.org

## Summery Of The Results:
The final conclusions are:
- If a customer is enrolled in a plan like "International calling" or "Voice
Mail", they churn at a lesser rate then customers not enrolled.
- If a customer calls more then 3 times, the chances of them Churning go up.

## The Data:
The data provided took customer account information like:
- Age of the account
- If a customer is enrolled in a plan
- Time of day usage
- Number of Customer service calls

## EDA:
- The "phone number" and "state" columns were removed because they didn't have
anything to offer.
- There was no null values to drop.
- Even though some columns showed collinearity, I choose not to remove because
they're important to the purpose of this study.

## Baseline Model:
For the base model, I created a function to run multiple models along with their
precision, recall, accuracy, and F1 scores. This allowed me to quickly see which
classifiers work best on this data. The two winning models in this case were
Random Forests and **XGBoost**. However, due to the **minority class**(Customers who
churned) only counting for about **15%** of the dataset

**RandomForest Base Model**

![alt text](https://github.com/Eduardoosorio23/Mod_3_project/blob/main/Data/Images/RandomForest%20Matrix%20Base.png?raw=true")

**XGBoost Base Model**

![alt text](https://github.com/Eduardoosorio23/Mod_3_project/blob/main/Data/Images/XGBoost%20Matrix%20Base.png?raw=true")
## Random Forests Model:
In order to address the class imbalance issue I used **Tomek Links** to help separate
the minority(Customers that churned) and majority classes(Customers that stayed).
Then, I proceeded to use **SMOTE** to artificially create new minority samples.
This help address the imbalance but the classifier was still overfitting the
training data. I used **GridSearch** to help find the optimal parameters. Once
I plugged in the new parameters, I was able to fix the train set overfitting. The
**testing recall score** did drop by **.009** but since the **training recall** dropped from
**1.0**, I believe this model is more accurate overall.


 ![alt text](https://github.com/Eduardoosorio23/Mod_3_project/blob/main/Data/Images/RandomForest%20Matrix%20final.png?raw=true")


## XGBoost Model:
For this model, I used **Tomek Links** like the previous Random Forest model
to address the class imbalance. I did not use SMOTE for this **XGBoost** Model since
it actually lowered the recall score. Like the Random Forest Model, I also used
**GridSearch** to find the optimal parameters. After I plugged them in and refitted
the model, the recall score went up **35%**. Even though this model is **2%**
less accurate than the Random Forest model, The confusion matrix seems to suggest
that it categorizes **True Negatives** and **True Positives** more accurately.

![alt text](https://github.com/Eduardoosorio23/Mod_3_project/blob/main/Data/Images/XGBoost%20Matrix%20final.png?raw=true")

According to The models, The most important features were:
- Customer service calls
- International calling plan
- Voice mail plan

![alt text](https://github.com/Eduardoosorio23/Mod_3_project/blob/main/Data/Images/Important%20Features%20RF%202.png?raw=true") ![alt text](https://github.com/Eduardoosorio23/Mod_3_project/blob/main/Data/Images/Important%20Features%20XGB.png?raw=true")

## Recommendations:
- More plans like their international calling plan
- If a customer calls **customer service** more than **3** times, SyriaTel should look
into maybe giving that customer more attention or better deals to prevent them from leaving.
- If they suspect a customer might be leaving soon, they could offer them a
special promotional deal to hold on to the customer.

## Next Steps:
- With more time, I would like to explore the relationship between customer churn
 and time of day usage. I believe with this information, SyriaTel could create
 certain plans to maybe offer free minutes at night or first x amount of minutes are free.
- Also I believe if we find more data on people leaving, we could greatly
increase the accuracy of the model.
