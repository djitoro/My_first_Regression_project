import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report


def data_processing(f):
    # fixing a bug with replace
    pd.set_option("future.no_silent_downcasting", True)

    def_data = pd.read_csv(f)
    def_data = def_data.dropna()  # dell null string

    # list of data that has no correlation:
    def_unnecessary_data = np.array(['adr', 'lead_time', 'arrival_date_year', 'reservation_status_date',
                                     'days_in_waiting_list', 'country', 'meal', 'stays_in_weekend_nights'])

    def_data.drop(def_unnecessary_data, axis=1, inplace=True)  # removing unnecessary tags

    # replacing names with numbers:
    def_data = def_data.replace(['No Deposit', 'Non Refund', 'Refundable'], [(8982 / 31253), (4295 / 44), (7 / 57)])
    def_data = def_data.replace(['Resort Hotel', 'City Hotel'], [(9919 / 19416), (3365 / 11938)])
    def_data = def_data.replace(
        ['April','August','December','February','January','July','June','March','May','November','October','September'],
        [(1325 / 2793), (1597 / 3635), (691 / 1792), (821 / 2285), (525 / 1738), (1390 / 3322), (1407 / 2693),
         (956 / 2722), (1466 / 2870), (609 / 1968), (1330 / 2877), (1167 / 2659)])
    def_data = def_data.replace(['Aviation','Complementary','Corporate','Direct','Groups','Offline TA/TO','Online TA'],
                        [(15 / 69), (21 / 287), (288 / 1758), (595 / 4327), (3595 / 3273), (2461/6624),(6306/15016)])
    def_data = def_data.replace(['Corporate', 'Direct', 'GDS', 'TA/TO'],[(417/2118), (777/4945), (19/64),(12071/24227)])
    # It's a little strange that this paragraph and the next one are different in length
    def_data = def_data.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'P'],
                        [(10070/21827), (117/316), (89/266), (1820/5536), (609/1879), (254/828),(241/543), (82/159), 2])
    def_data = def_data.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'P'],
                        [(9863 / 17206), (166 / 681), (130 / 779), (1913 / 7925), (619 / 2403), (262 / 1133),
                         (244 / 728), (83 / 202), (1 / 169), (1 / 128), (2)])
    def_data = def_data.replace(['Contract', 'Group', 'Transient', 'Transient-Party'],
                        [(371 / 1223), (19 / 217), (11011 / 21971), (1883 / 1883)])

    # ATTENTION: there are values in the test data set that are not found in the training set, here they are:
    def_data = def_data.replace('L',
                        sum([(9863 / 17206), (166 / 681), (130 / 779), (1913 / 7925), (619 / 2403), (262 / 1133),
                        (244 / 728), (83 / 202), (1 / 169), (1 / 128), (2)]) /
                        len([(9863 / 17206), (166 / 681), (130 / 779), (1913 / 7925), (619 / 2403), (262 / 1133),
                        (244 / 728), (83 / 202), (1 / 169), (1 / 128), (2)]))

    return def_data


with open(file='train.csv') as file:
    data = data_processing(file)
# define the predictor variables and the response variable
y = data['is_canceled']
data.drop('is_canceled', axis=1, inplace=True)
X = data
print(X)
# split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# print(X_train)

# instantiate the model
log_regression = LogisticRegression(max_iter=1500)

# fit the model using the training data
log_regression.fit(X_train, y_train)

# use model to make predictions on test data
y_pred = log_regression.predict(X_test)

# system response matrix:
# true guesses; true not guessed
# false guesses; false not guessed
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)  # judging by this data, we can say that the model lacks data in one of the categories

print(" Accuracy:", metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
# current efficiency = 0.811305256869773
# To increase efficiency, let's try other training models:

# increase the number of iterations
# (max_iter=1000)

# use regularization methods (L1 and L2)
# Create a LogisticRegression instance with L1 regularization and a high penalty
# model = LogisticRegression(penalty='l1', C=0.01, solver='liblinear')

# data normalization

# work:
with open(file='test.csv') as file:
    X_input = data_processing(file)
y_ans = log_regression.predict(X_input)
ans = pd.DataFrame(y_ans)
ans.to_csv('saved_ratings.csv', index=False)
