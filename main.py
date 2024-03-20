import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# fixing a bug with replace
pd.set_option("future.no_silent_downcasting", True)

with open(file='train.csv') as file:
    data_train = pd.read_csv(file)

data = data_train.dropna()  # dell null string
data = data.dropna()

all_head = np.array(['hotel','is_canceled','lead_time,arrival_date_year','arrival_date_month',
                     'arrival_date_week_number','arrival_date_day_of_month',
                     'stays_in_weekend_nights','stays_in_week_nights','adults,children','babies',
                     'meal','country','market_segment','distribution_channel','is_repeated_guest',
                     'previous_cancellations','previous_bookings_not_canceled',
                     'reserved_room_type','assigned_room_type','booking_changes','deposit_type',
                     'days_in_waiting_list','customer_type','adr','required_car_parking_spaces',
                     'total_of_special_requests','reservation_status_date'])


# methods for visualizing data allow you to determine which data is not important
'''
table = pd.crosstab(data.hotel, data.is_canceled)
print(table)
print(sum(table[1]))
print(table.iloc[1][1])

table = pd.crosstab(data.distribution_channel, data.is_canceled)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Dependence of hotel and cancellation rate')
plt.xlabel('arrival date month')
plt.ylabel('Number of cancellations')
plt.show()

pd.crosstab(data.distribution_channel,data.is_canceled).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')
plt.show()
'''

# tags that do not influence decision-making:
unnecessary_data = np.array(['adr', 'lead_time', 'arrival_date_year',
                             'reservation_status_date', 'days_in_waiting_list',
                             'country', 'meal', 'stays_in_weekend_nights'])

data.drop(unnecessary_data, axis=1, inplace=True)  # removing unnecessary tags

# replacing names with numbers:
data = data.replace(['No Deposit', 'Non Refund', 'Refundable'],
                    [(8982/31253), (4295/44), (7/57)])
data = data.replace(['Resort Hotel', 'City Hotel'], [(9919/19416), (3365/11938)])
data = data.replace(['April', 'August', 'December', 'February', 'January', 'July', 'June', 'March', 'May', 'November', 'October', 'September'],
                    [(1325/2793), (1597/3635), (691/1792), (821/2285), (525/1738), (1390/3322), (1407/2693), (956/2722), (1466/2870), (609/1968), (1330/2877), (1167/2659)])
data = data.replace(['Aviation', 'Complementary', 'Corporate', 'Direct', 'Groups', 'Offline TA/TO', 'Online TA'],
                    [(15/69), (21/287), (288/1758), (595/4327), (3595/3273), (2461/6624), (6306/15016)])
data = data.replace(['Corporate', 'Direct', 'GDS', 'TA/TO'],
                    [(417/2118), (777/4945), (19/64), (12071/24227)])
# ?
data = data.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'P'],
                    [(10070/21827), (117/316), (89/266), (1820/5536), (609/1879), (254/828), (241/543), (82/159), (2)])
data = data.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'P'],
                    [(9863/17206), (166/681), (130/779), (1913/7925), (619/2403), (262/1133), (244/728), (83/202), (1/169), (1/128), (2)])
data = data.replace(['Contract', 'Group', 'Transient', 'Transient-Party'],
                    [(371/1223), (19/217), (11011/21971), (1883/1883)])
print(data['arrival_date_month'])

# define the predictor variables and the response variable
y = data['is_canceled']
data.drop('is_canceled', axis=1, inplace=True)
X = data

# split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# print(X_train)

# instantiate the model
log_regression = LogisticRegression()
