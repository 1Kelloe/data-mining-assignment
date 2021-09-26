from re import L
import numpy as np
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

st.set_page_config(layout='wide')

st.title("Question 3")

cases_malaysia = pd.read_csv(
    "./covid19-public-main/epidemic/cases_malaysia.csv")
cases_state = pd.read_csv(
    "./covid19-public-main/epidemic/cases_state.csv")
tests_malaysia = pd.read_csv(
    "./covid19-public-main/epidemic/tests_malaysia.csv")
tests_state = pd.read_csv(
    "./covid19-public-main/epidemic/tests_state.csv")

states = list(cases_state['state'].unique())


def returnStateTotal(state):
    cases = []
    for i in range(len(cases_state['cases_new'])):
        temp = cases_state['state'][i]
        if temp == state:
            cases.append(cases_state['cases_new'][i])

    return sum(cases)


def totalPCR(state):
    tests = []
    for i in range(len(tests_state['pcr'])):
        temp = tests_state['state'][i]
        if temp == state:
            tests.append(tests_state['pcr'][i])

    return sum(tests)


def totalRTK(state):
    tests = []
    for i in range(len(tests_state['rtk-ag'])):
        temp = tests_state['state'][i]
        if temp == state:
            tests.append(tests_state['rtk-ag'][i])

    return sum(tests)


cases_total = []

for state in states:
    cases_total.append(returnStateTotal(state))

rtk_total = []

for state in states:
    rtk_total.append(totalRTK(state))

pcr_total = []

for state in states:
    pcr_total.append(totalPCR(state))

state_totals = {'State': states,
                'Cases': cases_total}

tests_types = {"test_type": ['rtk-ag', 'pcr'],
               "total": [tests_malaysia["rtk-ag"].sum(), tests_malaysia["pcr"].sum()]}

fig_1 = px.line(cases_malaysia, y='cases_new', x='date',
                labels={
                    "date": "Date",
                    "cases_new": "New Cases",
                },
                title="Covid-19 Cases in Malaysia")


fig_2 = px.line(cases_state, y='cases_new', x='date', color='state',
                labels={
                    "cases_new": "New Cases",
                    "date": "Date",
                    "tate": "State"
                },
                title="Cases by State")


fig_3 = px.bar(state_totals, x='State', y='Cases',
               title='Total Cases by State')

fig_4 = px.box(state_totals, y='Cases', title='Boxplot of Total State Cases')

fig_5 = px.line(tests_malaysia, y=tests_malaysia.columns[1:3], x='date',
                labels={
                    "date": "Date",
                    "value": "Tests",
                    "variable": "Type of Tests"
},
    title="Covid-19 Tests in Malaysia")


fig_6 = px.bar(tests_types, x='test_type', y='total',
               labels={
                   "test_type": "Test Type",
                   "total": "Total"
               },
               title="Total Type of Tests")

fig_7 = px.line(tests_state, y='rtk-ag', x='date', color='state',
                labels={
                    "date": "Date",
                    "state": "State"
                },
                title="RTK-Ag Tests by State")

fig_8 = px.line(tests_state, y='pcr', x='date', color='state',
                labels={
                    "date": "Date",
                    "state": "State"
                },
                title="PCR Tests by State")

state_tests_total = {"State": states,
                     "rtk-ag": rtk_total,
                     "pcr": pcr_total}

fig_9 = px.bar(state_tests_total, x='State', y=['rtk-ag', 'pcr'], barmode='group',
               labels={
    "value": "Tests"
},
    title='Tests Type by Test')


#Question (i)
st.header("(i)")

col1, col2, col3 = st.columns((1, 1, 1))

with col1:
    if st.checkbox("Display Missing Data"):
        with col1:
            st.write("Missing values for 'cases_malaysia.csv'")
            st.write(cases_malaysia.isnull().sum())

        with col2:
            st.write("Missing values for 'cases_state.csv'")
            st.write(cases_state.isnull().sum())

        with col3:
            st.write("Missing values for 'tests_malaysia.csv'")
            st.write(tests_malaysia.isnull().sum())

            st.write("Missing values for 'tests_state.csv'")
            st.write(tests_state.isnull().sum())

col4, col5 = st.columns((1, 1))

with col2:
    if st.checkbox("Display Cases Data"):
        with col4:
            st.plotly_chart(fig_1)
            st.plotly_chart(fig_2)

        with col5:
            st.plotly_chart(fig_3)
            st.plotly_chart(fig_4)

with col3:
    if st.checkbox("Display Tests Data"):
        with col4:
            st.plotly_chart(fig_5)
            st.plotly_chart(fig_7)
            st.plotly_chart(fig_6)

        with col5:
            st.plotly_chart(fig_9)
            st.plotly_chart(fig_8)


st.markdown(
    """As we can see by the line plot of the cases over time, Covid in Malaysia really ramped up after the start of 2021. 
    We can also see that Selangor(627k) leads the pack in terms of cases by more than 400k compared to second place Kuala Lumpur(177k). 
    It is also noted that Selangor is the only mathematical outlier standing miles away from the upper fence. Unfortunately
    we are not looking at the number of cases with the population of the states in mind, meaning that the bar plots may not tell the full story.
    It maybe the cases that a state is in a more severe state in relation with the population but does not show in the plots due to them 
    only showing volume.""")

st.markdown("""The total test type is split around 42:58 in favour of PCR. Suprisingly though Selangor seems to be one of the two states that
            seemingly favour RTK-Ag tests heavily alongside Pulau Pinang. The rest of the states are either failry balanced between the two test types
            or favour the PCR test type. It is also noted that there seems to be a cycle for the tests done for both test types. The cycle seems to always
            rapidly rise and peak which is then followed by a quick fall in testing. Initially, I had thought that this was a 7 day cycle with the peak
            being on a weekend and the dip being around Monday. However, after taking a closer look, that does not seem to be the case as there are 
            peaks which occured on a Tuesday or a Wednesday, which directly contradicts assumption.""")


#Question (ii) work
new_df = {'Date': []}
date_list = []

date_list = list(cases_state['date'].unique())
new_df['Date'] = date_list

for state in states:
    new_df[state] = []

for i in range(len(cases_state['date'])):
    temp = cases_state['state'][i]
    new_df[temp].append(cases_state['cases_new'][i])

new_df = pd.DataFrame.from_dict(new_df)

fig_10 = px.imshow(new_df.corr(method='kendall'))

temp1 = new_df[['Date', 'Pahang', 'Melaka', 'Kelantan', 'Terengganu']].copy()

fig_11 = px.line(temp1, y=temp1.columns[1:5], x='Date',
                 labels={
    "value": "New cases",
    "variable": "States"
})

temp2 = new_df[['Date', 'Johor', 'Selangor', 'Perak', 'Terengganu']].copy()

fig_12 = px.line(temp2, y=temp2.columns[1:5], x='Date',
                 labels={
    "value": "New cases",
    "variable": "States"
})

#Question (ii)

st.header("(ii)")

col6, col7, col8 = st.columns((1, 1, 1))

with col7:
    st.plotly_chart(fig_10)

st.table(new_df.corr(method='kendall'))
st.markdown("""From the heatmap and table above, we can see that most states have a positive
            correlation with all the other states. W.P Labuan seems to be the one outlier
            among the states with them having less than 0.5 correlation value with all other
            states. This would implicate that the cases in W.P Labuan are less dependant than 
            most other states.""")

st.markdown("""According to table, the states that show the highest correlation to Pahang are
            Melaka, Kelantan and Terengganu. The states that show the lowest correlation with
            Pahang are Sabah and the outlier W.P Labuan, with both of them showcasing a
            correlation value of around 0.5.""")

col9, col10, col11 = st.columns((1, 1, 1))

with col10:
    st.plotly_chart(fig_11)

st.markdown("""When in comes to Johor, all of the states have a correlation value greater than
            0.5 except W.P Labuan which is the outlier that we highlighted earlier. Perak,
            Selangor and Terengganu show the highest correlation with Johor, meaning that it is
            often the case that these states show the same fluctuations that Johor experience.""")

col12, col13, col14 = st.columns((1, 1, 1))

with col10:
    st.plotly_chart(fig_12)

st.markdown("""From the plot above, we can see that Selangor and Johor experienced a similarly
            shaped spike between January 2021 and March 2021. We can see another spike that both
            states shared around April 2021 and June 2021. For the third time we see a similar
            shaped influx around July 2021. However this time, Selangor's spike lasted shorter
            compared to Johor which is still experiencing rise in cases (upto the time when the
            dataset was downloaded). I suspect that if Selangor's cases were still increasing,
            the correlation value would be higher than it currently is.""")

st.markdown("""As for Perak and Terengganu, they both share a similar spike with Johor during
            April 2021 and July 2021. But during the spike in January 2021 both states didn't
            fully align with Johor. Perak for example, saw a rise in cases early February 2021
            and peaked in late February 2021, whereas Johor started rising in late December 2020
            and peaked once in January 2021 and another time in early February 2021.""")


fig_13 = px.line(new_df, y='Pahang', x='Date',
                 labels={
                    "value": "New cases"
                })


#Question iii Work Area
result = pd.merge(cases_state, tests_state, on=['date', 'state'])
result.head()
result = result.drop('date', axis=1)

def stateDF(state):
    df = result[result['state'] == state]
    df = df.drop('state', axis=1)
    df.dropna(subset=['cases_recovered'], inplace=True)
    return df

fig_14 = px.imshow(stateDF('Pahang').corr())
fig_15 = px.imshow(stateDF('Kedah').corr())
fig_16 = px.imshow(stateDF('Johor').corr())
fig_17 = px.imshow(stateDF('Selangor').corr())

def feature_select(state):
    df = stateDF(state)
    X = df.drop('cases_new', axis=1)
    y = df['cases_new']

    bestfeatures = SelectKBest(score_func=chi2, k='all')
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  
    
    return featureScores

#Question iii Answer
st.table(result.corr())
st.markdown("""For the overall cases, cases imported is a poor inidcator towards the daily cases count. Cases recovered and the the other testing values on the other
            hand are great indicators for the count of daily cases. But the best indicator for daily cases is cases recoverd.""")

st.subheader('Pahang')
st.table(feature_select('Pahang'))

col15, col16 = st.columns((1, 1))

with col15:
    st.table(stateDF('Pahang').corr())

with col16:
    st.plotly_chart(fig_14)

st.markdown("""According to the correlation function, cases recovered is the strongest indicator for daily cases,
            whereas SelectKBest predicts that the PCR test is the best indicator for daily cases.""")


st.subheader('Kedah')
st.table(feature_select('Kedah'))

col17, col18 = st.columns((1, 1))

with col17:
    st.table(stateDF('Kedah').corr())

with col18:
    st.plotly_chart(fig_15)

st.markdown("""As for Kedah, both the correlation function and the SelectKBest both predict that the strongest
            indicator for daily cases is cases recovered.""")

st.subheader('Johor')
st.table(feature_select('Johor'))

col19, col20 = st.columns((1, 1))

with col19:
    st.table(stateDF('Johor').corr())

with col20:
    st.plotly_chart(fig_16)

st.markdown("""Once again, both functions predicted the same indicator for daily cases, which in this state is the RTK-Ag tests.""")

st.subheader('Selangor')
st.table(feature_select('Selangor'))

col19, col20 = st.columns((1, 1))

with col19:
    st.table(stateDF('Selangor').corr())

with col20:
    st.plotly_chart(fig_17)

st.markdown("""For the last state of Selangor, the correlation function guessed that the best predictor is cases recovered.
            The SelectKBest function gave the highest score towards the RTK-Ag tests as the best predictor for daily cases.""")


#Question iv Work Area

result = pd.merge(cases_state, tests_state, on=['date', 'state'])
result.head()
result = result.drop(['date', 'cases_import'], axis=1)


def linearRegression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    lin = LinearRegression(positive=True)
    lin.fit(X_train, y_train)
    score = lin.score(X_test, y_test)
    y_pred = lin.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)

    return score, mae


def DTRegressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    regressor = DecisionTreeRegressor(
        max_depth=2, criterion="mae", splitter="best")
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    y_pred = regressor.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)

    return score, mae


def ridge(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    ridge = Ridge(normalize=False, tol=0.001,
                  solver='saga', fit_intercept=True,
                  random_state=42)
    ridge.fit(X_train, y_train)
    score = ridge.score(X_test, y_test)
    y_pred = ridge.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)

    return score, mae


def lasso(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    lasso = Lasso(positive=True,
                  alpha=1)
    lasso.fit(X_train, y_train)
    score = lasso.score(X_test, y_test)
    y_pred = lasso.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)

    return score, mae

totalScore = pd.DataFrame(columns=['State', 'Linear Regression Score', 'Linear Regression MAE', 'Decision Tree Regression Score',
                          'Decision Tree Regression MAE', 'Ridge Regression Score', 'Ridge Regression MAE', 'Lasso Regression Score',
                          'Lasso Regression MAE'])


def getScores(state):
    df = stateDF(state)

    X = df.drop('cases_new', axis=1)
    y = df['cases_new']

    lin_score = (linearRegression(X, y))[0]
    lin_mae = (linearRegression(X, y))[1]

    regressor_score = (DTRegressor(X, y))[0]
    regressor_mae = (DTRegressor(X, y))[1]

    ridge_score = (ridge(X, y))[0]
    ridge_mae = (ridge(X, y ))[1]

    lasso_score = (lasso(X, y))[0]
    lasso_mae = (lasso(X, y))[1]

    totalScore.loc[len(totalScore)] = [state, 
                                       lin_score,lin_mae, 
                                       regressor_score, regressor_mae,
                                       ridge_score, ridge_mae,
                                       lasso_score, lasso_mae]


# Question iv Answers
st.header('(iv)')
getScores('Pahang')
getScores('Kedah')
getScores('Johor')
getScores('Selangor')

st.table(totalScore)

st.markdown("""For Pahang the best scoring models were basically split 3 ways between Linear Regression, Ridge Regression and 
            Lasso Regression. With all of them having an MAE of 86. We see similar cases for the other two states of Kedah and 
            Johor. However, for Selangor, the best scoring model was actually the Decision Tree Regression. The model that performed 
            the worse between the last 3 states. Why is this the case? I believe this is because Selangor's case is distinctly
            different compared to the other 3 states in terms of volume of cases. This would mean that the Decision Tree
            Regression scores better for states that are experiencing a higher volume of cases.""")