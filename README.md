**Machine Learning I Group Assignment: Capital Bikeshare**

Lindsay Neff, Lejla Skahic, Prachi Pathak, Stephanie Palanca

26 March 2023

<u>Introduction</u>

Two key factors in the selection of modes of transportation are
reliability and convenience. Capital Bikeshare is an important method of
transportation in DC, but is not immune to operational inefficiencies.
One of the biggest challenges faced by Capital Bikeshare that affects
the aforementioned factors is ensuring daily bike and dock availability.
The lack of availability of bikes for pick up and docks for drop off can
dissuade people from choosing Capital Bikeshare and is a pertinent
concern for us as business analysts. The problem becomes more complex
than predicting the demand for the number of bikes to be picked up and
the number of docks available for drop off, as many variables such as
weather, date and time, season, and traffic conditions have the
potential to influence these two factors. In this report, we outline how
we used data analysis and machine learning techniques to solve this
operational problem, evaluate the predictive models we developed, report
our findings and offer recommendations to Capital Bikeshare to improve
the operational inefficiencies.

<u>The Operational Challenge</u>

The problem we aim to solve is a demand and supply problem: the
imbalance between the demand for bikes and supply of bikes available for
pick up, and the demand for docks and supply of docks available for drop
off. Additionally, the availability of bikes depends on a location’s
dock capacity and the number of drop offs prior to a pick up, while the
availability of docks depends on prior drop offs and the demand for pick
ups. We attempt to provide a solution to this problem by predicting the
demand for these, and hence the required supply for bikes and docks
required.

Ideally we would be able to predict these for all stations. For this
report, we limit the scope of this problem to two stations during a set
time period: 21st St & Pennsylvania Ave NW and 21st St & I St NW for
January 2022 - April 2022. Assumptions we made while creating the model
are discussed in the limitations section.

<u>Data Collection & Wrangling</u>

Our model is based on data collected from the [<u>Bikeshare System
Database</u>](https://ride.capitalbikeshare.com/system-data). We used
data from the months of January 2022-April 2022 for our analysis. In
addition to this, we also used DC historical weather data obtained from
the [<u>Visual Crossing
Database</u>](https://www.visualcrossing.com/weather-history/Washington%2CDC%2CUSA)
and [<u>Crash
data</u>](https://opendata.dc.gov/datasets/crashes-in-dc/explore?location=38.899161%2C-77.011046%2C13.66)
to analyze other factors that could influence demand for bike pick ups
and drop offs. We also sought out data pertaining to traffic,
construction and road closures, as well as events data to understand if
certain factors could increase or decrease demand for bikes around the
city. Unfortunately the data we found did not match our time periods or
was inaccessible.

From the bikeshare data we maintained most variables, but modified date
and time variables to help with our analysis. We split the date time
variable to have a single date column, an hours column (to represent
hour of the day 0-23) as well as a day, month, year column. We also
generated a season variable, month variable, day of the week variable,
and a variable called holiday which indicated if the date was a holiday
or not.

For the weather data we also converted the date to date time and dropped
unnecessary or redundant columns. Ultimately we maintained datetime,
temp, feelslike, humidity, precip, preciptype, snow, windspeed,
cloudcover, visibility, uvindex, conditions, and icon in the weather
dataset. In our exploratory analysis and model building, we further
reduce these variables as some are still highly correlated or similar,
and therefore pose a problem in our prediction efforts which we will
discuss later on.

Lastly, for our crash data it was necessary to substantially modify the
data. Our motivation to use the crash data was to consider whether an
accident in the vicinity of those two stations might contribute to
blockage or traffic and compel or reduce the demand for using bikeshare.
We used a map of DC’s police wards to evaluate the relevant zones we
should include in our analysis and subsetted the crash data to just
those in Ward 2 which might influence bike usage for the two stations in
our analysis. We then counted the number of crashes in Ward 2 for that
day and created a new column. Finally we merged all three data sets on
date to work from one dataframe in our model analysis and generation.

<u>Exploratory analysis</u>

We approached exploratory analysis through the lens of demand for bike
pick up and dock drop offs from 4 different perspectives: who bikes were
most popular with, the effect of weather and seasons, the effect of
different time dimensions, and road accidents.

Our analysis showed us that members used bikes 3 times more often than
non-members for both streets. This could suggest that bikes are more
often used for regular transport such as commutes or running errands as
compared to leisure activities like tourism during holidays. This is
further supported by our finding that there is a higher demand for bikes
on weekdays compared to weekends, with peak average pickups occurring
midweek. So, we decided to investigate how commuter demand could be an
influential factor in demand for pick ups and drop offs, which we
investigated by using the hour of the day as a predictor in our model.

As predicted, we also found that Spring is significantly more popular
for biking than Winter. Based on this finding, we decided that making
predictions using specific weather conditions would be necessary in
building a predictive model. Likely warmer temperatures and better
conditions contribute to increased demand in bike share usage.

In investigating time further, we found that for 21st and I Street, peak
pickups occurred around hour 15 to hour 17 while peak drop offs occurred
around hour 8. For 21st and Pennsylvania, peak pickups occurred around
hour 15 or 16 while peak drop offs occurred also around hour 8 or 9.

We also investigated the relationship between all of the weather
variables in our exploratory analysis. We used a heat map and
correlation matrix to evaluate the degree of correlation between these
variables. It is clear that similar variables like temperature,
temperature min and max, feels like and others are all highly
correlated. This further validates our assumption that we should remove
some of the redundant weather variables whose effect might be captured
in another one of the variables.

\*21st & I Exploratory [<u>Graphs</u>](#czg5gr3f38l9), 21st &
Pennsylvania Exploratory [<u>Graphs</u>](#wqnsqjgzaaoh)

<u>Choosing factors (broad areas and specific factors)</u>

In order to study the effect of weather, we chose to study specific
weather conditions and their influence on bike demand. The weather data
consisted of several different factors including temperature, humidity
and precipitation, cloud and sunlight, and wind. We decided to select
only the variables that would be likely to affect bike demand.
Additionally, we also selected variables that would best capture overall
conditions related to the above factors since many of the specific
variables were highly correlated with each other and would not offer
significant unique predictive power and could contribute to overfitting
in the model.

Another challenge posed to predicting demand involved the time
dimension. While the day of the week was an important factor in
predicting demand, we wanted to investigate bike demand at a more
granular level to assess if bike rebalancing considerations should be
made at shorter intervals, since demand could fluctuate significantly
during a 24 hour period. We converted and isolated the datetime variable
to display just the hour of the day so we could use each hour of the day
as a predictor in our model.

The final set of factors we considered were from the crash data. Given
that we wanted to investigate the effects of accidents on traffic
conditions, we used only the count of accidents in Ward 2 of DC since
that is where the stations of interest are located.

\*data dictionary [<u>table</u>](#y3pew16ld4n0)

<u>Predictive Modeling</u>

We created 4 prediction models for 4 categories: predicting pick up and
drop off at 21st and I street, and predicting pick up and drop off at
21st and Pennsylvania. We tested the following models for each: Linear
regression, Polynomial Regression, K Nearest Neighbors, LASSO, Ridge
Regression and Elastic Net. We felt it was necessary to separate the
stations because each location has different capacity and potentially
different demand. Additionally, modifications made to one station may
not necessarily be made to the other.

We considered several options in generating the models: initially we did
not account for hourly data, but then considered hour of the day as a
factor level variable. We also considered a 12 month dataset to consider
all seasons, but ultimately used just four months from January to April.
Each model had a few iterations as we removed some of the weather
variables which were insignificant.

<u>Model Performance Evaluation and Prediction</u>

In order to evaluate each model, we employed mean square error (MSE) to
assess how well each model performed. Additionally, we looked at graphs
of actual vs predicted values to evaluate if we had overfitting or
underfitting. In all of our models we split the training and testing
data 60-40 to reserve 40% of the data for test data to see how well the
model performed on a new dataset.

For 21st Street and I Street the best pickup model we created was the
polynomial model of order 2. However, for ease of calculation we ended
up choosing the second lowest MSE, the elastic net, where the MSE was
.002 higher. For the 21st Street and I Street drop off model, the model
with the minimum MSE was the LASSO Regression and regularization model.

For 21st Street and Pennsylvania Street, the best pickup model we
created was the LASSO Regression and regularization model, selected from
its minimum MSE. For the drop off models, the best model for 21st Street
and Pennsylvania Street was the elastic net model, also selected from
its minimum MSE.

Our objective was to predict the need for bikes on the most popular
days, so Capital Bikeshare would be able to maximize profits by meeting
demand for high-use days. Based on exploratory analyses we chose to make
predictions of the days of the week that had the highest pick-ups and
drop-offs (Wednesdays for Penn Station and Tuesdays for I Street), and
use the months with the highest demand (Spring months). We used the
average spring weather conditions for April and the mean number of
accidents that occurred in Ward 2. Using these variables, we made hourly
predictions for pick-ups and drop-offs at each of the stations, and then
summed the predictions per hour for all 24 hours to obtain daily demand.

**MSE VALUES**

| **21st and I St Test MSE Table**                                        | **21st and Pennsylvania Test MSE Table**                                 |
|-------------------------------------------------------------------------|--------------------------------------------------------------------------|
| <img src="media/image43.png" style="width:2.83854in;height:2.1499in" /> | <img src="media/image19.png" style="width:2.72917in;height:2.12854in" /> |

> 21st & I Model Evaluation [<u>Graphs</u>](#8ddlm1qpikb9) 21st & Penn
> Model Evaluation [<u>Graphs</u>](#ow0fz9785ocd)

<u>Findings and Recommendations</u>

Our findings for demand were as follows:

> \- Pick-ups on I street: 274 (11.41667 / hour ≈ 12 / hour)
>
> \- Drop-offs on I street: 380 (15.83333 / hour ≈ 16 / hour)
>
> \- Difference for I street: 106 (4.41667 / hour ≈ 5 / hour)
>
> \- Pick-ups at Penn Station: 41 (1.70833 / hour ≈ 2 / hour)
>
> \- Drop-offs at Penn Station: 181 (7.54167 / hour ≈ 8 / hour)
>
> \- Difference for Penn Station: 140 (5.83333 / hour ≈ 6 / hour)

\*Hourly demand rounded up to reflect maximum predicted demand

We have several solutions that Capital Bikeshare could implement. If
their goal is to meet the maximum predicted demand without losing
customers due to lack of bikes without investing in additional
infrastructure, an hourly rebalancing strategy could be employed for
these stations. However, this may not be feasible as rebalancing is a
costly and labor intensive task. If they find this to be the case,
rebalancing could be done fewer times per day, and performed leading up
to periods of high demand. Rebalancing efforts could also be focused on
stations that have higher overall pick-ups and drop-offs.

Alternatively, Capital Bikeshare could invest in increasing the number
of docks so more bikes are available for pick-up, and more spaces are
available for drop off. Although this does require an initial
investment, this strategy might be more effective as it reduces
recurring costs like the transportation costs and manpower required to
rebalance bikes which might be more efficient in the long run. This
approach would also be helpful for these stations since there is a
significant difference in the number of daily drop offs and pick-ups at
these stations (this difference is most apparent at Penn station). The
difference for drop off and pickups based on our average values are
about 5 or 6. Capital Bikeshare could consider adding 5 to 6 additional
docks and bikes to meet the hourly difference that is typical for a high
demand day.

<u>Assumptions and Limitations</u>

One of the limitations of our model is the time frame of data that we
chose to analyze. Selecting just some winter months and spring months
might not fully capture the effects of seasonality and demand for
particular time frames. For example, summer months might be met with an
increase in tourists, or decrease in students or have increased
popularity due to better weather conditions. We considered both 12 month
models and limited month models, but we found that the 12 month models
had higher MSE values, therefore we selected the four month model. We
made this assumption with the knowledge that DC weather conditions are
fairly temperate, and January-February should capture the winter months
sufficiently while March-April can effectively capture moderate to warm
temperatures that might be present in the spring or fall.

With respect to the crash data, we also have a challenge of interpreting
the count of accidents. While we can assess the frequency of crashes in
Ward 2, we have no actual knowledge of if this interfered with user
routes or if the crash occurred in the same area as the station at the
same time of a particular drop off or pickup. If we were able to, using
actual traffic data for cross streets and nearby areas would be more
useful and accurate in predicting bike usage.

Another limitation of our data collection and wrangling methods is
accounting for the Date Time variable. Initially, we considered building
a model with just pickups and dropoffs per day, however, we felt that
this generalization would be an unreliable predictor as pickups and
dropoffs vary hour to hour and depend on each other. For this reason, we
created the hours variable which at least divides pickups and drop offs
into 24 hour segments. Again, it is possible that this incrementation is
not the best way to further divide the data, but at least helps us
capture the effect that the time of day has on pickups and dropoffs.

Further, while we take into account pickups and dropoffs for a given
hour for particular members or casual users, it is challenging to
calculate the hourly turnover rate or usage rate for a particular dock
or station. This is because we are unaware of how many bikes were
already in the docks before certain bikes were dropped off or picked up.
Thus, subtracting the pickups from the dock’s capacity does not
necessarily indicate available docks or bikes. While we might be able to
extrapolate this information after the first entry in our dataset, we
still would fail to capture the bikes left over from the day prior to
the first day. Because of this, our analysis is limited and we can only
look at pickups and dropoffs for a particular location. This is also
another limitation of our research as we assume that demand can be
determined by pickups and dropoffs, but actual demand could be different
from these figures. This is a key point because demand for pickups
depends entirely on a bike’s availability. On the contrary, demand for
open docks or drop offs depends on how many bikes were picked up or are
present at the dock. Unfortunately we cannot obtain true consumer demand
data so we are only able to use these metrics to assess demand.

Ideally supply would meet demand perfectly, but this may or may not be
feasible. Rebalancing bikes at specific locations or adding additional
docks may not be profitable for Capital Bikeshare. It can be costly to
move bikes physically from one location to another. Structural changes
are also costly and might not pay off. Additionally, even if it was
simple and easy to rebalance bikes, there also may not be enough bikes
available in the overall vicinity, which would be an additional cost to
add more bikes or would cause a lack of availability for consumers.

<u>Appendix</u>

**21st Street and I Street Exploratory Analysis Graphs (return to
[<u>section</u>](#rkxtjpam4u1z)):**

**Number of Pickups (blue) & Drop Offs (orange) over time: 21st & I**

<img src="media/image36.png" style="width:6.5in;height:2.83333in" />

**Usage by Season: 21st & I**

<img src="media/image38.png" style="width:3.625in;height:3.65625in" />

**Pickups & Drop Offs by Day of Week: 21st & I**

<img src="media/image24.png" style="width:3.13021in;height:2.95768in" /><img src="media/image9.png" style="width:3.10938in;height:2.95645in" />

**Pickups & Drop Offs by Membership Status: 21st & I**

<img src="media/image31.png" style="width:3.15618in;height:2.88962in" /><img src="media/image11.png" style="width:3.16014in;height:2.91326in" />

**Pickups (blue) & Drop Off (orange) Count by Start or End Hour: 21st &
I**

<img src="media/image28.png" style="width:4.90625in;height:3.75in" />

**Pickup Histograms: 21st & I**

<img src="media/image37.png" style="width:4.11979in;height:4.01931in" />

**21st Street and I Street Regression Graphs (return to
[<u>section</u>](#dfugcuj98vfi))**

**Actual vs. Predicted Values: Linear Regression 21st & I Pickup Model**

<img src="media/image21.png" style="width:3.1875in;height:2.34375in" />

<img src="media/image42.png" style="width:6.15625in;height:4.5625in" />

**MSE Values of Different Order K: Polynomial Model 21st & I Pickup**

<img src="media/image29.png" style="width:4.87016in;height:3.59755in" />

**MSE Values of Different K: KNN Model 21st & I Pickup**

<img src="media/image14.png" style="width:4.73438in;height:3.58739in" />

**LASSO: 21st & I Pickup Model**

<img src="media/image20.png" style="width:3.05729in;height:2.12919in" /><img src="media/image10.png" style="width:3.1075in;height:2.13922in" />

**Ridge Regression: 21st & I Pickup Model**

<img src="media/image41.png" style="width:3.53125in;height:4.9375in" />

**Actual vs. Predicted Values: Linear Regression 21st & I Drop Off
Model**

<img src="media/image22.png" style="width:6.5in;height:2.95833in" />

**Train & Test MSE of Polynomial Model of Order K: 21st & I Drop Off
Model**

<img src="media/image25.png" style="width:6.25in;height:4.5625in" />

**Train & Test MSE of Different K: KNN 21st & I Drop Off Model**

<img src="media/image17.png" style="width:6.28125in;height:4.71875in" />

**LASSO: 21st & I Drop Off Model**

<img src="media/image26.png" style="width:3.20313in;height:2.23337in" /><img src="media/image5.png" style="width:3.02604in;height:2.172in" />

**Ridge Regression: 21st & I Drop Off Model**

<img src="media/image34.png" style="width:3.27339in;height:4.42047in" />

**21st Street and Pennsylvania Exploratory Analysis Graphs (return to
[<u>section</u>](#rkxtjpam4u1z)):**

**Pickups (blue) & Drop Offs (orange) for 21st & Pennsylvania Over
Time**

<img src="media/image30.png" style="width:4.26563in;height:3.08073in" />

**Usage by Season: 21st & Pennsylvania**

<img src="media/image2.png" style="width:4.03125in;height:3.65625in" />

**Pickups & Drop Offs by Day of Week: 21st & Pennsylvania**

<img src="media/image33.png" style="width:3.13157in;height:2.95172in" /><img src="media/image6.png" style="width:3.11979in;height:2.99063in" />

**Pickups & Drop Offs by Membership Status: 21st & Pennsylvania**

<img src="media/image3.png" style="width:3.09646in;height:2.77503in" /><img src="media/image7.png" style="width:3.02604in;height:2.86277in" />

**Pickups (blue) and Drop Offs (orange) by Start or End Hour: 21st &
Pennsylvania**

<img src="media/image32.png" style="width:5in;height:3.65625in" />

**Pickup Histograms: 21st & Pennsylvania**

<img src="media/image16.png" style="width:4.39063in;height:4.20548in" />

**21st Street and Pennsylvania Regression Graphs (return to
[<u>section</u>](#dfugcuj98vfi))**

**Actual vs. Predicted Values: Linear Regression Pick Up Model 21st &
Pennsylvania**

<img src="media/image35.png" style="width:4.84896in;height:3.65551in" />

**Train & Test MSE for Order K: Polynomial Model 21st & Pennsylvania
Pickups**

<img src="media/image18.png" style="width:5.49479in;height:3.95948in" />

**Train & Test MSE for Different K: KNN Model 21st & Pennsylvania
Pickups**

<img src="media/image12.png" style="width:5.07292in;height:3.77282in" />

**LASSO: 21st & Pennsylvania Pickups**

<img src="media/image15.png" style="width:3.30729in;height:2.16584in" /><img src="media/image40.png" style="width:3.00521in;height:2.09518in" />

**Ridge Regression: 21st & Pennsylvania Pickups**

<img src="media/image23.png" style="width:2.95313in;height:4.00781in" />

**Actual vs Predicted Values: Linear Regression 21st & Pennsylvania Drop
Offs**

<img src="media/image27.png" style="width:6.5in;height:3.05556in" />

**Train & Test MSE for Order K: Polynomial Model 21st & Pennsylvania
Drop Offs**

<img src="media/image39.png" style="width:5.34896in;height:3.80599in" />

**Train & Test MSE for Different K: KNN Model 21st & Pennsylvania Drop
Offs**

<img src="media/image4.png" style="width:4.84375in;height:3.62673in" />

**LASSO: 21st & Pennsylvania Drop Offs**

<img src="media/image13.png" style="width:3.22396in;height:2.10258in" /><img src="media/image8.png" style="width:3.16146in;height:2.14631in" />

**Ridge Regression: 21st & Pennsylvania Drop Offs**

<img src="media/image1.png" style="width:3.65625in;height:4.59375in" />

**Data Dictionary**

| **Attribute Name** | **Data Type** | **Description**                                                                            |
|--------------------|---------------|--------------------------------------------------------------------------------------------|
| seasons            | object        | Season of year: Spring, Summer, Fall, Winter                                               |
| day_of_week        | object        | Day of week: Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday        |
| holiday            | bool          | If the date is a holiday or not: True or False (False if not a holiday, True if a holiday) |
| member_casual      | object        | Is member for members of capital bikeshare or casual for casual users of bikeshare         |
| start_hour         | object        | Hour of day 0-23 (non continuous variable treated as a factor)                             |
| pu_ct              | int64         | The count of bikes picked up for each station                                              |
| No. Docks          | int64         | The dock capacity for a given location, 9 for 21st & Penn and 32 for 21st & I              |
| Temp               | float64       | The temperature for a given day                                                            |
| feelslike          | float64       | What the temperature feels like outside for a given day                                    |
| humidity           | float64       | What the humidity is for a given day                                                       |
| precip             | float64       | The expected or actual amount of precipitation in inches.                                  |
| preciptype         | object        | The type of precipitation, ex. Rain, snow, hail                                            |
| snow               | float64       | The amount of snow expected or actual in inches.                                           |
| windspeed          | float64       | The wind speed for a given day in miles per hour                                           |
| cloudcover         | float64       | The percentage of cloud cover for a given day                                              |
| visibility         | float64       | The visibility (in miles) for a given day                                                  |
| uvindex            | int64         | The UV index on an integer scale                                                           |
| conditions         | object        | Short text about the weather and a description                                             |
| icon               | object        | A weather icon used                                                                        |
| ward2_accidents    | int64         | Count of accidents in Ward 2 for a given day                                               |
