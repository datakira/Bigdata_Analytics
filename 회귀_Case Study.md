## 자전거 대여 수요 예측
- 데이터 불러오기 : bike_df=pd.read_csv('./dataset/bike-sharing-demand/bike_train.csv')
- 문자열 datetime을 날짜 타입 new_Date 칼럼으로 변경(new_Date 생성 및 datetime 삭제) 
- new_Date 칼럼에서 년, 월, 일, 시간 칼럼 생성 
- new_Date 칼럼, 중복 발생하는 사용자 대여 횟수('casual','registered') 칼럼 삭제
- 회귀 예측
 - 독립변수, 종속변수 분리
 - 훈련데이터, 검증데이터 7:3으로 분리
- 평가(RMSE, MAE, Variance score)
 - from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#### Data Description
You are provided hourly rental data spanning two years. For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. You must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

#### Data Fields
* datetime - hourly date + timestamp  
* season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
* holiday - whether the day is considered a holiday
* workingday - whether the day is neither a weekend nor holiday
* weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
* 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
* 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
* 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
* temp - temperature in Celsius
* atemp - "feels like" temperature in Celsius
* humidity - relative humidity
* windspeed - wind speed
* casual - number of non-registered user rentals initiated
* registered - number of registered user rentals initiated
* count - number of total rentals
