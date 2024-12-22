from dataclasses import dataclass

import httpx
from enum import Enum

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import json

WINDOW_SIZE = 30
BASE_URL = 'https://api.openweathermap.org/'

@dataclass
class CityWeatherData:
  city: str
  mean_temp: float
  min_temp: float
  max_temp: float
  season_profile: pd.DataFrame
  slope: np.ndarray
  outliers: pd.DataFrame

class SeasonEnum(str, Enum):
  autumn = 'autumn'
  spring = 'spring'
  summer = 'summer'
  winter = 'winter'

def detect_outliers(group):
  rolling = group['temperature'].rolling(window=WINDOW_SIZE, center=True, min_periods=1)
  rolling_mean = rolling.mean()
  rolling_std = rolling.std()

  std_coeff = 2 * rolling_std
  threshold1 = rolling_mean + std_coeff
  threshold2 = rolling_mean - std_coeff

  return group[(group['temperature'] > threshold1) | (group['temperature'] < threshold2)]

def analyze_city(df: pd.DataFrame, city: str) -> CityWeatherData:
  df_city = df[df['city'] == city]

  grouped_by_season = df_city.groupby('season')['temperature']
  outliers = df_city.groupby('season')[[*df.columns.values]].apply(detect_outliers)

  season_profile = grouped_by_season.agg(mean='mean', std='std').reset_index()

  timestamp_ordinal = df_city['timestamp'].apply(lambda x: pd.to_datetime(x).toordinal())
  lag = df_city['temperature'].shift(1, fill_value=0)

  df_ts = pd.DataFrame({'timestamp_ordinal': timestamp_ordinal, 'lag': lag, 'temperature': df_city['temperature']})

  X = df_ts[['timestamp_ordinal', 'lag']]
  y = df_ts['temperature']

  model = LinearRegression()
  model.fit(X, y)

  pred = pd.Series(model.predict(X))

  mean_temp = df_city['temperature'].mean()
  min_temp = df_city['temperature'].min()
  max_temp = df_city['temperature'].max()

  return CityWeatherData(city, mean_temp, min_temp, max_temp, season_profile, model.coef_, outliers)

def get_temperature_by_city(city_name: str, api_key: str,  state_code: str = '', country_code: str = '', limit: int = 1) -> float:
  try:
    client = httpx.Client(base_url=BASE_URL)

    params = {'q': f'{city_name},{state_code},{country_code}', 'limit': limit, 'appid': api_key}
    coords_json = client.get('/geo/1.0/direct', params=params).json()

    lat = coords_json[0]['lat']
    lon = coords_json[0]['lon']

    params = {'lat': lat, 'lon': lon, 'units': 'metric', 'appid': api_key}
    weather_json = client.get('/data/2.5/weather', params=params).json()

    return weather_json['main']['temp']

  except (BaseException, Exception):
    raise httpx.HTTPError('{"code": 401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}')

def is_temperature_anomal(city: str, temperature: float, season: str, city_weather_data: CityWeatherData) -> bool:
  season = city_weather_data.season_profile[city_weather_data.season_profile['season'] == season]
  mean = float(season['mean'].iloc[0])
  std = float(season['std'].iloc[0])

  std_coeff = 2 * std
  threshold1 = mean + std_coeff
  threshold2 = mean - std_coeff

  return (temperature > threshold1) | (temperature < threshold2)