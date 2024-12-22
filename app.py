import matplotlib.pyplot as plt
import streamlit as st

from service import *


def plot_outliers(city: str, df: pd.DataFrame, outliers: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(10, 6))

    df_city = df[df['city'] == city]
    temp = df_city['temperature']
    timestamp = df_city['timestamp']

    plt.plot(timestamp, temp, label=f'График температуры в городе {city}')
    plt.xlabel('Дата')
    plt.ylabel('Температура')

    plt.scatter(outliers['timestamp'], outliers['temperature'], color='red', label='Аномальные значения')

    plt.legend()

    st.pyplot(figure)


st.title('Анализ температурных данных и мониторинг текущей температуры через OpenWeatherMap API')

st.header('Шаг 1: Загрузите данные')

uploaded_file = st.file_uploader('Выберите csv-файл', type='csv')
df = pd.DataFrame()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    st.header('Шаг 2: Выберите город')
    cities = sorted(df['city'].unique().tolist())
    city = st.selectbox(label='Город', options=cities)

    if not df.empty:
        city_weather_data = analyze_city(df, city)

        mean_temp = round(city_weather_data.mean_temp, 2)
        min_temp = round(city_weather_data.min_temp, 2)
        max_temp = round(city_weather_data.max_temp, 2)

        st.subheader('Минимальная температура', divider='blue')
        st.text(str(min_temp) + '\u00b0C')

        st.subheader('Максимальная температура', divider='orange')
        st.text(str(max_temp) + '\u00b0C')

        st.subheader('Средняя температура', divider='green')
        st.text(str(mean_temp) + '\u00b0C')

        season_profile = city_weather_data.season_profile
        st.subheader('Сезонный профиль', divider='rainbow')
        st.dataframe(season_profile)

        slope_coeffs = np.around(city_weather_data.slope, 2)
        st.subheader('Уклон тренда', divider='violet')
        is_positive = all(x > 0 for x in slope_coeffs)
        if is_positive:
            st.success('Положительный')
        else:
            st.error('Отрицательный')

        st.write('**Коэффициенты**')
        st.text(slope_coeffs)

        st.subheader('Временной ряд температур с выделением аномалий')
        plot_outliers(city, df, city_weather_data.outliers)

        st.header('Шаг 3: Введите API-ключ OpenWeatherMap')
        api_key = st.text_input(label='API-ключ', max_chars=32)

        season = st.selectbox(label='Выберите сезон', options=[item.value for item in SeasonEnum])

        if api_key:
            try:
                temperature = get_temperature_by_city(city, api_key)

                st.subheader('Температура сейчас')
                st.text(str(temperature) + '\u00b0C')

                temperature_anomal = is_temperature_anomal(temperature, season, city_weather_data)

                if temperature_anomal:
                    st.error('Температура аномальна для текущего сезона')
                else:
                    st.success('Температура нормальна для текущего сезона')

            except httpx.HTTPError as e:
                st.exception(e)

else:
    st.write('Пожалуйста, загрузите csv-файл')
