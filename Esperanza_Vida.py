import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Desactivar las advertencias
warnings.filterwarnings("ignore")

# Título de la aplicación
st.title("ESPERANZA DE VIDA AL NACER")

st.markdown("<br>", unsafe_allow_html=True)  # Agregar espacio

# Subtítulo - Series de tiempo
st.subheader("I. SERIES DE TIEMPO")

# Subtítulo - Series de tiempo
st.subheader("  I.1. Análisis por País:")

# Cargar los datos históricos
data = pd.read_csv('data.csv')
data['Time'] = pd.to_datetime(data['Time'], format='%Y')
data = data.set_index('Time')

# Obtener la lista de años disponibles en orden descendente
available_years = data.index.year.unique()[::-1]

# Filtrar los datos hasta el año seleccionado
selected_year = st.selectbox("Selecciona el año límite:", available_years, key="year")

# Filtrar los datos solo para el año seleccionado
data_filtered = data[data.index.year <= selected_year]

# Lista de países de interés
paises_interes = data_filtered['Country Name'].unique()

# Selección del país
selected_pais = st.selectbox("Selecciona un país:", paises_interes, key="country")

# Filtrar los datos solo para el país seleccionado
datos_pais = data_filtered[data_filtered['Country Name'] == selected_pais]
dato = datos_pais[['Life expectancy at birth, total (years)']]

# Mostrar los datos históricos
show_data = st.checkbox("Mostrar datos históricos")
if show_data:
    st.subheader("Datos Históricos")
    st.dataframe(dato)

# Definir la variable forecast_steps fuera de la sección de análisis de series de tiempo
forecast_steps = st.number_input("Número de años para el pronóstico:", min_value=1, max_value=10, value=3, key="forecast_steps")

# Análisis de series de tiempo
if st.button("Realizar Análisis de Series de Tiempo"):
    # Definir los posibles valores de los parámetros
    p_values = range(0, 3)  # Valores posibles para p
    d_values = range(0, 2)  # Valores posibles para d
    q_values = range(0, 3)  # Valores posibles para q

    # Realizar la búsqueda en cuadrícula para encontrar los mejores parámetros
    best_aic = float('inf')  # Inicializar el mejor valor de AIC como infinito
    best_params = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(dato, order=(p, d, q))
                    model_fit = model.fit()

                    # Calcular el criterio de información de Akaike (AIC)
                    aic = model_fit.aic

                    # Actualizar los mejores parámetros si se encuentra un valor de AIC menor
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                except:
                    continue

    # Ajustar el modelo ARIMA con los mejores parámetros encontrados
    model = ARIMA(dato, order=best_params)
    model_fit = model.fit()

    # Generar pronósticos para los años siguientes
    #forecast_steps = forecast_steps = st.number_input("Número de años para el pronóstico:", min_value=1, max_value=10, value=3, key="forecast_steps")
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_values = forecast.predicted_mean

    # Visualizar los datos históricos y los pronósticos
    fig, ax = plt.subplots(figsize=(15, 8))  # Establecer el tamaño del gráfico
    ax.plot(dato.index, dato.values, label='Datos Históricos')
    ax.plot(forecast_values.index, forecast_values.values, label='Pronóstico')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Esperanza de Vida al Nacer (años)')
    ax.set_title(f"Esperanza de Vida al Nacer - {selected_pais}")  # Agregar título con el nombre del país
    ax.legend()

    # Agregar marcadores sobre la línea de la serie
    ax.scatter(dato.index, dato.values, color='green', zorder=5)

    # Agregar marcadores para los años pronosticados
    next_years = pd.date_range(start=dato.index[-1], periods=forecast_steps, freq='A')
    ax.scatter(next_years, forecast_values, color='orange', zorder=5)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    # Imprimir los mejores parámetros encontrados
    st.subheader("Mejores Parámetros (p, d, q)")
    st.write(best_params)

    # Imprimir los pronósticos para los años siguientes
    st.subheader("Pronósticos para los siguientes años:")
    for year, value in zip(next_years + pd.DateOffset(years=1), forecast_values):
        st.write(year.year, ":", round(value, 2))
st.markdown("<br>", unsafe_allow_html=True)  # Agregar espacio

#########
######### I.2. Crecimiento por Región:
import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Desactivar las advertencias
warnings.filterwarnings("ignore")

# Paso 1: Cargar los datos históricos
data = pd.read_csv('data.csv')
data['Time'] = pd.to_datetime(data['Time'], format='%Y')
data = data.set_index('Time')

# Subtítulo y selección de Región
st.subheader("  I.2. Crecimiento por Región:")

# Filtrar los datos hasta el año seleccionado
selected_year = st.selectbox("Selecciona el año límite:", data.index.year.unique().sort_values(ascending=False), key="year_selection")

# Filtrar los datos solo para el año seleccionado
data_filtered = data[data.index.year <= selected_year]

# Seleccionar las regiones deseadas como una lista
regiones = ['North America', 'Central America', 'South America', 'Oceania']

# Definir los países correspondientes a cada región seleccionada
paises_region = {
    'North America': ['Canada', 'United States', 'Mexico'],
    'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana',
                      'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela, RB'],
    'Central America': ['Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Nicaragua', 'Panama'],
    'Oceania': ['Australia', 'Fiji', 'Marshall Islands', 'Solomon Islands', 'Kiribati', 'Micronesia, Fed. Sts.',
                'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Tonga', 'Tuvalu', 'Vanuatu']
}

# Selección de Región
selected_regions = st.multiselect("Seleccionar regiones", regiones)

# Iterar sobre las regiones seleccionadas
for region in selected_regions:
    # Verificar si la región seleccionada existe en el diccionario de regiones
    if region not in paises_region:
        st.write(f"La región '{region}' no es válida.")
        continue

    # Obtener los países correspondientes a la región seleccionada
    paises_seleccionados = paises_region[region]

    # Paso 2: Definir los posibles valores de los parámetros
    p_values = range(3)  # Valores posibles para p
    d_values = range(2)  # Valores posibles para d
    q_values = range(3)  # Valores posibles para q

    # Paso 2-5: Iterar sobre cada país y realizar el análisis de series de tiempo
    crecimiento_paises = {}

    for pais, dato in data_filtered.groupby('Country Name')['Life expectancy at birth, total (years)']:
        # Verificar si el país pertenece a la región seleccionada
        if pais not in paises_seleccionados:
            continue

        # Paso 3: Realizar la búsqueda en cuadrícula para encontrar los mejores parámetros
        best_aic = float('inf')  # Inicializar el mejor valor de AIC como infinito
        best_params = None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(dato, order=(p, d, q))
                        model_fit = model.fit()

                        # Calcular el criterio de información de Akaike (AIC)
                        aic = model_fit.aic

                        # Actualizar los mejores parámetros si se encuentra un valor de AIC menor
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except:
                        continue

        # Paso 4: Ajustar el modelo ARIMA con los mejores parámetros encontrados
        model = ARIMA(dato, order=best_params)
        model_fit = model.fit()

        # Paso 5: Generar pronósticos para los años siguientes
        forecast_steps = 3
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean

        # Verificar si el crecimiento ha sido negativo en cada año del pronóstico
        crecimiento_positivo = all(forecast_values[i] > forecast_values[i - 1] for i in range(1, forecast_steps))

        # Calcular el decrecimiento esperado para los próximos 3 años si el crecimiento ha sido negativo en cada año
        if crecimiento_positivo:
            crecimiento = (forecast_values[-1] - dato.values[-1]) / dato.values[-1] * 100
            if pd.isnull(crecimiento):
                continue  # Ignorar este país si el decrecimiento es NaN

            crecimiento_paises[pais] = crecimiento

    # Paso 6: Ordenar los países según el crecimiento esperado
    paises_ordenados = sorted(crecimiento_paises, key=crecimiento_paises.get, reverse=True)

    # Mostrar los países con el mayor decrecimiento esperado y su expectativa de vida al nacer
    st.write(f"Región: {region}")
    for pais in paises_ordenados[:3]:
        crecimiento = crecimiento_paises[pais]
        st.write(f"{pais} ==> Crecimiento: {round(float(crecimiento), 2)}%")
    st.write()
st.markdown("<br>", unsafe_allow_html=True)  # Agregar espacio

#########
######### I.3. Decrecimiento por Región:
import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Desactivar las advertencias
warnings.filterwarnings("ignore")

# Paso 1: Cargar los datos históricos
data = pd.read_csv('data.csv')
data['Time'] = pd.to_datetime(data['Time'], format='%Y')
data = data.set_index('Time')

# Subtítulo y selección de Región
st.subheader("  I.3. Decrecimiento por Región:")

# Filtrar los datos hasta el año seleccionado
selected_year = st.selectbox("Selecciona el año límite:", data.index.year.unique().sort_values(ascending=False), key="year_selection2")

# Filtrar los datos solo para el año seleccionado
data_filtered = data[data.index.year <= selected_year]

# Seleccionar las regiones deseadas como una lista
regiones = ['North America', 'Central America', 'South America', 'Oceania']

# Definir los países correspondientes a cada región seleccionada
paises_region = {
    'North America': ['Canada', 'United States', 'Mexico'],
    'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana',
                      'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela, RB'],
    'Central America': ['Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Nicaragua', 'Panama'],
    'Oceania': ['Australia', 'Fiji', 'Marshall Islands', 'Solomon Islands', 'Kiribati', 'Micronesia, Fed. Sts.',
                'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Tonga', 'Tuvalu', 'Vanuatu']
}

# Selección de Región
selected_regions = st.multiselect("Seleccionar regiones", regiones, key="region_selection2")


# Iterar sobre las regiones seleccionadas
for region in selected_regions:
    # Verificar si la región seleccionada existe en el diccionario de regiones
    if region not in paises_region:
        st.write(f"La región '{region}' no es válida.")
        continue

    # Obtener los países correspondientes a la región seleccionada
    paises_seleccionados = paises_region[region]

    # Paso 2: Definir los posibles valores de los parámetros
    p_values = range(3)  # Valores posibles para p
    d_values = range(2)  # Valores posibles para d
    q_values = range(3)  # Valores posibles para q

    # Paso 2-5: Iterar sobre cada país y realizar el análisis de series de tiempo
    decrecimiento_paises = {}

    for pais, dato in data_filtered.groupby('Country Name')['Life expectancy at birth, total (years)']:
        # Verificar si el país pertenece a la región seleccionada
        if pais not in paises_seleccionados:
            continue

        # Paso 3: Realizar la búsqueda en cuadrícula para encontrar los mejores parámetros
        best_aic = float('inf')  # Inicializar el mejor valor de AIC como infinito
        best_params = None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(dato, order=(p, d, q))
                        model_fit = model.fit()

                        # Calcular el criterio de información de Akaike (AIC)
                        aic = model_fit.aic

                        # Actualizar los mejores parámetros si se encuentra un valor de AIC menor
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except:
                        continue

        # Paso 4: Ajustar el modelo ARIMA con los mejores parámetros encontrados
        model = ARIMA(dato, order=best_params)
        model_fit = model.fit()

        # Paso 5: Generar pronósticos para los años siguientes
        forecast_steps = 3
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_values = forecast.predicted_mean

        # Verificar si el crecimiento ha sido negativo en cada año del pronóstico
        crecimiento_negativo = all(forecast_values[i] < forecast_values[i - 1] for i in range(1, forecast_steps))

        # Calcular el decrecimiento esperado para los próximos 3 años si el crecimiento ha sido negativo en cada año
        if crecimiento_negativo:
            decrecimiento = (forecast_values[-1] - dato.values[-1]) / dato.values[-1] * 100
            if pd.isnull(decrecimiento):
                continue  # Ignorar este país si el decrecimiento es NaN

            decrecimiento_paises[pais] = decrecimiento

    # Paso 6: Ordenar los países según el crecimiento esperado
    paises_ordenados = sorted(decrecimiento_paises, key=decrecimiento_paises.get, reverse=False)

    # Mostrar los países con el mayor decrecimiento esperado y su expectativa de vida al nacer
    st.write(f"Región: {region}")
    for pais in paises_ordenados[:3]:
        decrecimiento = decrecimiento_paises[pais]
        st.write(f"{pais} ==> Decrecimiento: {round(float(decrecimiento), 2)}%")
    st.write()
st.markdown("<br>", unsafe_allow_html=True)  # Agregar espacio

#########
######### II. Regresión Lineal
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import matplotlib.pyplot as plt
import streamlit as st

# Desactivar las advertencias
warnings.filterwarnings("ignore")

# Cargar los datos
data = pd.read_csv('data.csv')

# Subtítulo y selección de Región
st.subheader("II. REGRESION LINEAL")

# Lista de países únicos en los datos
paises = data['Country Name'].unique()

# Seleccionar el país mediante un selectbox en Streamlit
selected_country = st.selectbox("Selecciona un país:", paises)

# Filtrar los datos para el país seleccionado
pais_data = data[data['Country Name'] == selected_country]

# Crear un nuevo DataFrame con las variables deseadas
model_data = pais_data[['Birth rate, crude (per 1,000 people)',
                        'Death rate, crude (per 1,000 people)',
                        'Fertility rate, total (births per woman)',
                        'Mortality rate, infant (per 1,000 live births)',
                        'Population growth (annual %)',
                        'Net migration',
                        'Rural population (% of total population)',
                        'Urban population (% of total population)',
                        'Adolescent fertility rate (births per 1,000 women ages 15-19)',
                        'Life expectancy at birth, total (years)']]

# Imputar los valores faltantes utilizando la media
imputer = SimpleImputer(strategy='mean')
model_data = pd.DataFrame(imputer.fit_transform(model_data), columns=model_data.columns)

# Obtener todas las combinaciones posibles de 2 y 3 variables
variables = ['Birth rate, crude (per 1,000 people)',
            'Death rate, crude (per 1,000 people)',
            'Fertility rate, total (births per woman)',
            'Mortality rate, infant (per 1,000 live births)',
            'Population growth (annual %)',
            'Net migration',
            'Rural population (% of total population)',
            'Urban population (% of total population)',
            'Adolescent fertility rate (births per 1,000 women ages 15-19)']

best_r2 = 0
best_X = None
best_coeficientes = None

scaler = StandardScaler()

for r in range(2, 4):  # Probar con 2 y 3 variables independientes
    for combo in combinations(variables, r):
        # Dividir los datos en características (X) y variable objetivo (y)
        X = model_data[list(combo)]
        y = model_data['Life expectancy at birth, total (years)']

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalar las características
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Crear y entrenar el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Calcular el coeficiente de determinación (R²)
        r2 = model.score(X_test_scaled, y_test)

        # Actualizar el mejor R², las variables seleccionadas y los coeficientes
        if r2 > best_r2:
            best_r2 = r2
            best_X = combo
            best_coeficientes = model.coef_

# Calcular los residuos para el mejor modelo
X_best = model_data[list(best_X)]
y_best = model_data['Life expectancy at birth, total (years)']
X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(X_best, y_best, test_size=0.2, random_state=42)
X_train_scaled_best = scaler.fit_transform(X_train_best)
X_test_scaled_best = scaler.transform(X_test_best)
model_best = LinearRegression()
model_best.fit(X_train_scaled_best, y_train_best)
y_pred_best = model_best.predict(X_test_scaled_best)
residuos_best = y_test_best - y_pred_best

# Crear un DataFrame con las variables y sus relaciones
relaciones_data = pd.DataFrame({'Variable': best_X, 'Relacion': best_coeficientes})
# Ordenar las variables por su relación con Life expectancy at birth
relaciones_data = relaciones_data.sort_values(by='Relacion', ascending=False)

# Graficar la relación entre las variables independientes y la expectativa de vida
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

axs[0].barh(relaciones_data['Variable'], relaciones_data['Relacion'],
            color=['green' if coeficiente > 0 else 'red' for coeficiente in relaciones_data['Relacion']])
axs[0].set_title('Relación: Life Expectancy y Variables Independientes - ' + selected_country)
axs[0].set_xlabel('Coeficiente de Regresión')
axs[0].set_ylabel('Variables Independientes')

for i, coeficiente in enumerate(relaciones_data['Relacion']):
    axs[0].text(coeficiente, i, f'{coeficiente:.2f}', ha='left', va='center')

axs[1].scatter(y_test_best, y_pred_best)
axs[1].plot([min(y_test_best), max(y_test_best)], [min(y_test_best), max(y_test_best)], color='r', linestyle='--')
axs[1].set_xlabel('Valores reales')
axs[1].set_ylabel('Pronósticos')
axs[1].set_title('Valores reales vs. Pronósticos')

# Mostrar el valor de R² en el gráfico
axs[1].text(0.05, 0.95, f'R² = {best_r2:.4f}', transform=axs[1].transAxes, ha='left', va='top')

plt.tight_layout()
st.pyplot(fig)

