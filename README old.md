Застосунок призначений для аналізу та прогнозування тенденцій ринку відеокарт. Реалізований на Python із використанням Streamlit, Pandas, Plotly та Statsmodels.
Додаток дозволяє переглядати історичні ціни GPU, аналізувати характеристики сучасних відеокарт, будувати прогнози та досліджувати ринкові частки виробників.

Структура проєкту
app.py
requirements.txt
/data
    gpu_price_history.csv
    gpu_specs_prices.csv
    gpu_shipments_market_share.csv
    GPU_Price_Index.csv

Установка
1. Переконайтесь, що встановлено Python 3.9–3.13
2. Встановіть залежності:

pip install -r requirements.txt

3. Запуск програми
У директорії проєкту виконайте команду:

streamlit run app.py

Якщо з’являється запит Email — просто натисніть Enter

4.Далі відкрийте сторінку:

http://localhost:8501

Зачекайте (поки проєкт збереться до купи!)

5. Використання
У застосунку доступні вкладки:
Market overview — сучасні GPU та ціни
Price dynamics & forecast — історія та прогнозування
Shipments & market share — частки ринку
US price index — індекс цін
Data — перегляд вихідних таблиць
