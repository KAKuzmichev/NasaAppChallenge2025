import lightkurve as lk

# 1. Завантаження файлу FITS
file_path = 'data\Kepler_confirmed_wget\kplr010000941-2009166043257_llc.fits'

# 2. Використання lightkurve для читання кривої блиску
lc = lk.read(file_path)

# 3. Отримання даних
time = lc.time.value    # Час спостереження
flux = lc.flux.value    # Світловий потік (яскравість)

print(lc.keys())
print(len(lc))


# 4. Виведення перших кількох точок даних для перевірки
print("Перші 5 точок часу:", time[:5])
print("Перші 5 значень потоку:", flux[:5])

# lightkurve також дозволяє легко візуалізувати дані:
lc.plot()

"""
import pandas as pd

# 1. Завантаження TBL-файлу. TBL-файли часто використовують пробіли як роздільник.
# Можливо, вам доведеться змінити 'sep' на інший символ (наприклад, ',' або ';')
file_path = 'data\\Kepler_confirmed_wget\\kplr010000941-2009166043257_llc_lc.tbl'

try:
    # Спроба читання як TSV (Tab-Separated Values) або space-separated
    df = pd.read_csv(file_path, sep='\s+')  # '\s+' означає один або більше пробілів

    # 2. Виведення заголовків стовпців для перевірки
    print("Доступні стовпці:", df.columns)

    # 3. Отримання даних (назви стовпців можуть відрізнятися, наприклад, 'TIME' та 'SAP_FLUX')
    time_data = df['TIME']
    flux_data = df['SAP_FLUX']

except Exception as e:
    print(f"Помилка читання файлу .tbl. Спробуйте інший роздільник або використовуйте FITS. Помилка: {e}")"""