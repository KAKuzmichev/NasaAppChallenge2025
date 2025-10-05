import pandas as pd
import glob
from astropy.table import Table
import os
import warnings
from astropy.units import UnitsWarning # Для придушення попереджень про одиниці
import itertools # Допоможе розділити список файлів на порції

# --- Налаштування ---
FITS_DIR = 'data/Kepler_Quarterly_wget'
FITS_PATTERN = os.path.join(FITS_DIR, '*.fits')
OUTPUT_CSV_FILE = 'Kepler_Quarterly_wget.csv'
CHUNK_SIZE = 1000 # Встановлюємо розмір порції (наприклад, 1000 файлів)

# Придушуємо попередження astropy UnitsWarning, щоб не засмічувати вивід
warnings.simplefilter('ignore', category=UnitsWarning)

# 1. Знаходимо всі FITS-файли
fits_files = glob.glob(FITS_PATTERN)

if not fits_files:
    print(f"Помилка: Файлів FITS не знайдено у {FITS_DIR}")
else:
    total_files = len(fits_files)
    print(f"Знайдено {total_files} FITS-файлів. Розмір порції: {CHUNK_SIZE}.")
    
    # Визначаємо, чи потрібно писати заголовок CSV (тільки для першої порції)
    write_header = True
    
    # Функція для створення порцій (chunks)
    # Згруповує файли у списки по CHUNK_SIZE елементів
    def chunk_list(iterable, size):
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, size))
            if not chunk:
                return
            yield chunk

    # 2. Обробляємо файли порціями
    for i, file_chunk in enumerate(chunk_list(fits_files, CHUNK_SIZE)):
        print(f"\n---> Обробка порції №{i+1} ({len(file_chunk)} файлів)...")
        
        chunk_dataframes = []
        
        # Читаємо файли в поточній порції
        for file_path in file_chunk:
            try:
                # Читаємо FITS-файл як astropy Table
                astropy_table = Table.read(file_path, format='fits')
                
                # Конвертуємо Table у pandas DataFrame
                df = astropy_table.to_pandas()
                
                # Додаємо стовпець з іменем вихідного файлу
                df['Original_FITS_File'] = os.path.basename(file_path)
                
                chunk_dataframes.append(df)
            except Exception as e:
                print(f"Помилка при читанні файлу {os.path.basename(file_path)}. Пропущено. Помилка: {e}")

        if chunk_dataframes:
            # 3. Об'єднуємо DataFrames у межах поточної порції
            chunk_dataframe_merged = pd.concat(chunk_dataframes, ignore_index=True)
            
            # 4. Записуємо/додаємо дані у вихідний CSV-файл
            # 'w' (write): перезаписує файл (використовується для першої порції)
            # 'a' (append): додає дані у кінець файлу (використовується для наступних порцій)
            mode = 'w' if write_header else 'a'
            header = write_header
            
            chunk_dataframe_merged.to_csv(
                OUTPUT_CSV_FILE, 
                mode=mode, 
                header=header, 
                index=False
            )
            
            # Встановлюємо write_header на False, щоб не писати заголовок знову
            write_header = False
            
            # Очищуємо пам'ять, видаляючи DataFrames поточної порції
            del chunk_dataframes
            del chunk_dataframe_merged
            
            print(f"Порція №{i+1} успішно записана у {OUTPUT_CSV_FILE}.")
            
    print("\n-------------------------------------------")
    print(f"✅ Готово! Усі файли FITS об'єднано у {OUTPUT_CSV_FILE}.")
    print("-------------------------------------------")