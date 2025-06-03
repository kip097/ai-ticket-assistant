import pandas as pd

def load_dataset(file_path='sample_tickets.csv'):
    df = pd.read_csv(file_path)
    print(f"Всего заявок: {len(df)}")
    print(df.head(10))  # Вывод первых 10 строк
    print("\nКатегории и количество заявок в каждой:")
    print(df['category'].value_counts())
    return df

if __name__ == "__main__":
    load_dataset()
