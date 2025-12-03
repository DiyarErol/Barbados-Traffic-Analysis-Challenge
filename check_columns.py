import pandas as pd

df = pd.read_csv('traffic_predictions_enhanced.csv')

print('='*60)
print('İSTENEN SÜTUNLAR KONTROLÜ')
print('='*60)

# Sütunların varlığını kontrol et
print('\n✓ Sütunların Varlığı:')
print(f"  - Kimlik: {'✓ Var' if 'Kimlik' in df.columns else '✗ Yok'}")
print(f"  - Hedef: {'✓ Var' if 'Hedef' in df.columns else '✗ Yok'}")
print(f"  - Hedef_Doğruluğu: {'✓ Var' if 'Hedef_Doğruluğu' in df.columns else '✗ Yok'}")

# İlk 10 kayıt
print('\n📋 İlk 10 Kayıt:')
print(df[['Kimlik', 'Hedef', 'Hedef_Doğruluğu']].head(10).to_string())

# İstatistikler
print('\n📊 İstatistikler:')
print(f'  Toplam kayıt: {len(df):,}')
print(f'  Kimlik aralığı: {df["Kimlik"].min()} - {df["Kimlik"].max()}')
print(f'  Benzersiz Hedef sayısı: {df["Hedef"].nunique()}')
print(f'\n  Hedef_Doğruluğu dağılımı:')
print(f'    - Doğru (1): {(df["Hedef_Doğruluğu"] == 1).sum():,} ({(df["Hedef_Doğruluğu"] == 1).mean():.2%})')
print(f'    - Yanlış (0): {(df["Hedef_Doğruluğu"] == 0).sum():,} ({(df["Hedef_Doğruluğu"] == 0).mean():.2%})')

# Hedef değerlerinin dağılımı
print('\n🎯 En Yaygın Hedef Değerleri:')
print(df['Hedef'].value_counts().head(10).to_string())

print('\n' + '='*60)
print('✅ TÜM SÜTUNLAR MEVCUT VE ÇALIŞIYOR!')
print('='*60)
