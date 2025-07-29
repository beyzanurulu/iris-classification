import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== 2. YENİ GÖRSELLEŞTİRME PLOTLARI ===")
print()

# Iris veri setini yükle
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# DataFrame oluştur
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

print("--- 1. Korelasyon Heatmap ---")
# Korelasyon matrisi hesapla
corr = df[feature_names].corr()
print("Korelasyon matrisi:")
print(corr)
print()

# Matplotlib settings
plt.rcParams['figure.figsize'] = (10, 8)
plt.style.use('default')

# 1. Korelasyon Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Özellikler Arası Korelasyon Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(" Korelasyon heatmap kaydedildi: correlation_heatmap.png")
plt.close()

print()
print("--- 2. Pair Plot (Çiftli Dağılım) ---")
# 2. Pair Plot
plt.figure(figsize=(12, 10))
# Pair plot için manual implementation (seaborn pairplot alternatifi)
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
colors = ['red', 'green', 'blue']
species_labels = target_names

for i in range(4):
    for j in range(4):
        if i == j:
            # Diagonal: histogram
            for k, species in enumerate(species_labels):
                mask = df['species'] == species
                axes[i, j].hist(df.loc[mask, feature_names[i]], 
                               alpha=0.7, color=colors[k], label=species, bins=15)
            axes[i, j].set_xlabel(feature_names[i])
            axes[i, j].legend()
        else:
            # Off-diagonal: scatter plot
            for k, species in enumerate(species_labels):
                mask = df['species'] == species
                axes[i, j].scatter(df.loc[mask, feature_names[j]], 
                                  df.loc[mask, feature_names[i]], 
                                  c=colors[k], alpha=0.7, label=species)
            axes[i, j].set_xlabel(feature_names[j])
            axes[i, j].set_ylabel(feature_names[i])
            if i == 0 and j == 1:  # Sadece bir legend ekle
                axes[i, j].legend()

plt.suptitle('Pair Plot - Özellikler Arası İlişkiler', fontsize=16)
plt.tight_layout()
plt.savefig('pair_plot.png', dpi=300, bbox_inches='tight')
print(" Pair plot kaydedildi: pair_plot.png")
plt.close()

print()
print("--- 3. Box Plot (Kutu Grafikleri) ---")
# 3. Box Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(feature_names):
    # Her özellik için türlere göre box plot
    data_for_box = [df[df['species'] == species][feature].values for species in target_names]
    
    box_plot = axes[i].boxplot(data_for_box, labels=target_names, patch_artist=True)
    
    # Renklendirme
    colors_box = ['lightcoral', 'lightgreen', 'lightblue']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
    
    axes[i].set_title(f'{feature} - Türlere Göre Dağılım')
    axes[i].set_ylabel(feature)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Box Plots - Özellik Dağılımları', fontsize=16)
plt.tight_layout()
plt.savefig('box_plots.png', dpi=300, bbox_inches='tight')
print(" Box plots kaydedildi: box_plots.png")
plt.close()

print()
print("--- 4. Feature Importance (Özellik Önemliliği) ---")
# 4. Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importance = rf.feature_importances_

# Feature importance dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=True)

print("Özellik önemliliği skorları:")
for idx, row in importance_df.iterrows():
    print(f"{row['Feature']:<25}: {row['Importance']:.4f}")
print()

# Feature importance plot
plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], 
                color=['skyblue', 'lightgreen', 'salmon', 'plum'])
plt.xlabel('Özellik Önemliliği')
plt.title('Random Forest - Özellik Önemliliği')
plt.grid(axis='x', alpha=0.3)

# Değerleri çubukların üzerine yazdır
for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
    plt.text(importance + 0.005, i, f'{importance:.3f}', 
             va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print(" Feature importance plot kaydedildi: feature_importance.png")
plt.close()

print()
print("--- 5. Özellik Dağılım Histogramları ---")
# 5. Feature Distribution Histograms
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(feature_names):
    for j, species in enumerate(target_names):
        mask = df['species'] == species
        axes[i].hist(df.loc[mask, feature], alpha=0.7, 
                    color=colors[j], label=species, bins=15)
    
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frekans')
    axes[i].set_title(f'{feature} - Dağılım Histogramı')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Özellik Dağılım Histogramları', fontsize=16)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
print(" Özellik dağılım histogramları kaydedildi: feature_distributions.png")
plt.close()

print()
print("=== ÖZET ===")
print(" Korelasyon Heatmap: Özellikler arası ilişkiler")
print(" Pair Plot: Tüm özellik çiftleri arasındaki ilişkiler")
print(" Box Plots: Türlere göre özellik dağılımları")
print(" Feature Importance: En önemli özellikler")
print(" Histogramlar: Özellik değer dağılımları")
print()
print(" 5 adet görselleştirme dosyası oluşturuldu!")
print(" Dosyalar: correlation_heatmap.png, pair_plot.png, box_plots.png,")
print("          feature_importance.png, feature_distributions.png") 