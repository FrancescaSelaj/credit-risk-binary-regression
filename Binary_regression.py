# =============================================================================
# BINARY REGRESSION - CREDIT APPROVAL CASE STUDY
# Business Case: A leasing company wants to understand which companies
# should be granted credit, considering macroeconomic outlook and firm features.
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
PALETTE  = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
PATH     = r"C:\Users\franc\Desktop\Data Analysis\DatabaseProposteFido_202602.xlsx"
PATH_OUT = r"C:\Users\franc\Desktop\Data Analysis\\"

# =============================================================================
# DATA LOADING
# =============================================================================

df = pd.read_excel(PATH)
df['target'] = (df['Esito_finale'].str.strip().str.upper() == 'SI').astype(int)
# =============================================================================
# DATA DICTIONARY PRINT
# =============================================================================
print("\n" + "=" * 75)
print("DATA DICTIONARY (Italian to English Mapping for Reviewers)")
print("-" * 75)

data_dict = """
  Esito_finale          -> Final Outcome (Target: YES/NO)
  VALORE FIDO RICHIESTO -> Requested Credit Amount
  Fascia Fido           -> Credit Bracket
  FATTURATO             -> Revenue / Turnover
  DIPENDENTI            -> Number of Employees
  NUMERO_IMMOBILI       -> Number of Owned Properties
  AFFIDATA              -> Already Granted Credit (Boolean)
  REVISIONE             -> Under Review (Boolean)
  NATURA_GIURIDICA      -> Legal Entity Type
  STATO_ATTIVITA        -> Business Status (Active, Bankrupt, etc.)
  DATA_CALCOLO          -> Calculation Date
"""
#Print the dictionary, removing leading and trailing newlines
print(data_dict.strip('\n'))
print("=" * 75 + "\n")

# =============================================================================
# STEP 0 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("=" * 65)
print("STEP 0.1 — DATA LOADING")
print("=" * 65)
print(f"  Rows   : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print(f"  Memory : {df.memory_usage(deep=True).sum()/1024:.1f} KB")

# --- Variable Types ---
print("\n" + "=" * 65)
print("STEP 0.2 — VARIABLE TYPES")
print("=" * 65)
tipo_map = {
    'Target'          : ['Esito_finale'],
    'Identifiers'  : ['ID', 'CCIAA_IMPRESA'],
    'Score / Risk' : ['SCORE_INNOLVA', 'CLASSE_SCORE_INNOLVA', 'SCORE_SONEPAR', 'ITP'],
    'Financials'     : ['VALORE FIDO RICHIESTO', 'Fascia Fido', 'FATTURATO',
                         'DIPENDENTI', 'NUMERO_IMMOBILI', 'AFFIDATA', 'REVISIONE'],
    'Company Profile'     : ['NATURA_GIURIDICA', 'STATO_ATTIVITA', 'BRAND_SONEPAR',
                         'CODICE_ATECO', 'CODICE_ATECO_NP'],
    'Dates'            : ['DATA_CALCOLO', 'DATA_ISCRIZIONE', 'DATA_INZIO_ATTIVITA'],
    'Macro (External)' : ["Aspettative Inflazione Italia a 12 mesi (fonte: Banca d'Italia)",
                         "Crescita attesa PIL Italia per l'anno 2024 (Fonte: ISTAT)"]
}
for cat, cols in tipo_map.items():
    cols_ok = [c for c in cols if c in df.columns]
    print(f"\n  [{cat}]")
    for c in cols_ok:
        print(f"    - {c}  ({df[c].dtype})")

# --- Target ---
print("\n" + "=" * 65)
print("STEP 0.3 — TARGET")
print("=" * 65)
vc  = df['Esito_finale'].value_counts()
pct = df['Esito_finale'].value_counts(normalize=True) * 100
for val in vc.index:
    print(f"  {val:>5}  →  {vc[val]:>5} ({pct[val]:.1f}%)")
print(f"\n  ⚠️  YES/NO Imbalance = {vc['SI']/vc['NO']:.1f}x")
print(f"      Requires attention when choosing the z-threshold and handling imbalance")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("TARGET — Esito_finale Distribution", fontsize=13, fontweight='bold')
axes[0].bar(['YES (1)', 'NO (0)'], [vc['SI'], vc['NO']],
            color=[PALETTE[0], PALETTE[1]], width=0.5)
axes[0].set_title("Absolute Count")
axes[0].set_ylabel("N. observations")
for i, v in enumerate([vc['SI'], vc['NO']]):
    axes[0].text(i, v+30, str(v), ha='center', fontweight='bold')
axes[1].pie([vc['SI'], vc['NO']],
            labels=[f"YES\n{pct['SI']:.1f}%", f"NO\n{pct['NO']:.1f}%"],
            colors=[PALETTE[0], PALETTE[1]], startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title("Proportions")
plt.tight_layout()
plt.savefig(PATH_OUT + "plot_target.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot saved: plot_target.png")

# --- Missing Values ---
print("\n" + "=" * 65)
print("STEP 0.4 — MISSING VALUES")
print("=" * 65)
miss    = df.isnull().sum()
miss_pct = (miss / len(df)) * 100
miss_df = pd.DataFrame({'Missing_N': miss, 'Missing_%': miss_pct.round(1)})
miss_df = miss_df[miss_df['Missing_N'] > 0].sort_values('Missing_%', ascending=False)
print(miss_df.to_string())
print("\n Diagnostic summary:")
for col, row in miss_df.iterrows():
    p = row['Missing_%']
    flag = "🔴 CRITICAL" if p > 40 else ("🟡 HIGH" if p > 20 else "🟢 MANAGEABLE")
    print(f"    {col:<45} {p:>5.1f}%  {flag}")

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(df[miss_df.index.tolist()].isnull().astype(int).T,
            ax=ax, cbar=False, cmap=['#e8f5e9', '#F44336'],
            yticklabels=miss_df.index.tolist(), xticklabels=False)
ax.set_title("Missing values map (red = NaN)", fontsize=12, fontweight='bold')
ax.set_xlabel("Observations")
plt.tight_layout()
plt.savefig(PATH_OUT + "plot_missing.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot saved: plot_missing.png")

# --- Macro ---
print("\n" + "=" * 65)
print("STEP 0.5 — EXTERNAL MACRO VARIABLES (constants → dropped from the model)")
print("=" * 65)
col_inflaz = "Aspettative Inflazione Italia a 12 mesi (fonte: Banca d'Italia)"
col_pil    = "Crescita attesa PIL Italia per l'anno 2024 (Fonte: ISTAT)"
print(f"  Inflation unique values: {df[col_inflaz].nunique()} → {df[col_inflaz].unique()}")
print(f"  GDP       unique values: {df[col_pil].nunique()}    → constant")
print("  ⚠️  Variance = 0 → not identifiable in the model → dropped")
print("  📌 Mentioned in the report narrative to contextualize the business case")

# --- Descrittive numeriche raw ---
print("\n" + "=" * 65)
print("STEP 0.6 — NUMERICAL DESCRIPTIVE STATISTICS (original variables)")
print("=" * 65)
num_raw = ['SCORE_INNOLVA','ITP','FATTURATO','DIPENDENTI','NUMERO_IMMOBILI','VALORE FIDO RICHIESTO']
desc_raw = df[num_raw].describe(percentiles=[.25,.5,.75,.90,.95]).T
desc_raw['skewness'] = df[num_raw].skew().round(2)
desc_raw['kurtosis'] = df[num_raw].kurtosis().round(2)
print(desc_raw[['count','mean','std','min','25%','50%','75%','90%','95%','max','skewness','kurtosis']].to_string())
print("""
  GUIDE: |skewness|>1 → highly skewed 🔴 | kurtosis>3 → heavy tails ⚠️
""")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("STEP 0.6 — Numerical Variable Distributions (raw)", fontsize=13, fontweight='bold')
axes = axes.flatten()
for i, col in enumerate(num_raw):
    data = df[col].dropna()
    axes[i].hist(data, bins=40, color=PALETTE[0], alpha=0.8, edgecolor='white')
    axes[i].axvline(data.median(), color=PALETTE[1], linestyle='--', linewidth=1.5,
                    label=f'Med: {data.median():,.0f}')
    axes[i].axvline(data.mean(),   color=PALETTE[2], linestyle='-',  linewidth=1.5,
                    label=f'Mean: {data.mean():,.0f}')
    axes[i].set_title(col, fontsize=10)
    axes[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig(PATH_OUT + "plot_distribuzioni_raw.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot saved: plot_distribuzioni_raw.png")

# --- Categorical Variables ---
print("\n" + "=" * 65)
print("STEP 0.7 — CATEGORICAL VARIABLES")
print("=" * 65)

cat_cols = ['CLASSE_SCORE_INNOLVA','SCORE_SONEPAR','NATURA_GIURIDICA','STATO_ATTIVITA','Fascia Fido']

for col in cat_cols:
    vc2 = df[col].value_counts(dropna=False)
    print(f"\n  [{col}]  ({vc2.shape[0]} categories)")
    print(vc2.to_string())

labels_cat = {
    'CLASSE_SCORE_INNOLVA': 'Innolva Score Class',
    'SCORE_SONEPAR':        'Sonepar Score',
    'NATURA_GIURIDICA':     'Legal Entity Type',
    'STATO_ATTIVITA':       'Business Status',
    'Fascia Fido':          'Credit Bracket'
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Categorical Variable Distributions",
             fontsize=13, fontweight='bold')
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    vc3 = df[col].value_counts(dropna=False).head(12)
    axes[i].barh(vc3.index.astype(str)[::-1], vc3.values[::-1],
                 color=PALETTE[i % len(PALETTE)])
    axes[i].set_title(labels_cat.get(col, col), fontsize=10, fontweight='bold')
    axes[i].set_xlabel("Count")
    axes[i].set_xlim(0, vc3.values.max() * 1.12)
    for j, v in enumerate(vc3.values[::-1]):
        axes[i].text(v + vc3.values.max() * 0.01, j, str(v),
                     va='center', ha='left',
                     fontsize=8, color='black', fontweight='bold')
axes[5].axis('off')
plt.tight_layout()
plt.savefig(PATH_OUT + "plot_categoriali.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot saved: plot_categoriali.png")
# =============================================================================
# STEP 1 — FEATURE ENGINEERING AND PREPROCESSING
# =============================================================================

print("\n" + "=" * 65)
print("STEP 1 — FEATURE ENGINEERING AND PREPROCESSING")
print("=" * 65)
print("  Philosophy: missing values are not random → they are information")
print("  Better a real-world model that answers the business case")
print("  than a statistically perfect but unrealistic one")

# --- 1.1 SCORE_INNOLVA = 1000 ---
print("\nSTEP 1.1 — INVESTIGATING SCORE_INNOLVA = 1000")
mask_1000   = df['SCORE_INNOLVA'] == 1000
n_1000      = mask_1000.sum()
n_normal    = (~mask_1000).sum()
t_1000      = df[mask_1000]['target'].value_counts()
t_norm      = df[~mask_1000]['target'].value_counts()
print(f"  Records with score=1000   : {n_1000} ({n_1000/len(df)*100:.1f}%)")
print(f"  Approval rate score=1000: {t_1000.get(1,0)/n_1000*100:.1f}%  ← molto più bassa!")
print(f"  Approval rate score<1000: {t_norm.get(1,0)/n_normal*100:.1f}%")
print(f"  Innolva Class score=1000: {df[mask_1000]['CLASSE_SCORE_INNOLVA'].value_counts().to_dict()}")
print(f"  STATO_ATTIVITA score=1000: {df[mask_1000]['STATO_ATTIVITA'].value_counts().to_dict()}")
print("  → DECISION: it is a sentinel value → flag_score_nd=1 + NaN in score")

df['flag_score_nd'] = (df['SCORE_INNOLVA'] == 1000).astype(int)
df.loc[df['SCORE_INNOLVA'] == 1000, 'SCORE_INNOLVA'] = np.nan

# --- 1.2 ITP ---
print("\nSTEP 1.2 — ITP: cleaning -999 sentinel and missing flag")
n_999 = (df['ITP'] == -999).sum()
print(f"  ITP = -999 (sentinel): {n_999} records → replaced with NaN")
df.loc[df['ITP'] == -999, 'ITP'] = np.nan
df['flag_itp'] = df['ITP'].notna().astype(int)
appr_itp = df.groupby('flag_itp')['target'].mean()
print(f"  flag_itp=0 (unknown): {(df['flag_itp']==0).sum():,}  approv: {appr_itp[0]:.1%}")
print(f"  flag_itp=1 (available): {df['flag_itp'].sum():,}  approv: {appr_itp[1]:.1%}")
print("  → DO NOT impute: missing = unknown company = risk information")

# --- 1.3 FATTURATO ---
print("\nSTEP 1.3 — FATTURATO: flag + log (no imputation)")
df['flag_fatturato'] = df['FATTURATO'].notna().astype(int)
df['log_fatturato']  = np.where(
    df['FATTURATO'].notna() & (df['FATTURATO'] > 0),
    np.log1p(df['FATTURATO']), np.nan)
appr_fatt = df.groupby('flag_fatturato')['target'].mean()
print(f"  flag_fatturato=0 (missing) : {(df['flag_fatturato']==0).sum():,}  approv: {appr_fatt[0]:.1%}")
print(f"  flag_fatturato=1 (present) : {df['flag_fatturato'].sum():,}  approv: {appr_fatt[1]:.1%}")
print("  → DO NOT impute (41% missing): 2408 fake values = pure noise")
print("  → flag captures financial opacity, log the value where available")

# --- 1.4 Log Transformations ---
print("\nSTEP 1.4 — LOG TRANSFORMATIONS (variables with extreme skewness)")
df['log_dipendenti'] = np.log1p(df['DIPENDENTI'])
df['log_immobili']   = np.log1p(df['NUMERO_IMMOBILI'])
for orig, trasf in [('DIPENDENTI','log_dipendenti'),('NUMERO_IMMOBILI','log_immobili')]:
    print(f"  {orig:<20} skew prima: {df[orig].skew():>7.2f}  →  dopo log: {df[trasf].skew():.2f}")

# --- 1.5 Ordinal Encoding of CLASSE_SCORE_INNOLVA ---
print("\nSTEP 1.5 — ENCODING CLASSE_SCORE_INNOLVA (ordinal, 0=ND → 9=AAA)")
classe_order = {'AAA':9,'AA':8,'A':7,'BBB':6,'BB':5,'B':4,'CCC':3,'CC':2,'C':1,'ND':0}
df['score_innolva_ord'] = df['CLASSE_SCORE_INNOLVA'].map(classe_order)
for k, v in classe_order.items():
    n  = (df['CLASSE_SCORE_INNOLVA'] == k).sum()
    tr = df[df['CLASSE_SCORE_INNOLVA'] == k]['target'].mean()
    print(f"  {k:<5} → {v}  (n={n:>4}, approv={tr:.1%})")
print("  → Ordinal encoding: only 1 interpretable beta (better class → +prob. approval)")

# --- 1.6 Encoding SCORE_SONEPAR (empirical rank from data) ---
print("\nSTEP 1.6 — SCORE_SONEPAR ENCODING (empirical rank from approval rate)")
tasso_sonepar = df.groupby('SCORE_SONEPAR')['target'].mean().sort_values()
rank_sonepar = tasso_sonepar.rank(method='dense').astype(int)
df['score_sonepar_ord'] = df['SCORE_SONEPAR'].map(rank_sonepar)
print(df.groupby('score_sonepar_ord').agg(
    category=('SCORE_SONEPAR', 'first'),
    n=('target','count'),
    approv=('target','mean')
).sort_values('score_sonepar_ord')
.assign(approv=lambda x: x['approv'].apply(lambda v: f"{v:.1%}")))

# --- 1.7 Date Feature Engineering ---
print("\nSTEP 1.7 — DATE FEATURE ENGINEERING (seniority/tenure)")
def parse_date_int(s):
    return pd.to_datetime(s.astype(str).str.split('.').str[0], format='%Y%m%d', errors='coerce')
df['data_calcolo_dt']    = parse_date_int(df['DATA_CALCOLO'])
df['data_iscrizione_dt'] = parse_date_int(df['DATA_ISCRIZIONE'])
df['data_inizio_dt']     = parse_date_int(df['DATA_INZIO_ATTIVITA'])
df['anzianita_azienda_anni'] = (df['data_calcolo_dt'] - df['data_inizio_dt']).dt.days / 365.25
df['anzianita_cliente_anni'] = (df['data_calcolo_dt'] - df['data_iscrizione_dt']).dt.days / 365.25
print(f"  Company age    mean: {df['anzianita_azienda_anni'].mean():.1f} years  missing: {df['anzianita_azienda_anni'].isna().sum()}")
print(f"  Client tenure  mean: {df['anzianita_cliente_anni'].mean():.1f} years  missing: {df['anzianita_cliente_anni'].isna().sum()}")

# --- 1.8 Company Profile Categoricals ---
print("\nSTEP 1.8 — COMPANY PROFILE CATEGORICAL ENCODING")
df['azienda_attiva'] = (df['STATO_ATTIVITA'] == 'A').astype(int)
appr_att = df.groupby('azienda_attiva')['target'].mean()
print(f"  azienda_attiva=0 (inactive): {(df['azienda_attiva']==0).sum():,}  approval: {appr_att[0]:.1%}")
print(f"  azienda_attiva=1 (active)    : {df['azienda_attiva'].sum():,}  approval: {appr_att[1]:.1%}")
print("  → Difference ~28pp: very strong credit discriminator")

top5_natura = df['NATURA_GIURIDICA'].value_counts().nlargest(5).index.tolist()
df['natura_giuridica_clean'] = df['NATURA_GIURIDICA'].apply(
    lambda x: x if x in top5_natura else 'Other')
print(f"\n  NATURA_GIURIDICA → top5 + Other:")
print(df.groupby('natura_giuridica_clean').agg(
    n=('target','count'), approv=('target','mean')
).assign(approv=lambda x: x['approv'].apply(lambda v: f"{v:.1%}")).to_string())

# --- 1.9 Final Dataset ---
print("\nSTEP 1.9 — FINAL DATASET")
features_finali = [
    'SCORE_INNOLVA','flag_score_nd',
    'score_innolva_ord','score_sonepar_ord',
    'flag_itp','ITP',
    'VALORE FIDO RICHIESTO',
    'flag_fatturato','log_fatturato',
    'log_dipendenti','log_immobili',
    'AFFIDATA','REVISIONE',
    'azienda_attiva','natura_giuridica_clean',
    'anzianita_azienda_anni','anzianita_cliente_anni',
    'target'
]
df_model = df[features_finali].copy()
print(f"  Rows    : {df_model.shape[0]:,}")
print(f"  Features : {df_model.shape[1]-1} + 1 target")
print("  Remaining missing values:")
miss_res = df_model.isnull().sum()
print(miss_res[miss_res > 0].to_string())

# --- 1.10 Bivariate Analysis ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Approval Rate by Feature — Bivariate Analysis",
             fontsize=13, fontweight='bold')
axes = axes.flatten()

# 1. Innolva Score
gr1 = df.groupby('score_innolva_ord')['target'].mean().reset_index()
axes[0].bar(gr1['score_innolva_ord'], gr1['target']*100, color=PALETTE[0])
axes[0].set_title("Approval by Innolva Class")
axes[0].set_xlabel("Ordinal Score (0=ND, 9=AAA)")
axes[0].axhline(df['target'].mean()*100, color='red', linestyle='--', alpha=0.7)

# 2. ITP Flag
gr2 = df.groupby('flag_itp')['target'].mean().reset_index()
bars2 = axes[1].bar(['Missing ITP','Available ITP'], gr2['target']*100,
                    color=[PALETTE[1], PALETTE[2]])
axes[1].set_title("Approvazione per flag_itp")
axes[1].axhline(df['target'].mean()*100, color='red', linestyle='--', alpha=0.7)
for bar, val in zip(bars2, gr2['target']*100):
    axes[1].text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val:.1f}%', ha='center', fontweight='bold')

# 3. Revenue Flag
gr3 = df.groupby('flag_fatturato')['target'].mean().reset_index()
bars3 = axes[2].bar(['Missing Revenue','Present Revenue'], gr3['target']*100,
                    color=[PALETTE[1], PALETTE[2]])
axes[2].set_title("Approval by flag_fatturato")
axes[2].axhline(df['target'].mean()*100, color='red', linestyle='--', alpha=0.7)
for bar, val in zip(bars3, gr3['target']*100):
    axes[2].text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val:.1f}%', ha='center', fontweight='bold')

# 4. Activity Status
gr4 = df.groupby('azienda_attiva')['target'].mean().reset_index()
bars4 = axes[3].bar(['Inactive','Active'], gr4['target']*100,
                    color=[PALETTE[1], PALETTE[2]])
axes[3].set_title("Approval by Business Status")
axes[3].axhline(df['target'].mean()*100, color='red', linestyle='--', alpha=0.7)
for bar, val in zip(bars4, gr4['target']*100):
    axes[3].text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val:.1f}%', ha='center', fontweight='bold')

# 5. Legal Entity Type
gr5 = df.groupby('natura_giuridica_clean')['target'].mean().sort_values()
axes[4].barh(gr5.index, gr5.values*100, color=PALETTE[4])
axes[4].set_title("Approval by Legal Entity Type")
axes[4].axvline(df['target'].mean()*100, color='red', linestyle='--', alpha=0.7)

# 6. Sonepar Score
gr6 = df.groupby('score_sonepar_ord')['target'].mean().reset_index()
axes[5].bar(gr6['score_sonepar_ord'], gr6['target']*100, color=PALETTE[3])
axes[5].set_title("Approval by Sonepar Score")
axes[5].set_xlabel("Ordinal Score")
axes[5].axhline(df['target'].mean()*100, color='red', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(PATH_OUT + "plot_bivariata_approvazione.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot saved: plot_bivariata_approvazione.png")

# Intermediate Saving
df_model.to_csv(PATH_OUT + "preprocessed_dataset.csv", index=False)
print("  ✅ preprocessed_dataset.csv salvato")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
PALETTE = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
PATH     = r"C:\Users\franc\Desktop\Data Analysis\DatabaseProposteFido_202602.xlsx"
PATH_OUT = r"C:\Users\franc\Desktop\Data Analysis\\"

# Quick dataset reconstruction
df = pd.read_excel(PATH)
df['target'] = (df['Esito_finale'].str.strip().str.upper() == 'SI').astype(int)

df['flag_score_nd'] = (df['SCORE_INNOLVA'] == 1000).astype(int)
df.loc[df['SCORE_INNOLVA'] == 1000, 'SCORE_INNOLVA'] = np.nan
df.loc[df['ITP'] == -999, 'ITP'] = np.nan
df['flag_itp'] = df['ITP'].notna().astype(int)

classe_order = {'AAA':9,'AA':8,'A':7,'BBB':6,'BB':5,'B':4,'CCC':3,'CC':2,'C':1,'ND':0}
df['score_innolva_ord'] = df['CLASSE_SCORE_INNOLVA'].map(classe_order)

tasso_sonepar = df.groupby('SCORE_SONEPAR')['target'].mean().sort_values()
rank_sonepar  = tasso_sonepar.rank(method='dense').astype(int)
df['score_sonepar_ord'] = df['SCORE_SONEPAR'].map(rank_sonepar)

df['azienda_attiva'] = (df['STATO_ATTIVITA'] == 'A').astype(int)

media_target = df['target'].mean() * 100

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Approval Rate by Feature — Bivariate Analysis",
             fontsize=13, fontweight='bold')

# --- 1. Innolva Class ---
gr1 = df.groupby('score_innolva_ord')['target'].mean().reset_index()
axes[0].bar(gr1['score_innolva_ord'], gr1['target']*100,
            color=PALETTE[0], width=0.7)
axes[0].axhline(media_target, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
axes[0].set_title("Approval by Innolva Class", fontsize=10, fontweight='bold')
axes[0].set_xlabel("Ordinal Score (0=ND, 9=AAA)")
axes[0].set_ylabel("Approval Rate (%)")
axes[0].set_ylim(0, 105)
for _, row in gr1.iterrows():
    axes[0].text(row['score_innolva_ord'], row['target']*100 + 1.5,
                 f"{row['target']*100:.0f}%",
                 ha='center', fontsize=7.5, fontweight='bold')

# --- 2. Business Status ---
gr2 = df.groupby('azienda_attiva')['target'].mean().reset_index()
bars2 = axes[1].bar(['Non attiva', 'Attiva'], gr2['target']*100,
                    color=[PALETTE[1], PALETTE[2]], width=0.5)
axes[1].axhline(media_target, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
axes[1].set_title("Approval by Business Status", fontsize=10, fontweight='bold')
axes[1].set_ylabel("Approval Rate (%)")
axes[1].set_ylim(0, 105)
for bar, val in zip(bars2, gr2['target']*100):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 1.5,
                 f"{val:.1f}%", ha='center', fontsize=10, fontweight='bold')

# --- 3. ITP Flag ---
gr3 = df.groupby('flag_itp')['target'].mean().reset_index()
bars3 = axes[2].bar(['Missing ITP', 'Available ITP'], gr3['target']*100,
                    color=[PALETTE[1], PALETTE[2]], width=0.5)
axes[2].axhline(media_target, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
axes[2].set_title("Approval by ITP Availability", fontsize=10, fontweight='bold')
axes[2].set_ylabel("Approval Rate (%)")
axes[2].set_ylim(0, 105)
for bar, val in zip(bars3, gr3['target']*100):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 1.5,
                 f"{val:.1f}%", ha='center', fontsize=10, fontweight='bold')

# --- 4. Sonepar Score ---
gr4 = df.groupby('score_sonepar_ord')['target'].mean().reset_index()
axes[3].bar(gr4['score_sonepar_ord'], gr4['target']*100,
            color=PALETTE[3], width=0.7)
axes[3].axhline(media_target, color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
axes[3].set_title("Approval by Sonepar Score", fontsize=10, fontweight='bold')
axes[3].set_xlabel("Ordinal Score (1=peggiore, 18=migliore)")
axes[3].set_ylabel("Approval Rate (%)")
axes[3].set_ylim(0, 105)

plt.tight_layout()
plt.savefig(PATH_OUT + "plot_bivariate_selection.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Plot saved: plot_bivariate_selection.png")
# =============================================================================
# STEP 1B — DISTRIBUTION ANALYSIS OF FINAL FEATURES
# (pre-model estimation: skewness, kurtosis, outliers, correlation)
# =============================================================================

print("\n" + "=" * 65)
print("STEP 1B — FINAL FEATURES DISTRIBUTION ANALYSIS POST-PREPROCESSING")
print("=" * 65)
print("  Purpose: understand the distribution shape of the")
print("  engineered features before proceeding with Logit estimation")

num_continue = ['SCORE_INNOLVA','ITP','log_fatturato',
                'log_dipendenti','log_immobili',
                'anzianita_azienda_anni','anzianita_cliente_anni',
                'VALORE FIDO RICHIESTO']

# --- 1B.1 Statistiche descrittive complete ---
print("\nSTEP 1B.1 — STATISTICHE DESCRITTIVE FEATURES NUMERICHE CONTINUE")
desc = df_model[num_continue].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
desc['skewness']  = df_model[num_continue].skew().round(3)
desc['kurtosis']  = df_model[num_continue].kurtosis().round(3)
desc['missing_%'] = (df_model[num_continue].isna().mean()*100).round(1)
print(desc[['count','mean','std','min','1%','5%','25%','50%',
            '75%','95%','99%','max','skewness','kurtosis','missing_%']].to_string())
print("""
  GUIDA LETTURA SKEWNESS:
    |skew| < 0.5   → quasi simmetrica           ✅  colore verde nei grafici
    |skew| 0.5-1   → moderatamente asimmetrica  ⚠️  colore arancio
    |skew| > 1     → fortemente asimmetrica     🔴  colore rosso
  GUIDA LETTURA KURTOSIS:
    kurtosis > 3   → code pesanti, outlier probabili  ⚠️
    kurtosis > 10  → outlier severi presenti           🔴
""")

# --- 1B.2 Flag e ordinali ---
print("STEP 1B.2 — VARIABILI ORDINALI E FLAG (distribuzione + tasso approvazione)")
for col in ['score_innolva_ord','score_sonepar_ord','flag_itp',
            'flag_fatturato','flag_score_nd','azienda_attiva','AFFIDATA','REVISIONE']:
    vc2 = df_model[col].value_counts(dropna=False).sort_index()
    print(f"\n  [{col}]")
    for val, cnt in vc2.items():
        mask = df_model[col] == val
        appr = df_model[mask]['target'].mean() if mask.sum() > 0 else np.nan
        appr_s = f"{appr:.1%}" if not (isinstance(appr, float) and np.isnan(appr)) else "N/A"
        print(f"    {str(val):>5}  →  {cnt:>5} ({cnt/len(df_model)*100:>5.1f}%)  approv: {appr_s}")

# --- 1B.3 Grafici distribuzioni: istogramma + boxplot + SI vs NO ---
print("\nSTEP 1B.3 — GRAFICI: istogramma | boxplot | SI vs NO per ogni feature")
fig, axes = plt.subplots(len(num_continue), 3, figsize=(16, len(num_continue)*3.5))
fig.suptitle("Distribuzioni Features Numeriche — Post Preprocessing\n"
             "(verde=ok | arancio=moderata | rosso=asimmetrica)",
             fontsize=12, fontweight='bold', y=1.01)

for i, col in enumerate(num_continue):
    data_c = df_model[col].dropna()
    sk     = data_c.skew()
    ku     = data_c.kurtosis()
    color  = PALETTE[2] if abs(sk) < 0.5 else (PALETTE[3] if abs(sk) < 1 else PALETTE[1])

    # Istogramma
    axes[i,0].hist(data_c, bins=40, color=color, alpha=0.8, edgecolor='white')
    axes[i,0].axvline(data_c.median(), color='red',  linestyle='--', linewidth=1.5,
                      label=f'Med: {data_c.median():.2f}')
    axes[i,0].axvline(data_c.mean(),   color='blue', linestyle='-',  linewidth=1.5,
                      label=f'Mea: {data_c.mean():.2f}')
    axes[i,0].set_title(f'{col}\nskew={sk:.2f}  kurt={ku:.2f}', fontsize=9)
    axes[i,0].legend(fontsize=7)
    axes[i,0].set_ylabel("Frequenza")

    # Boxplot
    axes[i,1].boxplot(data_c, vert=True, patch_artist=True,
                      boxprops=dict(facecolor=color, alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      flierprops=dict(marker='o', markersize=2,
                                      markerfacecolor='gray', alpha=0.3))
    axes[i,1].set_title(f'{col} — Boxplot\n(pallini grigi = outlier)', fontsize=9)

    # SI vs NO sovrapposti
    axes[i,2].hist(df_model[df_model['target']==1][col].dropna(), bins=30,
                   alpha=0.6, color=PALETTE[0], label='SI (approvato)',
                   density=True, edgecolor='white')
    axes[i,2].hist(df_model[df_model['target']==0][col].dropna(), bins=30,
                   alpha=0.6, color=PALETTE[1], label='NO (rifiutato)',
                   density=True, edgecolor='white')
    axes[i,2].set_title(f'{col} — SI vs NO\n(separazione = potere predittivo)', fontsize=9)
    axes[i,2].legend(fontsize=7)
    axes[i,2].set_ylabel("Densità")

plt.tight_layout()
plt.savefig(PATH_OUT + "plot_distribuzioni_features_finali.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Grafico salvato: plot_distribuzioni_features_finali.png")
print("  📌 Colonna 3 (SI vs NO): più le distribuzioni SI/NO sono separate,")
print("     più la variabile è predittiva del target")

# --- 1B.4 Analisi outlier IQR ---
print("\nSTEP 1B.4 — ANALISI OUTLIER (metodo IQR)")
print("  Nota: in Logit gli outlier non si eliminano automaticamente,")
print("  ma bisogna sapere quanti sono e dove stanno\n")
for col in num_continue:
    d     = df_model[col].dropna()
    Q1    = d.quantile(0.25)
    Q3    = d.quantile(0.75)
    IQR   = Q3 - Q1
    lb    = Q1 - 1.5*IQR
    ub    = Q3 + 1.5*IQR
    n_out = ((d < lb) | (d > ub)).sum()
    pct_out = n_out / len(d) * 100
    flag  = "✅" if pct_out < 2 else ("⚠️" if pct_out < 10 else "🔴")
    print(f"  {col:<30} outlier: {n_out:>5} ({pct_out:>5.1f}%)  "
          f"bounds: [{lb:.2f}, {ub:.2f}]  {flag}")

# --- 1B.5 Matrice di correlazione ---
print("\nSTEP 1B.5 — MATRICE DI CORRELAZIONE")
cols_corr = ['SCORE_INNOLVA','score_innolva_ord','score_sonepar_ord',
             'ITP','log_fatturato','log_dipendenti','log_immobili',
             'anzianita_azienda_anni','anzianita_cliente_anni',
             'VALORE FIDO RICHIESTO','flag_itp','flag_fatturato',
             'azienda_attiva','AFFIDATA','REVISIONE','target']

corr_matrix = df_model[cols_corr].corr()

print("\n  Correlazione di Pearson con il TARGET (ordinata per |r|):")
corr_target = corr_matrix['target'].drop('target').sort_values(key=abs, ascending=False)
for feat, val in corr_target.items():
    bar   = '█' * int(abs(val)*30)
    segno = '+' if val > 0 else '-'
    flag  = "🔴" if abs(val) > 0.3 else ("⚠️" if abs(val) > 0.1 else "  ")
    print(f"  {feat:<32} {val:>+.3f}  {segno}{bar} {flag}")

print("\n  Coppie con |r| > 0.5 (possibile multicollinearità → beta instabili):")
found = False
for i in range(len(cols_corr)-1):
    for j in range(i+1, len(cols_corr)-1):
        c1, c2 = cols_corr[i], cols_corr[j]
        r = corr_matrix.loc[c1, c2]
        if abs(r) > 0.5:
            print(f"  {c1:<32} ↔ {c2:<32}  r = {r:.3f}  ⚠️")
            found = True
if not found:
    print("  ✅ Nessuna coppia con |r| > 0.5 — nessun rischio di multicollinearità")

# Heatmap
fig, ax = plt.subplots(figsize=(14, 11))
mask_tri = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask_tri, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, annot_kws={'size': 7})
ax.set_title("Matrice di Correlazione — Features Finali + Target",
             fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(PATH_OUT + "plot_correlazione.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Grafico salvato: plot_correlazione.png")

# =============================================================================
# RIEPILOGO FINALE
# =============================================================================

print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │ RIEPILOGO COMPLETO — PRONTO PER STEP 2 (STIMA MODELLO LOGIT)    │
  ├──────────────────────────────────────────────────────────────────┤
  │ TARGET         : 81% SI / 19% NO → sbilanciato                  │
  │ MACRO          : eliminate (costanti), citate nel report         │
  │ SCORE_INNOLVA  : sentinella 1000 → flag_score_nd + NaN          │
  │ ITP            : flag_itp (no imputa.) + valore dove disponibile │
  │ FATTURATO      : flag_fatturato (no imputa.) + log_fatturato     │
  │ DIPENDENTI     : → log_dipendenti                                │
  │ IMMOBILI       : → log_immobili                                  │
  │ INNOLVA classe : → score_innolva_ord (0-9)                       │
  │ SONEPAR        : → score_sonepar_ord (-1 → 10)                   │
  │ STATO ATTIV.   : → azienda_attiva (0/1)                          │
  │ NATURA GIUR.   : → top5 + Altro                                  │
  │ DATE           : → anzianita_azienda + anzianita_cliente          │
  ├──────────────────────────────────────────────────────────────────┤
  │ NEXT → STEP 2:                                                   │
  │   1. Train / Validation / Test split (60/20/20)                 │
  │   2. Gestione sbilanciamento (class_weight o threshold tuning)  │
  │   3. One-hot encoding natura_giuridica_clean                    │
  │   4. Standardizzazione features continue                        │
  │   5. Stima modello Logit (MLE via BFGS)                         │
  │   6. Coefficienti, standard error, t-ratio, p-value             │
  │   7. Effetti marginali medi                                     │
  │   8. McFadden R², Forecast R², AUC/ROC, confusion matrix        │
  └──────────────────────────────────────────────────────────────────┘
""")
# =============================================================================
# STEP 1B.3 — GRAFICI DISTRIBUZIONI SEPARATI (uno per variabile)
# Da aggiungere alla fine di analisi_completa_v2.py
# OPPURE eseguire come file separato dopo aver già fatto girare analisi_completa_v2.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
PALETTE  = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
PATH     = r"C:\Users\franc\Desktop\Data Analysis\DatabaseProposteFido_202602.xlsx"
PATH_OUT = r"C:\Users\franc\Desktop\Data Analysis\\"

# =============================================================================
# RICOSTRUZIONE RAPIDA DEL DATASET PREPROCESSATO
# =============================================================================

df = pd.read_excel(PATH)
df['target'] = (df['Esito_finale'].str.strip().str.upper() == 'SI').astype(int)

# Pulizie e trasformazioni
df['flag_score_nd'] = (df['SCORE_INNOLVA'] == 1000).astype(int)
df.loc[df['SCORE_INNOLVA'] == 1000, 'SCORE_INNOLVA'] = np.nan
df.loc[df['ITP'] == -999, 'ITP'] = np.nan
df['flag_itp']       = df['ITP'].notna().astype(int)
df['flag_fatturato'] = df['FATTURATO'].notna().astype(int)
df['log_fatturato']  = np.where(df['FATTURATO'].notna() & (df['FATTURATO']>0),
                                 np.log1p(df['FATTURATO']), np.nan)
df['log_dipendenti'] = np.log1p(df['DIPENDENTI'])
df['log_immobili']   = np.log1p(df['NUMERO_IMMOBILI'])

def parse_date_int(s):
    return pd.to_datetime(s.astype(str).str.split('.').str[0], format='%Y%m%d', errors='coerce')
df['data_calcolo_dt']        = parse_date_int(df['DATA_CALCOLO'])
df['data_iscrizione_dt']     = parse_date_int(df['DATA_ISCRIZIONE'])
df['data_inizio_dt']         = parse_date_int(df['DATA_INZIO_ATTIVITA'])
df['anzianita_azienda_anni'] = (df['data_calcolo_dt'] - df['data_inizio_dt']).dt.days / 365.25
df['anzianita_cliente_anni'] = (df['data_calcolo_dt'] - df['data_iscrizione_dt']).dt.days / 365.25

features_finali = [
    'SCORE_INNOLVA','flag_score_nd','ITP','log_fatturato',
    'log_dipendenti','log_immobili',
    'anzianita_azienda_anni','anzianita_cliente_anni',
    'VALORE FIDO RICHIESTO','flag_itp','flag_fatturato',
    'target'
]
df_model = df[features_finali].copy()
"""
# =============================================================================
# GRAFICI SEPARATI — uno per variabile numerica continua
# Ogni grafico ha: istogramma, boxplot, SI vs NO
# =============================================================================

num_continue = [
    ('SCORE_INNOLVA',          'Score Innolva (numerico, post pulizia sentinella)'),
    ('ITP',                    'ITP — Indice Tempestività Pagamenti (giorni)'),
    ('log_fatturato',          'Log Fatturato (log1p, NaN dove mancante)'),
    ('log_dipendenti',         'Log Dipendenti (log1p)'),
    ('log_immobili',           'Log Numero Immobili (log1p)'),
    ('anzianita_azienda_anni', 'Anzianità Azienda (anni da inizio attività)'),
    ('anzianita_cliente_anni', 'Anzianità Cliente Sonepar (anni da iscrizione)'),
    ('VALORE FIDO RICHIESTO',  'Valore Fido Richiesto (€)'),
]

print("Generazione grafici distribuzioni separati...")
print("=" * 60)

for col, titolo in num_continue:

    data_all = df_model[col].dropna()
    data_si  = df_model[df_model['target']==1][col].dropna()
    data_no  = df_model[df_model['target']==0][col].dropna()

    sk  = data_all.skew()
    ku  = data_all.kurtosis()
    med = data_all.median()
    mea = data_all.mean()
    mis = df_model[col].isna().mean() * 100

    # Colore in base a skewness
    if abs(sk) < 0.5:
        color = PALETTE[2]   # verde → simmetrica
        label_sk = "✅ simmetrica"
    elif abs(sk) < 1:
        color = PALETTE[3]   # arancio → moderata
        label_sk = "⚠️ moderatamente asimmetrica"
    else:
        color = PALETTE[1]   # rosso → asimmetrica
        label_sk = "🔴 fortemente asimmetrica"

    # Outlier IQR
    Q1  = data_all.quantile(0.25)
    Q3  = data_all.quantile(0.75)
    IQR = Q3 - Q1
    lb  = Q1 - 1.5*IQR
    ub  = Q3 + 1.5*IQR
    n_out    = ((data_all < lb) | (data_all > ub)).sum()
    pct_out  = n_out / len(data_all) * 100
    flag_out = "✅" if pct_out < 2 else ("⚠️" if pct_out < 10 else "🔴")

    # --- FIGURA: 1 riga, 3 colonne ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{titolo}\n"
                 f"skew={sk:.2f} ({label_sk})  |  kurt={ku:.2f}  |  missing={mis:.1f}%  |  "
                 f"outlier={n_out} ({pct_out:.1f}%) {flag_out}",
                 fontsize=11, fontweight='bold')

    # --- Colonna 1: Istogramma ---
    axes[0].hist(data_all, bins=40, color=color, alpha=0.85, edgecolor='white')
    axes[0].axvline(med, color='red',  linestyle='--', linewidth=2,
                    label=f'Mediana: {med:.2f}')
    axes[0].axvline(mea, color='blue', linestyle='-',  linewidth=2,
                    label=f'Media: {mea:.2f}')
    axes[0].set_title("Distribuzione (tutti i record)", fontsize=10)
    axes[0].set_xlabel("Valore")
    axes[0].set_ylabel("Frequenza")
    axes[0].legend(fontsize=9)

    # Info statistiche nel grafico
    stats_text = (f"n={len(data_all):,}\n"
                  f"min={data_all.min():.2f}\n"
                  f"max={data_all.max():.2f}\n"
                  f"Q1={Q1:.2f}\n"
                  f"Q3={Q3:.2f}")
    axes[0].text(0.97, 0.97, stats_text, transform=axes[0].transAxes,
                 fontsize=8, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Colonna 2: Boxplot con outlier evidenziati ---
    bp = axes[1].boxplot(data_all, vert=True, patch_artist=True,
                         boxprops=dict(facecolor=color, alpha=0.7),
                         medianprops=dict(color='red', linewidth=2.5),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         flierprops=dict(marker='o', markersize=3,
                                         markerfacecolor='gray',
                                         markeredgecolor='gray', alpha=0.4))
    axes[1].set_title(f"Boxplot\n(pallini grigi = outlier IQR: {n_out} pari a {pct_out:.1f}%)",
                      fontsize=10)
    axes[1].set_ylabel("Valore")
    axes[1].set_xticks([])

    # Linee percentili
    for perc, style, label in [(0.05, ':', '5° perc.'), (0.95, ':', '95° perc.')]:
        val = data_all.quantile(perc)
        axes[1].axhline(val, color='purple', linestyle=style, linewidth=1.2,
                        alpha=0.7, label=f'{label}: {val:.2f}')
    axes[1].legend(fontsize=8)

    # --- Colonna 3: Distribuzione SI vs NO sovrapposta ---
    # Calcola range comune
    x_min = data_all.quantile(0.01)
    x_max = data_all.quantile(0.99)
    bins  = np.linspace(x_min, x_max, 35)

    axes[2].hist(data_si, bins=bins, alpha=0.65, color=PALETTE[0],
                 label=f'SI approvato (n={len(data_si):,})',
                 density=True, edgecolor='white')
    axes[2].hist(data_no, bins=bins, alpha=0.65, color=PALETTE[1],
                 label=f'NO rifiutato (n={len(data_no):,})',
                 density=True, edgecolor='white')

    # Mediane SI e NO
    axes[2].axvline(data_si.median(), color=PALETTE[0], linestyle='--',
                    linewidth=2, label=f'Med SI: {data_si.median():.2f}')
    axes[2].axvline(data_no.median(), color=PALETTE[1], linestyle='--',
                    linewidth=2, label=f'Med NO: {data_no.median():.2f}')

    axes[2].set_title("SI (approvato) vs NO (rifiutato)\n"
                      "(più le curve sono separate → più la variabile è predittiva)",
                      fontsize=10)
    axes[2].set_xlabel("Valore")
    axes[2].set_ylabel("Densità (normalizzata)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fname = f"plot_dist_{col.replace(' ','_').lower()}.png"
    plt.savefig(PATH_OUT + fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✅ {col:<35} skew={sk:>7.2f}  kurt={ku:>8.2f}  outlier={pct_out:.1f}%  → {fname}")

print("\n" + "=" * 60)
print("INTERPRETAZIONE GRAFICI — COSA GUARDARE:")
print("=" * 60)
print("""
"""
  COLONNA 1 — ISTOGRAMMA:
    • Se media ≫ mediana → coda destra (skew positiva)
    • Se media ≪ mediana → coda sinistra (skew negativa)
    • Distribuzione a campana = ottima per MLE

  COLONNA 2 — BOXPLOT:
    • La scatola = interquartile range (Q1-Q3, 50% dei dati)
    • La linea rossa = mediana
    • I pallini grigi = outlier (fuori da 1.5×IQR)
    • Se il boxplot è "schiacciato in basso" = molti outlier alti

  COLONNA 3 — SI vs NO:
    ✅ Curve BEN SEPARATE → variabile predittiva (buona feature)
    ⚠️  Curve SOVRAPPOSTE → variabile poco predittiva
    Nota le mediane (linee tratteggiate): se sono lontane tra SI e NO
    → quella variabile discrimina bene il target
"""
""")
"""
# =============================================================================
# BINARY REGRESSION - CREDIT APPROVAL CASE STUDY
# Business Case: A leasing company wants to understand which companies
# should be granted credit, considering macroeconomic outlook and firm features.
#
# STEP 2 — VERSIONE FINALE PULITA
#
# STRUTTURA:
#   Step 2.1-2.10  : Modello Completo (con SCORE_SONEPAR) — Scenario A e B
#   Step 2.11      : Modello Oggettivo (senza SCORE_SONEPAR) — confronto
#   Step 2.14      : Confronto flag_fatturato vs eliminare righe
#
# DECISIONI METODOLOGICHE FINALI:
#   flag_fatturato   → NEL MODELLO (opacità finanziaria, no imputazione)
#   log_fatturato    → RIMOSSO (p=0.925 non significativo)
#   flag_itp + log_itp → ENTRAMBI NEL MODELLO (complementari)
#   log_produttivita → NON aggiunto (p=0.378 non significativo)
#   Scenario A       → anzianita_azienda_anni (p=0.032 significativa)
#   Scenario B       → anzianita_cliente_anni (p=0.219 non significativa)
#
# CAMPIONE: 5.857 righe totali — nessuna riga eliminata
# DATA LEAKAGE CORRETTO: mediane calcolate solo sul train post-split
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
PALETTE = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
PATH     = r"C:\Users\franc\Desktop\Data Analysis\DatabaseProposteFido_202602.xlsx"
PATH_OUT = r"C:\Users\franc\Desktop\Data Analysis\\"

# =============================================================================
# RICOSTRUZIONE DATASET
# =============================================================================

df = pd.read_excel(PATH)
df['target'] = (df['Esito_finale'].str.strip().str.upper() == 'SI').astype(int)

df['flag_score_nd'] = (df['SCORE_INNOLVA'] == 1000).astype(int)
df.loc[df['SCORE_INNOLVA'] == 1000, 'SCORE_INNOLVA'] = np.nan
df.loc[df['ITP'] == -999, 'ITP'] = np.nan

df['flag_fatturato'] = df['FATTURATO'].notna().astype(int)
df['flag_itp']       = df['ITP'].notna().astype(int)
df['log_itp']        = np.where(df['ITP'].notna(), np.log1p(df['ITP']), np.nan)
df['log_dipendenti'] = np.log1p(df['DIPENDENTI'])
df['log_immobili']   = np.log1p(df['NUMERO_IMMOBILI'])

def parse_date_int(s):
    return pd.to_datetime(s.astype(str).str.split('.').str[0],
                          format='%Y%m%d', errors='coerce')

df['data_calcolo_dt']        = parse_date_int(df['DATA_CALCOLO'])
df['data_iscrizione_dt']     = parse_date_int(df['DATA_ISCRIZIONE'])
df['data_inizio_dt']         = parse_date_int(df['DATA_INZIO_ATTIVITA'])
df['anzianita_azienda_anni'] = (df['data_calcolo_dt'] - df['data_inizio_dt']).dt.days / 365.25
df['anzianita_cliente_anni'] = (df['data_calcolo_dt'] - df['data_iscrizione_dt']).dt.days / 365.25

classe_order = {'AAA':9,'AA':8,'A':7,'BBB':6,'BB':5,'B':4,'CCC':3,'CC':2,'C':1,'ND':0}
df['score_innolva_ord'] = df['CLASSE_SCORE_INNOLVA'].map(classe_order)

tasso_sonepar = df.groupby('SCORE_SONEPAR')['target'].mean().sort_values()
rank_sonepar  = tasso_sonepar.rank(method='dense').astype(int)
df['score_sonepar_ord'] = df['SCORE_SONEPAR'].map(rank_sonepar)

df['azienda_attiva'] = (df['STATO_ATTIVITA'] == 'A').astype(int)

top5_natura = df['NATURA_GIURIDICA'].value_counts().nlargest(5).index.tolist()
df['natura_giuridica_clean'] = df['NATURA_GIURIDICA'].apply(
    lambda x: x if x in top5_natura else 'Altro')

# =============================================================================
# FUNZIONI COMUNI
# =============================================================================

def prepara_dataset(df, features, target_col='target'):
    df_prep = df[features + [target_col]].copy()
    df_prep = pd.get_dummies(df_prep, columns=['natura_giuridica_clean'],
                              drop_first=True, dtype=int)
    y = df_prep[target_col]
    X = df_prep.drop(columns=[target_col])
    return X, y

def split_dataset(X, y, seed=42):
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=seed, stratify=y_tv)
    return X_train, X_val, X_test, y_train, y_val, y_test

def standardizza(X_train, X_val, X_test):
    cols_std = [c for c in X_train.columns
                if X_train[c].nunique() > 2
                and c not in ['score_innolva_ord', 'score_sonepar_ord']]
    scaler = StandardScaler()
    X_tr = X_train.copy(); X_v = X_val.copy(); X_te = X_test.copy()
    X_tr[cols_std] = scaler.fit_transform(X_train[cols_std])
    X_v[cols_std]  = scaler.transform(X_val[cols_std])
    X_te[cols_std] = scaler.transform(X_test[cols_std])
    return X_tr, X_v, X_te, scaler, cols_std

def stima_logit(X_train, y_train, nome_scenario):
    X_sm      = sm.add_constant(X_train)
    risultato = sm.Logit(y_train, X_sm).fit(method='bfgs', maxiter=500, disp=False)
    print(f"\n  {'='*55}")
    print(f"  SCENARIO {nome_scenario} — Risultati Logit")
    print(f"  {'='*55}")
    print(risultato.summary2())
    return risultato

def mcfadden_r2(risultato, y_train):
    ll_full = risultato.llf
    N1 = y_train.sum(); N0 = len(y_train)-N1; N = len(y_train)
    ll_0 = N1*np.log(N1/N) + N0*np.log(N0/N)
    return 1-(ll_full/ll_0), 1-1/(1+2*(ll_full-ll_0)/N), ll_full, ll_0

def calibra_soglia(risultato, X_val, y_val, nome_scenario):
    prob_val = risultato.predict(sm.add_constant(X_val, has_constant='add'))
    p_hat = y_val.mean()
    EP0   = 1 - p_hat if p_hat > 0.5 else p_hat
    soglie = np.arange(0.20, 0.85, 0.01)
    ris_z = []
    for z in soglie:
        y_p    = (prob_val >= z).astype(int)
        EP1    = ((y_val - y_p)**2).mean()
        fr2    = 1 - EP1/EP0 if EP0 > 0 else np.nan
        tp = ((y_p==1)&(y_val==1)).sum(); fp = ((y_p==1)&(y_val==0)).sum()
        fn = ((y_p==0)&(y_val==1)).sum()
        prec   = tp/(tp+fp) if (tp+fp)>0 else 0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        f1     = 2*prec*recall/(prec+recall) if (prec+recall)>0 else 0
        ris_z.append({'z':z,'forecast_r2':fr2,'accuracy':(y_val==y_p).mean(),
                      'precision':prec,'recall':recall,'f1':f1})
    df_z = pd.DataFrame(ris_z)
    idx_best   = df_z['forecast_r2'].idxmax()
    z_ottimale = df_z.loc[idx_best, 'z']
    print(f"\n  Scenario {nome_scenario}:")
    print(f"    EP0 (bias-only)              : {EP0:.4f}")
    print(f"    z ottimale (max Forecast R²) : {z_ottimale:.2f}")
    print(f"    Forecast R² a z ottimale     : {df_z.loc[idx_best,'forecast_r2']:.4f}")
    print(f"    Forecast R² a z=0.50         : "
          f"{df_z[df_z['z'].round(2)==0.50]['forecast_r2'].values[0]:.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Scenario {nome_scenario} — Calibrazione Soglia z",
                 fontsize=12, fontweight='bold')
    axes[0].plot(df_z['z'], df_z['forecast_r2'], color=PALETTE[0], linewidth=2)
    axes[0].axvline(z_ottimale, color=PALETTE[1], linestyle='--', linewidth=2,
                    label=f'z={z_ottimale:.2f}')
    axes[0].axvline(0.50, color='gray', linestyle=':', linewidth=1.5, label='z=0.50')
    axes[0].set_title("Forecast R²"); axes[0].set_xlabel("Soglia z"); axes[0].legend()
    for m, c in [('accuracy',PALETTE[2]),('precision',PALETTE[3]),
                 ('recall',PALETTE[4]),('f1',PALETTE[1])]:
        axes[1].plot(df_z['z'], df_z[m], color=c, linewidth=2, label=m)
    axes[1].axvline(z_ottimale, color='black', linestyle='--', linewidth=1.5)
    axes[1].set_title("Metriche"); axes[1].set_xlabel("Soglia z"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    fname = f"plot_soglia_z_scenario{nome_scenario[0]}.png"
    plt.savefig(PATH_OUT + fname, dpi=150, bbox_inches='tight'); plt.show()
    print(f"    ✅ Grafico salvato: {fname}")
    return z_ottimale, df_z

def valuta_test(risultato, X_test, y_test, z_ottimale, nome_scenario, nome_var):
    prob_test = risultato.predict(sm.add_constant(X_test, has_constant='add'))
    y_pred    = (prob_test >= z_ottimale).astype(int)
    p_hat = y_test.mean(); EP0 = 1-p_hat if p_hat>0.5 else p_hat
    EP1   = ((y_test-y_pred)**2).mean(); fr2 = 1-EP1/EP0
    auc   = roc_auc_score(y_test, prob_test)
    cm    = confusion_matrix(y_test, y_pred)
    acc   = (y_test==y_pred).mean()
    tp=cm[1,1]; fp=cm[0,1]; fn=cm[1,0]; tn=cm[0,0]
    prec   = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0
    f1     = 2*prec*recall/(prec+recall) if (prec+recall)>0 else 0
    print(f"\n  SCENARIO {nome_scenario} [anzianità: {nome_var}]")
    print(f"  Soglia z : {z_ottimale:.2f}")
    print(f"  {'─'*50}")
    print(f"  AUC          : {auc:.4f}")
    print(f"  Forecast R²  : {fr2:.4f}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"  F1           : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Pred NO   Pred SI")
    print(f"  Reale NO       {tn:>6}    {fp:>6}")
    print(f"  Reale SI       {fn:>6}    {tp:>6}")
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    return {'scenario':nome_scenario,'z':z_ottimale,'auc':auc,'forecast_r2':fr2,
            'accuracy':acc,'precision':prec,'recall':recall,'f1':f1,
            'fpr':fpr,'tpr':tpr,'prob_test':prob_test,'y_pred':y_pred,'cm':cm}

def analisi_coefficienti(risultato, nome_scenario):
    print(f"\n  SCENARIO {nome_scenario}")
    print(f"  {'Variabile':<35} {'Beta':>8} {'SE':>8} {'z':>8} {'p-val':>8} {'Signif':>8}")
    print(f"  {'-'*80}")
    for var in risultato.params.index:
        p   = risultato.pvalues[var]
        sig = ('***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else '')
        print(f"  {var:<35} {risultato.params[var]:>8.4f} {risultato.bse[var]:>8.4f} "
              f"{risultato.tvalues[var]:>8.3f} {p:>8.4f} {sig:>8}")
    print(f"\n  *** p<0.01  ** p<0.05  * p<0.10")
    margeff = risultato.get_margeff()
    print(f"\n  EFFETTI MARGINALI MEDI (AME):")
    print(f"  {'Variabile':<35} {'dP/dX':>10} {'SE':>8} {'p-val':>8}")
    print(f"  {'-'*65}")
    me_df = margeff.summary_frame()
    for var in me_df.index:
        p   = me_df.loc[var, 'Pr(>|z|)']
        sig = ('***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else '')
        print(f"  {var:<35} {me_df.loc[var,'dy/dx']:>10.4f} "
              f"{me_df.loc[var,'Std. Err.']:>8.4f} {p:>8.4f}  {sig}")

# =============================================================================
# STEP 2.1 — DEFINIZIONE FEATURES
# =============================================================================

print("=" * 65)
print("STEP 2.1 — DEFINIZIONE FEATURES")
print("=" * 65)

features_comuni = [
    'score_innolva_ord', 'flag_score_nd', 'score_sonepar_ord',
    'flag_fatturato', 'flag_itp', 'log_itp',
    'VALORE FIDO RICHIESTO', 'log_dipendenti', 'log_immobili',
    'AFFIDATA', 'REVISIONE', 'azienda_attiva', 'natura_giuridica_clean',
]
features_A = features_comuni + ['anzianita_azienda_anni']
features_B = features_comuni + ['anzianita_cliente_anni']

print(f"""
  DECISIONI VARIABILI:
  ❌ log_fatturato    → RIMOSSO (p=0.925)
  ✅ flag_fatturato   → NEL MODELLO (opacità finanziaria)
  ✅ flag_itp         → NEL MODELLO (segnala ITP reale vs imputato)
  ✅ log_itp          → NEL MODELLO (p<0.001, AME=-5.6%)
  ✅ score_sonepar_ord→ Modello Completo / rimosso nel Modello Oggettivo

  Scenario A: anzianita_azienda_anni (p=0.032 nel Modello Oggettivo)
  Scenario B: anzianita_cliente_anni (p=0.219 non significativa)
  Totale features Modello Completo: {len(features_A)}
""")

# =============================================================================
# STEP 2.2-2.3 — ENCODING, SPLIT
# =============================================================================

print("=" * 65)
print("STEP 2.2 — ONE-HOT ENCODING")
print("=" * 65)

X_A, y_A = prepara_dataset(df, features_A)
X_B, y_B = prepara_dataset(df, features_B)
print(f"  Scenario A shape: {X_A.shape}  |  Scenario B shape: {X_B.shape}")

print("\n" + "=" * 65)
print("STEP 2.3 — SPLIT 60/20/20 + IMPUTAZIONE POST-SPLIT")
print("=" * 65)

X_train_A, X_val_A, X_test_A, y_train_A, y_val_A, y_test_A = split_dataset(X_A, y_A)
X_train_B, X_val_B, X_test_B, y_train_B, y_val_B, y_test_B = split_dataset(X_B, y_B)

cols_imputa = ['log_itp', 'anzianita_azienda_anni']
for X_tr, X_v, X_te, nome in [
        (X_train_A, X_val_A, X_test_A, 'A'),
        (X_train_B, X_val_B, X_test_B, 'B')]:
    print(f"  Scenario {nome}:  Train={len(X_tr):,}  Val={len(X_v):,}  Test={len(X_te):,}")
    for col in cols_imputa:
        if col in X_tr.columns:
            n_tr=X_tr[col].isna().sum(); n_v=X_v[col].isna().sum(); n_te=X_te[col].isna().sum()
            med = X_tr[col].median()
            X_tr[col]=X_tr[col].fillna(med); X_v[col]=X_v[col].fillna(med)
            X_te[col]=X_te[col].fillna(med)
            print(f"    {col}: mediana={med:.3f}  NaN→ train:{n_tr} val:{n_v} test:{n_te}")

# =============================================================================
# STEP 2.4 — STANDARDIZZAZIONE
# =============================================================================

X_tr_A, X_v_A, X_te_A, scaler_A, cols_std_A = standardizza(X_train_A, X_val_A, X_test_A)
X_tr_B, X_v_B, X_te_B, scaler_B, cols_std_B = standardizza(X_train_B, X_val_B, X_test_B)
print(f"\n  Colonne standardizzate: {cols_std_A}")

# =============================================================================
# STEP 2.5 — STIMA MODELLO LOGIT COMPLETO
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2.5 — STIMA MODELLO COMPLETO (con SCORE_SONEPAR)")
print("=" * 65)

res_A = stima_logit(X_tr_A, y_train_A, 'A — Completo (anzianita_azienda)')
res_B = stima_logit(X_tr_B, y_train_B, 'B — Completo (anzianita_cliente)')

# =============================================================================
# STEP 2.6 — MCFADDEN R²
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2.6 — GOODNESS OF FIT: McFadden R²")
print("=" * 65)

r2_mf_A, _, ll_A, ll0_A = mcfadden_r2(res_A, y_train_A)
r2_mf_B, _, ll_B, ll0_B = mcfadden_r2(res_B, y_train_B)

print(f"""
  {'Misura':<30} {'Scenario A':>15} {'Scenario B':>15}
  {'-'*60}
  {'McFadden R²':<30} {r2_mf_A:>15.4f} {r2_mf_B:>15.4f}

  📌 0.10-0.20 accettabile | 0.20-0.30 buono | >0.30 eccellente
""")

# =============================================================================
# STEP 2.7 — COEFFICIENTI E AME
# =============================================================================

print("=" * 65)
print("STEP 2.7 — COEFFICIENTI E EFFETTI MARGINALI MEDI (AME)")
print("=" * 65)

analisi_coefficienti(res_A, 'A — Completo (anzianita_azienda)')
analisi_coefficienti(res_B, 'B — Completo (anzianita_cliente)')

# =============================================================================
# STEP 2.8 — CALIBRAZIONE SOGLIA Z
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2.8 — CALIBRAZIONE SOGLIA Z SUL VALIDATION SET")
print("=" * 65)

z_A, df_z_A = calibra_soglia(res_A, X_v_A, y_val_A, 'A (anzianita_azienda)')
z_B, df_z_B = calibra_soglia(res_B, X_v_B, y_val_B, 'B (anzianita_cliente)')

# =============================================================================
# STEP 2.9 — TEST SET
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2.9 — VALUTAZIONE FINALE SUL TEST SET — MODELLO COMPLETO")
print("=" * 65)

ris_A = valuta_test(res_A, X_te_A, y_test_A, z_A, 'A', 'anzianita_azienda_anni')
ris_B = valuta_test(res_B, X_te_B, y_test_B, z_B, 'B', 'anzianita_cliente_anni')

# =============================================================================
# STEP 2.10 — CONFRONTO A vs B
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2.10 — CONFRONTO SCENARI A vs B")
print("=" * 65)

metriche = ['auc','forecast_r2','accuracy','precision','recall','f1']
print(f"\n  {'Metrica':<25} {'Scenario A':>15} {'Scenario B':>15} {'Migliore':>10}")
print(f"  {'─'*65}")
for m in metriche:
    vA=ris_A[m]; vB=ris_B[m]; best='A ✅' if vA>vB else 'B ✅'
    print(f"  {m:<25} {vA:>15.4f} {vB:>15.4f} {best:>10}")

print("""
  SCELTA: Scenario A — anzianita_azienda_anni significativa (p=0.032 **)
          Scenario B — anzianita_cliente_anni non significativa (p=0.219)
""")

# =============================================================================
# STEP 2.11 — MODELLO OGGETTIVO (senza SCORE_SONEPAR)
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2.11 — MODELLO COMPLETO vs MODELLO OGGETTIVO")
print("=" * 65)

print("""
  MOTIVAZIONE:
  SCORE_SONEPAR è uno score interno → rischio tautologia.
  Il Modello Oggettivo identifica i driver strutturali del merito
  creditizio basati su caratteristiche esterne e oggettive.
""")

features_A_obj = [f for f in features_A if f != 'score_sonepar_ord']
X_A_obj, y_A_obj = prepara_dataset(df, features_A_obj)
X_train_Ao, X_val_Ao, X_test_Ao, y_train_Ao, y_val_Ao, y_test_Ao = \
    split_dataset(X_A_obj, y_A_obj)

for col in ['log_itp', 'anzianita_azienda_anni']:
    if col in X_train_Ao.columns:
        med=X_train_Ao[col].median()
        n_tr=X_train_Ao[col].isna().sum(); n_v=X_val_Ao[col].isna().sum()
        n_te=X_test_Ao[col].isna().sum()
        X_train_Ao[col]=X_train_Ao[col].fillna(med)
        X_val_Ao[col]=X_val_Ao[col].fillna(med)
        X_test_Ao[col]=X_test_Ao[col].fillna(med)
        if n_tr > 0:
            print(f"  {col}: mediana={med:.3f}  NaN→ train:{n_tr} val:{n_v} test:{n_te}")

X_tr_Ao, X_v_Ao, X_te_Ao, scaler_Ao, _ = standardizza(X_train_Ao, X_val_Ao, X_test_Ao)

print("\n  Stima Modello Oggettivo...")
res_Ao = stima_logit(X_tr_Ao, y_train_Ao, 'Oggettivo (senza SCORE_SONEPAR)')
z_Ao, _ = calibra_soglia(res_Ao, X_v_Ao, y_val_Ao, 'Oggettivo')
ris_Ao  = valuta_test(res_Ao, X_te_Ao, y_test_Ao, z_Ao, 'Oggettivo', 'anzianita_azienda_anni')

r2_mf_Ao, _, ll_Ao, ll0_Ao = mcfadden_r2(res_Ao, y_train_Ao)
delta_auc = ris_A['auc'] - ris_Ao['auc']
delta_r2  = r2_mf_A - r2_mf_Ao

print(f"""
  CONFRONTO COMPLETO vs OGGETTIVO:
  {'Metrica':<30} {'Completo':>12} {'Oggettivo':>12}
  {'─'*55}
  {'Variabili':<30} {len(features_A):>12} {len(features_A_obj):>12}
  {'McFadden R²':<30} {r2_mf_A:>12.4f} {r2_mf_Ao:>12.4f}
  {'AUC (test)':<30} {ris_A['auc']:>12.4f} {ris_Ao['auc']:>12.4f}
  {'Accuracy':<30} {ris_A['accuracy']:>12.4f} {ris_Ao['accuracy']:>12.4f}
  {'Precision':<30} {ris_A['precision']:>12.4f} {ris_Ao['precision']:>12.4f}
  {'Recall':<30} {ris_A['recall']:>12.4f} {ris_Ao['recall']:>12.4f}
  {'F1':<30} {ris_A['f1']:>12.4f} {ris_Ao['f1']:>12.4f}

  Delta AUC        : {delta_auc:+.4f} → {'PICCOLO (<0.02) → Modello Oggettivo competitivo' if abs(delta_auc)<0.02 else 'MODERATO'}
  Delta McFadden R²: {delta_r2:+.4f}
""")

# Variabili significative Modello Oggettivo
print("  VARIABILI SIGNIFICATIVE NEL MODELLO OGGETTIVO:")
print(f"  {'Variabile':<35} {'AME':>10} {'p-val':>8} {'Signif':>8}")
print(f"  {'─'*65}")
margeff_obj = res_Ao.get_margeff().summary_frame()
for var in margeff_obj.index:
    p   = margeff_obj.loc[var, 'Pr(>|z|)']
    dy  = margeff_obj.loc[var, 'dy/dx']
    sig = ('***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else '')
    if sig:
        print(f"  {var:<35} {dy:>10.4f} {p:>8.4f}  {sig}")

# Grafico ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Modello Completo vs Oggettivo — Test Set", fontsize=12, fontweight='bold')
axes[0].plot(ris_A['fpr'], ris_A['tpr'], color=PALETTE[0], linewidth=2.5,
             label=f"Completo (AUC={ris_A['auc']:.3f})")
axes[0].plot(ris_Ao['fpr'], ris_Ao['tpr'], color=PALETTE[1], linewidth=2.5,
             label=f"Oggettivo (AUC={ris_Ao['auc']:.3f})")
axes[0].plot([0,1],[0,1],'k--',linewidth=1,label='Random')
axes[0].fill_between(ris_A['fpr'],ris_A['tpr'],alpha=0.1,color=PALETTE[0])
axes[0].fill_between(ris_Ao['fpr'],ris_Ao['tpr'],alpha=0.1,color=PALETTE[1])
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve"); axes[0].legend()

me_comp = res_A.get_margeff().summary_frame()
me_obj  = res_Ao.get_margeff().summary_frame()
vars_sig = set()
for idx in me_comp.index:
    if me_comp.loc[idx,'Pr(>|z|)']<0.10: vars_sig.add(idx)
for idx in me_obj.index:
    if me_obj.loc[idx,'Pr(>|z|)']<0.10: vars_sig.add(idx)
vars_plot = [v for v in vars_sig if v!='score_sonepar_ord' and v in me_obj.index]
if vars_plot:
    ame_comp=[me_comp.loc[v,'dy/dx'] if v in me_comp.index else 0 for v in vars_plot]
    ame_obj =[me_obj.loc[v,'dy/dx']  if v in me_obj.index  else 0 for v in vars_plot]
    x=np.arange(len(vars_plot)); w=0.35
    axes[1].bar(x-w/2,ame_comp,w,label='Completo',color=PALETTE[0],alpha=0.8)
    axes[1].bar(x+w/2,ame_obj, w,label='Oggettivo',color=PALETTE[1],alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [v.replace('natura_giuridica_clean_','nat_').replace('anzianita_azienda_anni','anzianita')
          .replace('azienda_attiva','az_attiva').replace('VALORE FIDO RICHIESTO','fido')
         for v in vars_plot], rotation=35, ha='right', fontsize=8)
    axes[1].axhline(0,color='black',linewidth=0.8)
    axes[1].set_title("AME significativi"); axes[1].legend()
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'{x:.1%}'))
plt.tight_layout()
plt.savefig(PATH_OUT+"plot_completo_vs_oggettivo.png",dpi=150,bbox_inches='tight')
plt.show()
print("  ✅ Grafico salvato: plot_completo_vs_oggettivo.png")

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │ RACCOMANDAZIONE FINALE — MODELLO DA USARE                        │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │ MODELLO SCELTO: Oggettivo — Scenario A                           │
  │   AUC = {ris_Ao['auc']:.3f}  McFadden R² = {r2_mf_Ao:.3f}                   │
  │   z ottimale (Forecast R²) = {z_Ao:.2f}                              │
  │                                                                  │
  │ MOTIVAZIONE:                                                     │
  │   Delta AUC = {delta_auc:+.4f} → piccolo → scelta metodologica ok  │
  │   Risponde al business case: driver strutturali oggettivi        │
  │   Non dipende dallo score interno proprietario                   │
  │                                                                  │
  │ MODELLO COMPLETO MANTENUTO COME RIFERIMENTO:                     │
  │   AUC = {ris_A['auc']:.3f}  McFadden R² = {r2_mf_A:.3f}                   │
  │   Per chi vuole massimizzare la capacità predittiva              │
  └──────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# STEP 2.14 — FLAG_FATTURATO vs ELIMINARE RIGHE
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2.14 — ELIMINARE vs TENERE LE RIGHE SENZA FATTURATO")
print("=" * 65)

n_totale = len(df); appr_totale = df['target'].mean()
n_con  = df['FATTURATO'].notna().sum(); appr_con  = df[df['FATTURATO'].notna()]['target'].mean()
n_sans = df['FATTURATO'].isna().sum();  appr_sans = df[df['FATTURATO'].isna()]['target'].mean()

print(f"""
  Con fatturato    : {n_con:,} ({n_con/n_totale*100:.1f}%)  approv: {appr_con:.1%}
  Senza fatturato  : {n_sans:,} ({n_sans/n_totale*100:.1f}%)  approv: {appr_sans:.1%}
  Differenza       : {(appr_con-appr_sans)*100:+.1f}pp → le righe NON sono casuali
""")

# V_BASE — Modello Oggettivo già stimato = ris_Ao
auc_b    = ris_Ao['auc']
recall_b = ris_Ao['recall']
f1_b     = ris_Ao['f1']
r2_b     = r2_mf_Ao

# V_ELIM
df['log_fatturato_v'] = np.where(df['FATTURATO'].notna() & (df['FATTURATO']>0),
                                  np.log1p(df['FATTURATO']), np.nan)
features_elim = [f for f in features_A_obj if f!='flag_fatturato'] + ['log_fatturato_v']
X_elim_raw, y_elim_raw = prepara_dataset(df, features_elim)
mask = X_elim_raw['log_fatturato_v'].notna()
X_elim = X_elim_raw[mask].copy(); y_elim = y_elim_raw[mask].copy()

print(f"  V_ELIM: {len(y_elim):,} righe  ({len(y_elim)/n_totale*100:.1f}% del totale)")
print(f"  Tasso approvazione: {y_elim.mean():.1%}  (distorto {(y_elim.mean()-appr_totale)*100:+.1f}pp)")

X_tr_e, X_val_e, X_te_e, y_tr_e, y_val_e, y_te_e = split_dataset(X_elim, y_elim)
for col in ['log_itp','anzianita_azienda_anni']:
    if col in X_tr_e.columns:
        med=X_tr_e[col].median()
        X_tr_e[col]=X_tr_e[col].fillna(med); X_val_e[col]=X_val_e[col].fillna(med)
        X_te_e[col]=X_te_e[col].fillna(med)
X_tr_es, X_val_es, X_te_es, _, _ = standardizza(X_tr_e, X_val_e, X_te_e)
res_e = sm.Logit(y_tr_e, sm.add_constant(X_tr_es)).fit(method='bfgs',maxiter=500,disp=False)
prob_ve = res_e.predict(sm.add_constant(X_val_es,has_constant='add'))
p_hat_e = y_val_e.mean(); EP0_e = 1-p_hat_e if p_hat_e>0.5 else p_hat_e
soglie_e = np.arange(0.20,0.85,0.01)
fr2_e = [1-((y_val_e-(prob_ve>=z).astype(int))**2).mean()/EP0_e for z in soglie_e]
z_e   = soglie_e[np.argmax(fr2_e)]
prob_te_e = res_e.predict(sm.add_constant(X_te_es,has_constant='add'))
y_pred_e  = (prob_te_e>=z_e).astype(int)
auc_e     = roc_auc_score(y_te_e, prob_te_e)
tp_e=((y_pred_e==1)&(y_te_e==1)).sum(); fp_e=((y_pred_e==1)&(y_te_e==0)).sum()
fn_e=((y_pred_e==0)&(y_te_e==1)).sum()
recall_e  = tp_e/(tp_e+fn_e) if (tp_e+fn_e)>0 else 0
prec_e    = tp_e/(tp_e+fp_e) if (tp_e+fp_e)>0 else 0
f1_e      = 2*prec_e*recall_e/(prec_e+recall_e) if (prec_e+recall_e)>0 else 0
ll_e=res_e.llf; N1_e=y_tr_e.sum(); N0_e=len(y_tr_e)-N1_e; N_e=len(y_tr_e)
ll0_e=N1_e*np.log(N1_e/N_e)+N0_e*np.log(N0_e/N_e); r2_e=1-(ll_e/ll0_e)
p_fatt = res_e.pvalues.get('log_fatturato_v',1.0)
b_fatt = res_e.params.get('log_fatturato_v',0.0)
sig_f  = '***' if p_fatt<0.01 else ('**' if p_fatt<0.05 else ('*' if p_fatt<0.10 else 'n.s.'))

print(f"\n  CONFRONTO:")
print(f"  {'Metrica':<18} {'V_BASE (flag)':>15} {'V_ELIM (log)':>15} {'Delta':>10}")
print(f"  {'─'*62}")
for lbl,vb,ve,is_f in [
    ('Righe training', len(X_tr_Ao), len(y_tr_e), False),
    ('AUC',  auc_b,    auc_e,    True),
    ('Recall', recall_b, recall_e, True),
    ('F1',   f1_b,    f1_e,    True)]:
    if is_f:
        d=ve-vb; m=' ✅' if d>0.001 else (' ❌' if d<-0.001 else ' ≈')
        print(f"  {lbl:<18} {vb:>15.4f} {ve:>15.4f} {d:>+9.4f}{m}")
    else:
        print(f"  {lbl:<18} {int(vb):>15,} {int(ve):>15,} {int(ve-vb):>+9,}")

print(f"  log_fatturato: β={b_fatt:+.4f}  p={p_fatt:.4f}  {sig_f}")

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │ CONCLUSIONE: TENERE flag_fatturato (V_BASE)                      │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │ Eliminare le righe senza fatturato:                              │
  │  1. Perde {n_sans:,} righe dal training (-{n_sans/n_totale*100:.0f}% del campione)        │
  │  2. Distorce il campione verso aziende trasparenti               │
  │  3. Rende il modello inapplicabile al {n_sans/n_totale*100:.1f}% del portafoglio  │
  │  4. log_fatturato rimane non significativo (p=0.925)             │
  │  5. Recall peggiore — più contratti buoni rifiutati erroneamente │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │ RIEPILOGO FINALE STEP 2                                          │
  ├──────────────────────────────────────────────────────────────────┤
  │ MODELLO: Oggettivo — Scenario A (senza SCORE_SONEPAR)            │
  │ AUC = {ris_Ao['auc']:.3f}  McFadden R² = {r2_mf_Ao:.3f}                       │
  │ z ottimale (Forecast R²) = {z_Ao:.2f}                                │
  │ Righe training: 3.513  |  Test: 1.172                            │
  │                                                                  │
  │ NEXT → STEP 3: Analisi costi asimmetrici                        │
  └──────────────────────────────────────────────────────────────────┘
""")
# =============================================================================
# BINARY REGRESSION - CREDIT APPROVAL CASE STUDY
# Business Case: A leasing company wants to understand which companies
# should be granted credit, considering macroeconomic outlook and firm features.
#
# STEP 3 — ANALISI COSTI ASIMMETRICI E SOGLIA Z COST-OTTIMALE
#
# MODELLO DI RIFERIMENTO: Modello Oggettivo — Scenario A
#   → Senza SCORE_SONEPAR (score interno)
#   → Risponde direttamente al business case: identifica i driver
#     strutturali del merito creditizio basati su caratteristiche
#     esterne e oggettive dell'azienda
#   → AUC = 0.785, McFadden R² = 0.196
#   → z ottimale (Forecast R²) = 0.49
#
# CONTESTO MACRO:
#   - Inflazione attesa Italia 12m: 1.5% (Banca d'Italia)
#   - Crescita attesa PIL 2024: +0.5% (ISTAT)
#   → Contesto moderatamente cauto → politica creditizia prudente
#
# FORMULA SOGLIA OTTIMALE:
#   z* = C(FP) / (C(FP) + C(FN))
#   C(FP) = EAD × LGD    (approvo chi non rimborsa → perdo capitale)
#   C(FN) = EAD × margine (rifiuto chi rimborsa → perdo ricavo)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
PALETTE = ['#2196F3', '#FF9800', '#F44336', '#4CAF50', '#9C27B0']
PATH     = r"C:\Users\franc\Desktop\Data Analysis\DatabaseProposteFido_202602.xlsx"
PATH_OUT = r"C:\Users\franc\Desktop\Data Analysis\\"

# =============================================================================
# STEP 3.0 — RICOSTRUZIONE MODELLO OGGETTIVO SCENARIO A
# =============================================================================

print("=" * 65)
print("STEP 3.0 — RICOSTRUZIONE MODELLO OGGETTIVO SCENARIO A")
print("=" * 65)

print("""
  Modello di riferimento: Oggettivo (senza SCORE_SONEPAR)
  Motivazione: il business case richiede di identificare i driver
  strutturali del merito creditizio basati su caratteristiche
  esterne e oggettive — non sullo score interno dell'azienda.
  z ottimale (Forecast R²): 0.49
""")

df = pd.read_excel(PATH)
df['target'] = (df['Esito_finale'].str.strip().str.upper() == 'SI').astype(int)

# Preprocessing identico allo Step 2
df['flag_score_nd'] = (df['SCORE_INNOLVA'] == 1000).astype(int)
df.loc[df['SCORE_INNOLVA'] == 1000, 'SCORE_INNOLVA'] = np.nan
df.loc[df['ITP'] == -999, 'ITP'] = np.nan
df['flag_itp']       = df['ITP'].notna().astype(int)
df['flag_fatturato'] = df['FATTURATO'].notna().astype(int)
df['log_dipendenti'] = np.log1p(df['DIPENDENTI'])
df['log_immobili']   = np.log1p(df['NUMERO_IMMOBILI'])
df['log_itp']        = np.where(df['ITP'].notna(),
                                 np.log1p(df['ITP']), np.nan)

def parse_date_int(s):
    return pd.to_datetime(s.astype(str).str.split('.').str[0],
                          format='%Y%m%d', errors='coerce')

df['data_calcolo_dt']        = parse_date_int(df['DATA_CALCOLO'])
df['data_inizio_dt']         = parse_date_int(df['DATA_INZIO_ATTIVITA'])
df['anzianita_azienda_anni'] = (df['data_calcolo_dt'] -
                                 df['data_inizio_dt']).dt.days / 365.25

classe_order = {'AAA':9,'AA':8,'A':7,'BBB':6,'BB':5,'B':4,
                'CCC':3,'CC':2,'C':1,'ND':0}
df['score_innolva_ord'] = df['CLASSE_SCORE_INNOLVA'].map(classe_order)
df['azienda_attiva']    = (df['STATO_ATTIVITA'] == 'A').astype(int)

top5 = df['NATURA_GIURIDICA'].value_counts().nlargest(5).index.tolist()
df['natura_giuridica_clean'] = df['NATURA_GIURIDICA'].apply(
    lambda x: x if x in top5 else 'Altro')

# Features Modello Oggettivo — SENZA score_sonepar_ord
# log_fatturato RIMOSSO (p=0.925 non significativo)
features_obj = [
    'score_innolva_ord',      # rating esterno Innolva
    'flag_score_nd',          # sentinella score non calcolabile
    'flag_fatturato',         # bilancio depositato o no
    'flag_itp',               # storico pagamenti disponibile o no
    'log_itp',                # tempestività pagamenti
    'VALORE FIDO RICHIESTO',  # importo richiesto
    'log_dipendenti',         # dimensione aziendale
    'log_immobili',           # garanzie reali
    'AFFIDATA',               # già affidato su altro brand
    'REVISIONE',              # cliente con storico commerciale
    'azienda_attiva',         # stato operativo
    'anzianita_azienda_anni', # solidità storica
    'natura_giuridica_clean', # forma giuridica
]

# Estrai EAD prima del get_dummies
ead_series = df['VALORE FIDO RICHIESTO'].copy()

# Prepara dataset
df_prep = df[features_obj + ['target']].copy()
df_prep = pd.get_dummies(df_prep, columns=['natura_giuridica_clean'],
                          drop_first=True, dtype=int)

y = df_prep['target']
X = df_prep.drop(columns=['target'])

# Split identico (stesso seed dello Step 2)
X_tv, X_test, y_tv, y_test, ead_tv, ead_test = train_test_split(
    X, y, ead_series,
    test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val, ead_train, ead_val = train_test_split(
    X_tv, y_tv, ead_tv,
    test_size=0.25, random_state=42, stratify=y_tv)

# Imputazione post-split — mediana SOLO sul train
for col in ['log_itp', 'anzianita_azienda_anni']:
    if col in X_train.columns:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_val[col]   = X_val[col].fillna(med)
        X_test[col]  = X_test[col].fillna(med)

# Standardizzazione — fit solo sul train
cols_std = [c for c in X_train.columns
            if X_train[c].nunique() > 2
            and c not in ['score_innolva_ord']]
scaler   = StandardScaler()
X_tr     = X_train.copy()
X_te     = X_test.copy()
X_tr[cols_std] = scaler.fit_transform(X_train[cols_std])
X_te[cols_std] = scaler.transform(X_test[cols_std])

# Stima modello
X_sm_tr   = sm.add_constant(X_tr)
res_obj   = sm.Logit(y_train, X_sm_tr).fit(
    method='bfgs', maxiter=500, disp=False)

# Probabilità sul test set
X_sm_te   = sm.add_constant(X_te, has_constant='add')
prob_test = res_obj.predict(X_sm_te)

ead_test_arr = ead_test.values
# Verifica che il test set sia identico allo Step 2
print(f"\n  VERIFICA COERENZA CON STEP 2:")
print(f"    len(y_test)      : {len(y_test):,}  ← deve essere 1.172")
print(f"    y_test.mean()    : {y_test.mean():.3f}  ← deve essere ~0.807")
assert len(y_test) == 1172, "ERRORE: test set diverso dallo Step 2!"
assert abs(y_test.mean() - 0.807) < 0.005, "ERRORE: distribuzione target diversa!"
print(f"    ✅ Test set coerente con Step 2")
print(f"  Modello Oggettivo Scenario A ricostruito ✅")
print(f"  Test set     : {len(y_test):,} osservazioni")
print(f"  EAD medio    : €{float(ead_test.mean()):,.0f}")
print(f"  EAD mediano  : €{float(ead_test.median()):,.0f}")
print(f"  z statistico : 0.49 (ottimale su Forecast R²)")

# =============================================================================
# STEP 3.1 — DEFINIZIONE SCENARI DI COSTO
# =============================================================================

print("\n" + "=" * 65)
print("STEP 3.1 — DEFINIZIONE SCENARI DI COSTO")
print("=" * 65)

print("""
  CONTESTO MACRO (dal dataset):
  ┌─────────────────────────────────────────────────────────────┐
  │  Inflazione attesa Italia 12m : 1.5%  (Banca d'Italia)     │
  │  Crescita attesa PIL 2024     : +0.5% (ISTAT)              │
  │  Tasso deterioramento imprese : 2.4-2.5% atteso 2025-26    │
  │  (Banca d'Italia, Rapporto Stabilità Finanziaria, apr.2025)│
  │  → Contesto moderatamente cauto → politica prudente         │
  └─────────────────────────────────────────────────────────────┘

  LOGICA DEI DUE TIPI DI ERRORE:

  • Falso Positivo (approvo chi non pagherà):
    Perdita = EAD × LGD
    (LGD = % del capitale non recuperata in caso di default)

  • Falso Negativo (rifiuto chi avrebbe pagato):
    Perdita = EAD × margine netto
    (margine = tasso attivo - costo del funding)

  FORMULA SOGLIA OTTIMALE:
  ┌─────────────────────────────────────────────────────────────┐
  │  z* = C(FP) / (C(FP) + C(FN))                             │
  │  • C(FP) = C(FN) → z* = 0.50  (costi uguali)             │
  │  • C(FP) >> C(FN) → z* → 1.0  (molto selettivi)          │
  │  • C(FP) << C(FN) → z* → 0.0  (approva tutto)            │
  └─────────────────────────────────────────────────────────────┘
""")

scenari = {
    'Ottimistico': {
        'lgd':       0.23,
        'margine':   0.07,
        'fonte_lgd': 'Review of Managerial Science (2021), 26.750 contratti leasing SME',
        'fonte_marg':'TEGM BdI Q3 2024 (9.75%) - funding 2.75%',
        'colore':    PALETTE[3]
    },
    'Base': {
        'lgd':       0.35,
        'margine':   0.06,
        'fonte_lgd': 'EBA IRB Benchmarking Report (2023)',
        'fonte_marg':'TEGM BdI Q3 2024 (9.75%) - funding 3.75%',
        'colore':    PALETTE[1]
    },
    'Pessimistico': {
        'lgd':       0.45,
        'margine':   0.05,
        'fonte_lgd': 'EBA/GL/2020/05, floor regolamentare vigente',
        'fonte_marg':'TEGM BdI Q3 2024 (9.75%) - funding 4.75%',
        'colore':    PALETTE[2]
    }
}

print(f"  {'Scenario':<15} {'LGD':>6} {'Margine':>9} "
      f"{'C(FP)/€10k':>12} {'C(FN)/€10k':>12} {'Ratio':>8} {'z*':>8}")
print(f"  {'─'*75}")

for nome, s in scenari.items():
    cfp_unit        = 10000 * s['lgd']
    cfn_unit        = 10000 * s['margine']
    ratio           = cfp_unit / cfn_unit
    z_star          = cfp_unit / (cfp_unit + cfn_unit)
    s['cfp_per_euro'] = s['lgd']
    s['cfn_per_euro'] = s['margine']
    s['ratio']      = ratio
    s['z_star']     = z_star
    print(f"  {nome:<15} {s['lgd']:>6.0%} {s['margine']:>9.0%} "
          f"  €{cfp_unit:>8,.0f}   €{cfn_unit:>8,.0f} "
          f"{ratio:>8.1f}x {z_star:>8.3f}")
    print(f"  {'':15}   LGD: {s['fonte_lgd']}")
    print(f"  {'':15}   Margine: {s['fonte_marg']}")
    print()

print(f"""
  📌 RISULTATO CHIAVE:
     La soglia statistica z=0.49 (Forecast R² sul Modello Oggettivo)
     è molto più bassa delle soglie cost-ottimali (0.77 - 0.90).
     Con z=0.49 la società approva contratti che, considerando
     i costi reali del leasing, andrebbero rifiutati.
""")

# =============================================================================
# STEP 3.2 — APPLICAZIONE SOGLIE AL TEST SET
# =============================================================================

print("=" * 65)
print("STEP 3.2 — APPLICAZIONE SOGLIE AL TEST SET")
print("=" * 65)

# z_stat = soglia ottimale Forecast R² del Modello Oggettivo
z_stat = 0.49
soglie_confronto = {f'z={z_stat:.2f} (Forecast R² — Step 2)': z_stat}
for nome, s in scenari.items():
    soglie_confronto[f"z={s['z_star']:.3f} ({nome})"] = s['z_star']

risultati = {}

print(f"\n  {'Soglia':<38} {'FP':>6} {'FN':>6} {'TP':>6} {'TN':>6} "
      f"{'Approv.':>8} {'Prec.':>7} {'Recall':>7}")
print(f"  {'─'*88}")

for etichetta, z in soglie_confronto.items():
    y_pred   = (prob_test >= z).astype(int)
    tp = ((y_pred==1) & (y_test==1)).sum()
    fp = ((y_pred==1) & (y_test==0)).sum()
    fn = ((y_pred==0) & (y_test==1)).sum()
    tn = ((y_pred==0) & (y_test==0)).sum()
    prec     = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall   = tp/(tp+fn) if (tp+fn) > 0 else 0
    n_approv = tp + fp
    risultati[etichetta] = {
        'z': z, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'prec': prec, 'recall': recall,
        'n_approv': n_approv, 'y_pred': y_pred
    }
    print(f"  {etichetta:<38} {fp:>6} {fn:>6} {tp:>6} {tn:>6} "
          f"{n_approv:>8} {prec:>7.1%} {recall:>7.1%}")

# =============================================================================
# STEP 3.3 — COSTO TOTALE STIMATO IN EURO
# =============================================================================

print("\n" + "=" * 65)
print("STEP 3.3 — COSTO TOTALE STIMATO IN EURO (test set)")
print("=" * 65)

print("""
  Metodologia:
  • C(FP) individuale = EAD_i × LGD_scenario
  • C(FN) individuale = EAD_i × margine_scenario
  • Costo totale = Σ C(FP)_i + Σ C(FN)_i
  • EAD = VALORE FIDO RICHIESTO effettivo di ogni contratto
""")

print(f"\n  {'':38} {'Ottimistico':>15} {'Base':>15} {'Pessimistico':>15}")
print(f"  {'─'*85}")

costi_tabella = {}

for etichetta, r in risultati.items():
    y_pred   = r['y_pred']
    y_test_a = y_test.values
    riga     = {}

    for nome_sc, s in scenari.items():
        mask_fp  = (y_pred == 1) & (y_test_a == 0)
        mask_fn  = (y_pred == 0) & (y_test_a == 1)
        costo_fp = (ead_test_arr[mask_fp] * s['cfp_per_euro']).sum()
        costo_fn = (ead_test_arr[mask_fn] * s['cfn_per_euro']).sum()
        riga[nome_sc] = {'fp': costo_fp, 'fn': costo_fn,
                         'tot': costo_fp + costo_fn}

    costi_tabella[etichetta] = riga
    vals = [riga[n]['tot'] for n in ['Ottimistico','Base','Pessimistico']]
    print(f"  {etichetta:<38} "
          f"€{vals[0]:>13,.0f} €{vals[1]:>13,.0f} €{vals[2]:>13,.0f}")

print(f"\n  SOGLIA A MINOR COSTO PER SCENARIO:")
print(f"  {'─'*60}")
for nome_sc in ['Ottimistico','Base','Pessimistico']:
    costi          = {et: costi_tabella[et][nome_sc]['tot']
                      for et in risultati}
    soglia_migliore = min(costi, key=costi.get)
    costo_min       = costi[soglia_migliore]
    print(f"  {nome_sc:<15}: {soglia_migliore}")
    print(f"             Costo totale: €{costo_min:,.0f}")

# =============================================================================
# STEP 3.4 — GRAFICI COST-SENSITIVE ANALYSIS
# =============================================================================

print("\n" + "=" * 65)
print("STEP 3.4 — GRAFICI COST-SENSITIVE ANALYSIS")
print("=" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Analisi Costi Asimmetrici — Costo Totale al variare della Soglia z\n"
             "(Modello Oggettivo — Scenario A | Test Set)",
             fontsize=12, fontweight='bold')

soglie_range = np.arange(0.20, 0.95, 0.01)

for idx, (nome_sc, s) in enumerate(scenari.items()):
    costi_z = []; costi_fp_z = []; costi_fn_z = []

    for z in soglie_range:
        y_pred_z = (prob_test >= z).astype(int)
        y_test_a = y_test.values
        mask_fp  = (y_pred_z == 1) & (y_test_a == 0)
        mask_fn  = (y_pred_z == 0) & (y_test_a == 1)
        cfp = (ead_test_arr[mask_fp] * s['cfp_per_euro']).sum()
        cfn = (ead_test_arr[mask_fn] * s['cfn_per_euro']).sum()
        costi_z.append(cfp + cfn)
        costi_fp_z.append(cfp)
        costi_fn_z.append(cfn)

    costi_z    = np.array(costi_z)
    costi_fp_z = np.array(costi_fp_z)
    costi_fn_z = np.array(costi_fn_z)
    z_min      = soglie_range[np.argmin(costi_z)]

    ax = axes[idx]
    ax.plot(soglie_range, costi_z,    color=s['colore'], linewidth=2.5,
            label='Costo totale')
    ax.plot(soglie_range, costi_fp_z, color=PALETTE[2], linewidth=1.5,
            linestyle='--', label='Costo FP')
    ax.plot(soglie_range, costi_fn_z, color=PALETTE[3], linewidth=1.5,
            linestyle='--', label='Costo FN')
    ax.axvline(z_min, color=s['colore'], linestyle='-', linewidth=2,
               label=f'z* empirico = {z_min:.3f}')
    ax.axvline(z_stat, color='gray', linestyle=':', linewidth=1.5,
               label=f'z={z_stat} (Forecast R²)')
    ax.axvline(s['z_star'], color='black', linestyle='--', linewidth=1.5,
               label=f"z* teorico = {s['z_star']:.3f}")
    ax.set_title(f"Scenario {nome_sc}\n"
                 f"(LGD={s['lgd']:.0%}, margine={s['margine']:.0%})",
                 fontweight='bold')
    ax.set_xlabel("Soglia z")
    ax.set_ylabel("Costo stimato (€)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(PATH_OUT + "plot_costo_soglia_scenari.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Grafico salvato: plot_costo_soglia_scenari.png")

# Grafico 2: confronto FP/FN/costo per soglia
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Impatto della Soglia z su FP, FN e Costo Totale",
             fontsize=12, fontweight='bold')

etichette_plot = list(risultati.keys())
x_pos = np.arange(len(etichette_plot))

for col_idx, nome_sc in enumerate(['Ottimistico','Base','Pessimistico']):
    s   = scenari[nome_sc]
    fps = [risultati[et]['fp'] for et in etichette_plot]
    fns = [risultati[et]['fn'] for et in etichette_plot]
    ax0 = axes[0, col_idx]
    b1  = ax0.bar(x_pos - 0.2, fps, 0.35,
                  label='FP', color=PALETTE[2], alpha=0.8)
    b2  = ax0.bar(x_pos + 0.2, fns, 0.35,
                  label='FN', color=PALETTE[3], alpha=0.8)
    ax0.set_title(f"{nome_sc} — FP e FN", fontweight='bold')
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels([et.split('(')[0].strip()
                         for et in etichette_plot],
                        rotation=25, ha='right', fontsize=8)
    ax0.set_ylabel("Numero errori"); ax0.legend(fontsize=8)
    for bar in list(b1) + list(b2):
        ax0.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+1,
                 str(int(bar.get_height())),
                 ha='center', fontsize=8)

    costi_tot = [costi_tabella[et][nome_sc]['tot'] for et in etichette_plot]
    costi_fp  = [costi_tabella[et][nome_sc]['fp']  for et in etichette_plot]
    costi_fn  = [costi_tabella[et][nome_sc]['fn']  for et in etichette_plot]
    ax1 = axes[1, col_idx]
    ax1.bar(x_pos, costi_fp, label='Costo FP',
            color=PALETTE[2], alpha=0.8)
    ax1.bar(x_pos, costi_fn, bottom=costi_fp,
            label='Costo FN', color=PALETTE[3], alpha=0.8)
    idx_min = np.argmin(costi_tot)
    ax1.bar(idx_min, costi_tot[idx_min], color='none',
            edgecolor='gold', linewidth=3,
            label=f'Min: €{costi_tot[idx_min]:,.0f}')
    ax1.set_title(f"{nome_sc} — Costo Totale", fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([et.split('(')[0].strip()
                         for et in etichette_plot],
                        rotation=25, ha='right', fontsize=8)
    ax1.set_ylabel("Costo stimato (€)")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
    ax1.legend(fontsize=7)

plt.tight_layout()
plt.savefig(PATH_OUT + "plot_confronto_soglie_costi.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Grafico salvato: plot_confronto_soglie_costi.png")

# =============================================================================
# STEP 3.5 — SENSITIVITY TABLE
# =============================================================================

print("\n" + "=" * 65)
print("STEP 3.5 — SENSITIVITY TABLE: z* al variare di LGD e margine")
print("=" * 65)

print("""
  Il cliente può individuare la propria cella e applicare
  la soglia corrispondente senza dover rieseguire il modello.
  z* = C(FP) / (C(FP) + C(FN)) = LGD / (LGD + margine)
""")

lgd_vals     = [0.15, 0.20, 0.23, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
margine_vals = [0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]

print(f"  {'LGD':>6}", end="")
for m in margine_vals:
    print(f"  {'marg='+f'{m:.0%}':>9}", end="")
print()
print(f"  {'─'*75}")

for lgd in lgd_vals:
    marker = " ← Leaseurope" if lgd == 0.23 else ""
    print(f"  {lgd:>5.0%}", end="")
    for m in margine_vals:
        z_opt = lgd / (lgd + m)
        print(f"  {z_opt:>9.3f}", end="")
    print(marker)

print(f"""
  📌 Come usare la tabella:
     1. Stima il tuo LGD (% del capitale non recuperata in caso di default)
     2. Stima il tuo margine netto (tasso attivo - costo del funding)
     3. Trova la cella → quella è la tua soglia z ottimale
     4. Applica al modello per classificare le nuove richieste

  CONTESTO MACRO:
     PIL +0.5%, inflazione 1.5% → scenario Base (LGD~35%, margine~6%)
     appare coerente con il contesto macroeconomico del dataset.
""")

# =============================================================================
# STEP 3.6 — RIEPILOGO E RACCOMANDAZIONI
# =============================================================================

print("=" * 65)
print("STEP 3.6 — RIEPILOGO E RACCOMANDAZIONI AL CLIENTE")
print("=" * 65)

y_test_a = y_test.values

for nome_sc, s in scenari.items():
    # Costo con z statistico (Forecast R²)
    y_stat  = (prob_test >= z_stat).astype(int)
    fp_stat = ((y_stat==1) & (y_test_a==0))
    fn_stat = ((y_stat==0) & (y_test_a==1))
    c_stat  = (ead_test_arr[fp_stat] * s['cfp_per_euro']).sum() + \
              (ead_test_arr[fn_stat] * s['cfn_per_euro']).sum()

    # Costo con z* ottimale
    z_opt  = s['z_star']
    y_opt  = (prob_test >= z_opt).astype(int)
    fp_opt = ((y_opt==1) & (y_test_a==0))
    fn_opt = ((y_opt==0) & (y_test_a==1))
    c_opt  = (ead_test_arr[fp_opt] * s['cfp_per_euro']).sum() + \
              (ead_test_arr[fn_opt] * s['cfn_per_euro']).sum()

    saving     = c_stat - c_opt
    saving_pct = saving / c_stat * 100 if c_stat > 0 else 0

    print(f"\n  Scenario {nome_sc} (z*={z_opt:.3f}):")
    print(f"    Costo con z={z_stat}        : €{c_stat:>10,.0f}")
    print(f"    Costo con z* ottimale   : €{c_opt:>10,.0f}")
    print(f"    Risparmio potenziale    : €{saving:>10,.0f} ({saving_pct:.1f}%)")

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │ MESSAGGI CHIAVE PER IL CLIENTE                                   │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │ 1. SOGLIA ATTUALE TROPPO PERMISSIVA                             │
  │    La soglia z=0.49 (calibrata su Forecast R²) approva molte    │
  │    richieste che, considerando i costi reali del leasing,        │
  │    andrebbero rifiutate. Le soglie cost-ottimali sono tra        │
  │    0.77 e 0.90 a seconda dello scenario di costo adottato.      │
  │                                                                  │
  │ 2. IL VANTAGGIO STRUTTURALE DEL LEASING                        │
  │    L'LGD del leasing (23% medio, Leaseurope 2021) è            │
  │    strutturalmente più basso del credito corporate ordinario    │
  │    (31-36%, EBA 2023). La proprietà del bene riduce             │
  │    strutturalmente la perdita in caso di default.               │
  │                                                                  │
  │ 3. IL CONTESTO MACRO SUGGERISCE PRUDENZA                       │
  │    PIL +0.5% e tassi di deterioramento imprese 2.4-2.5%         │
  │    (Banca d'Italia, apr.2025) → rischio credito non             │
  │    trascurabile. Scenario Base (z*≈0.85) appare coerente.      │
  │                                                                  │
  │ 4. DRIVER CHIAVE DEL RISCHIO (Modello Oggettivo)               │
  │    Le variabili più significative per identificare              │
  │    le aziende affidabili sono:                                  │
  │    • Rating esterno Innolva (+5.2% per classe)                 │
  │    • Revisione cliente (+18.7%)                                 │
  │    • Tempestività pagamenti ITP (-5.6%)                        │
  │    • Azienda operativa (+9.1%)                                  │
  │    • Anzianità aziendale (-1.6% — aziende anziane più          │
  │      complesse → più selettive)                                 │
  │                                                                  │
  │ 5. PERSONALIZZAZIONE                                            │
  │    I parametri LGD e margine sono stime di settore.             │
  │    Sostituirli con i valori reali della propria struttura       │
  │    contrattuale usando la Sensitivity Table (Step 3.5).         │
  └──────────────────────────────────────────────────────────────────┘
""")
