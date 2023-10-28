import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics as st
from scipy.stats import skew
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.read_csv("oasis_longitudinal_demographics_latest_scans.csv")  # Read the nWBV data column and convert to list
group = data['Group'].tolist()
nWBV = data['nWBV'].tolist()

non_demented = []
demented = []
converted = []

for i in range(len(nWBV)):
    if group[i] == 'Nondemented':
        non_demented.append(nWBV[i])
    if group[i] == 'Demented':
        demented.append(nWBV[i])
    if group[i] == 'Converted':
        converted.append(nWBV[i])

allGroups = [non_demented, demented, converted]

sns.histplot(data=data, x='nWBV', hue='Group', stat='probability', kde=True,
             palette='flare', multiple='layer',
             line_kws={'lw': 2})
plt.figure(figsize=(7, 11))
plt.subplot(311)
sns.histplot(data=non_demented, stat='probability', kde=True, bins=int(len(non_demented) / 5),
             color='#AF0171', facecolor='#F9CEEE', edgecolor=None,
             line_kws={'lw': 2})
plt.axvline(x=np.mean(non_demented), color='#790252', ls='solid', lw='3', label='Mean')
plt.axvline(x=np.median(non_demented), color='#FF9999', ls='dashed', lw='3', label='Median')
plt.axvline(x=st.mode(non_demented), color='#E93B81', ls='dotted', lw='3', label='Mode')
plt.legend()
plt.title('Non-demented distribution')

plt.subplot(312)
sns.histplot(demented, stat='probability', kde=True, bins=int(len(demented) / 5),
             color='#177A72', facecolor='#CEE5D0', edgecolor=None,
             line_kws={'lw': 2})
plt.axvline(x=np.mean(demented), color='#557B83', ls='solid', lw='3', label='Mean')
plt.axvline(x=np.median(demented), color='#FFD124', ls='dashed', lw='3', label='Median')
plt.axvline(x=st.mode(demented), color='#166E66', ls='dotted', lw='3', label='Mode')
plt.title('Demented distribution')
plt.legend()
plt.subplot(313)

sns.histplot(converted, stat='probability', kde=True,
             color='#3F3B6C', facecolor='#98BAE7', edgecolor=None,
             line_kws={'lw': 2})
plt.axvline(x=np.mean(converted), color='#0F3460', ls='solid', lw='3', label='Mean')
plt.axvline(x=np.median(converted), color='#150050', ls='dashed', lw='3', label='Median')
plt.axvline(x=st.mode(converted), color='#064663', ls='dotted', lw='3', label='Mode')
plt.legend()
plt.title('Converted distribution')
plt.tight_layout()
plt.show()

# Skewness
print('Skewness of Nondemented group:', skew(np.array(non_demented)))
print('Skewness of Demented group:', skew(np.array(demented)))
print('Skewness of Converted group:', skew(np.array(converted)), '\n')
print('-' * 50)

# Normality Tests:
# Kolmogorov-Smirnov
print('\nKS Test:')
print('Nondemented:', stats.kstest(non_demented, 'norm'))
print('Demented:', stats.kstest(demented, 'norm'))
print('Converted:', stats.kstest(converted, 'norm'), '\n')

# Lilliefors Test:
print('Lilliefors Test:')
ksstat = []
ltest_pvalue = []
for i in range(3):
    ksstat.append(sm.stats.diagnostic.lilliefors(allGroups[i], dist='norm')[0])
    ltest_pvalue.append(sm.stats.diagnostic.lilliefors(allGroups[i], dist='norm')[1])

print('Nondemented:', 'KS stats: %.5f, P-value: %.5f' % (ksstat[0], ltest_pvalue[0]))
print('Demented:', 'KS stats: %.5f, P-value: %.5f' % (ksstat[1], ltest_pvalue[1]))
print('Converted:', 'KS stats: %.5f, P-value: %.5f' % (ksstat[2], ltest_pvalue[2]), '\n')

# Jarque-Bera Test:
print('Jarque-Bera Test:')
print('Nondemented:', stats.jarque_bera(non_demented))
print('Demented:', stats.jarque_bera(demented))
print('Converted:', stats.jarque_bera(converted), '\n')

# Anderson-Darling Test:
print('Anderson-Darling Test:')
print('Nondemented:\n', stats.anderson(non_demented))
print('Demented:\n', stats.anderson(demented))
print('Converted:\n', stats.anderson(converted), '\n')
print('-' * 50)

# Null Hypothesis: The average of nWBV is equal between each two groups (a.k.a having dementia is not related to
# the brain volume)
# Z-test
print('\nZ-test:')
# Test 1: H0: Mean of Demented (m1) and Non-demented (m2) patients are equal. H1: m1<m2. alpha: 0.05
tstat, pvalue = ztest(demented, non_demented, value=0, alternative='smaller')
print('Demented vs Non-demented:\ntstat: %.3f, P-value: %f' % (tstat, pvalue))
if pvalue < 0.05:
    print('H0 is rejected. The average of whole brain volume in patients diagnosed with alzheimer\'s is less than '
          'non-demented patients.')
else:
    print(
        'H0 is accepted. The average of whole brain volume in patients diagnosed with alzheimer\'s '
        'is equal to non-demented patients.')

# Test 2: H0: Mean of Demented (m1) and Converted (m2) patients are equal. H1: m1<m2. alpha: 0.05
tstat, pvalue = ztest(demented, converted, value=0, alternative='smaller')
print('\nDemented vs Converted:\ntstat: %.3f, P-value: %f' % (tstat, pvalue))
if pvalue < 0.05:
    print('H0 is rejected. The average of whole brain volume in patients diagnosed with alzheimer\'s is less than '
          'converted patients.')
else:
    print(
        'H0 is accepted. The average of whole brain volume in patients diagnosed with alzheimer\'s '
        'is equal to converted patients.')

# Test 3: H0: Mean of Non-demented (m1) and Converted (m2) patients are equal. H1: m1<m2. alpha: 0.05
tstat, pvalue = ztest(non_demented, converted, value=0, alternative='larger')
print('\nNon-demented vs Converted:\ntstat: %.3f, P-value: %f' % (tstat, pvalue))
if pvalue < 0.05:
    print('H0 is rejected. The average of whole brain volume in non-demented patients is less than '
          'converted patients.\n')
else:
    print(
        'H0 is accepted. The average of whole brain volume in non-demented patients is equal to converted patients.\n')

print('-' * 50)
# t-test
print('\nt-test:')
# Test 1: H0: m1=m2. H1: m1<m2. alpha: 0.05
statistic, pvalue = stats.ttest_ind(demented, non_demented, alternative='less')
print('Demented vs Non-demented:\ntstat: %.3f, P-value: %f' % (statistic, pvalue))
if pvalue < 0.05:
    print('H0 is rejected. The average of whole brain volume in patients diagnosed with alzheimer\'s is less than '
          'non-demented patients.')
else:
    print(
        'H0 is accepted. The average of whole brain volume in patients diagnosed with alzheimer\'s '
        'is equal to non-demented patients.')
# Test 2
statistic, pvalue = stats.ttest_ind(demented, converted, alternative='less')
print('\nDemented vs Converted:\ntstat: %.3f, P-value: %f' % (statistic, pvalue))
if pvalue < 0.05:
    print('H0 is rejected. The average of whole brain volume in patients diagnosed with alzheimer\'s is less than '
          'converted patients.')
else:
    print(
        'H0 is accepted. The average of whole brain volume in patients diagnosed with alzheimer\'s '
        'is equal to converted patients.')

# Test 3
statistic, pvalue = stats.ttest_ind(non_demented, converted, alternative='greater')
print('\nNon-demented vs Converted:\ntstat: %.3f, P-value: %f' % (statistic, pvalue))
if pvalue < 0.05:
    print('H0 is rejected. The average of whole brain volume in non-demented patients is less than '
          'converted patients.\n')
else:
    print(
        'H0 is accepted. The average of whole brain volume in non-demented patients is equal to converted patients.\n')
print('-' * 50)
# ANOVA Test:
sm.qqplot(np.array(non_demented), line='45', fit=True)
plt.title('Non-demented')
sm.qqplot(np.array(demented), line='45', fit=True)
plt.title('Demented')
sm.qqplot(np.array(converted), line='45', fit=True)
plt.title('Converted')

print('\nANOVA Test:')
fvalue, pvalue = stats.f_oneway(non_demented, demented, converted)
print(f'F-value: {fvalue},P-value: {pvalue}')

print('\nANOVA Table:')
model = ols('nWBV ~ Group', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=1)
print('\n', anova_table)

# Shapiro-Wilk test for testing if the residuals are normally distributed
w, pvalue = stats.shapiro(model.resid)
print('\nShapiro-Wilk test: ', w, pvalue)

# Bartlett test to check the homogenity of variances
w, pvalue = stats.bartlett(non_demented, demented, converted)
print('\nBartlett test: ', w, pvalue)

# Tukey test for testing statistical significance
tukey = pairwise_tukeyhsd(endog=nWBV, groups=group, alpha=0.05)
tukey.plot_simultaneous()
plt.vlines(x=np.mean(nWBV), ymin=-5, ymax=5, color="red")
plt.show()
print('\n', tukey.summary())
