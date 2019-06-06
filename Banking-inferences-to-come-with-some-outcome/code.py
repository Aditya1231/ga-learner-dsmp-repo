# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
data = pd.read_csv(path)
#print(data)

data_sample = data.sample(n=sample_size,random_state = 0)

#print(data_sample)
sample_mean = data_sample['installment'].mean()

sample_std = data_sample['installment'].std()

margin_of_error = z_critical * (sample_std/math.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error,sample_mean + margin_of_error)
print('SM',sample_mean)
print('CI',confidence_interval)

# finding the true mean
true_mean = data['installment'].mean()
print('True Mean',true_mean)









# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig , axes = plt.subplots(3,1)

for i in range(len(sample_size)):
    m=[]
    for j in range(1000):
        sample = data.sample(n=sample_size[i])
        #sample_size == sample_size[i]
        m.append(sample['installment'].mean())        
    mean_series = pd.Series(m)
    axes[i] = plt.hist(mean_series)


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = (data['int.rate'].str.strip("%").astype(float))/100
#rint(data['int.rate'])


z_statistic,p_value = ztest(x1=data[data['purpose']=='small_business']['int.rate'],value = data['int.rate'].mean(),alternative='larger')

print(z_statistic,p_value)
inference = 'Reject' if p_value<0.05 else 'Accept'
print(inference)


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic ,p_value = ztest(x1 = data[data['paid.back.loan']=='No']['installment'],x2= data[data['paid.back.loan']=='Yes']['installment'] )

print(z_statistic,p_value)


inference = 'Reject' if p_value<0.05 else 'Accept'

print('We',inference,'Null Hypothesis')


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan'] == 'Yes']['purpose'].value_counts()

print(type(yes))

no = data[data['paid.back.loan'] == 'No']['purpose'].value_counts()

print(type(no))
observed = pd.concat([yes.transpose(),no.transpose()],axis=1,keys=['Yes','No'])
#print(observed)


chi2,p,dof,ex = chi2_contingency(observed)

inference = 'Reject' if chi2>critical_value else 'Accept'

print('we',inference,'The null Hypothesis')




