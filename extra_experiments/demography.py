import pandas as pd
import numpy as np

path='../data/gait_outcomes.csv'
bad_data_id = [144, 178, 147, 4] # these subjects has no more than 10 segments (9, 3, 9, 6);
# 4 subjects have 10, 3 has 11, a lot of subjects has 12, so we sample 10 segments for each subject

df = pd.read_csv(path)
del_index = []
# remove bad id
for i in range(0, len(bad_data_id)):
    bad_index = df.index[df.id ==bad_data_id[i]]
    del_index.extend(bad_index)
# remove the nan feature data
na_index = df.index[df.isnull().T.any()].tolist()
del_index.extend(na_index)
df = df.drop(del_index, axis=0)

subjects = np.array(df.id)
ages = np.array(df.age)
uid, uid_index = np.unique(subjects, return_index=True)

# set age labels
groups = np.zeros(len(ages))
groups[ages > 65] = 1
u_group = groups[uid_index]

demo = pd.read_csv('../data/demography.csv')
demo['Gender'].replace('F', 0, inplace=True)
demo['Gender'].replace('M', 1, inplace=True)
demo['Height'].astype(float)
demo['Gender'].astype(float)
# y-m group
ym_age_list = []
ym_sex_list = []
ym_height_list = []
ym_weight_list = []
for id in uid[np.where(u_group == 0)]:
    sub = demo.loc[demo.id == id]
    ym_sex_list.append(list(sub['Gender'])[0])
    ym_age_list.append(list(sub['age'])[0])
    ym_height_list.append(list(sub['Height'])[0])
    ym_weight_list.append(list(sub['Weight'])[0])
ym_sex_list = np.array(ym_sex_list).astype(float)
print(f'y-m group has {len(ym_weight_list)} participants')
print(f'y-m group has {len(np.where(ym_sex_list==1)[0])} man')
print(f'y-m group has {len(np.where(ym_sex_list==0)[0])} woman')
print(f'y-m group averaged age is {np.nanmean( ym_age_list)} ')
print(f'y-m group averaged height is {np.nanmean(ym_height_list)}')
print(f'y-m group averaged weight is {np.nanmean( ym_weight_list)}')

o_age_list = []
o_sex_list = []
o_height_list = []
o_weight_list = []
for id in uid[np.where(u_group == 1)]:
    sub = demo.loc[demo.id == id]
    o_sex_list.append(list(sub['Gender'])[0])
    o_age_list.append(list(sub['age'])[0])
    o_height_list.append(list(sub['Height'])[0])
    o_weight_list.append(list(sub['Weight'])[0])
o_sex_list = np.array(o_sex_list).astype(float)
print(f'o group has {len(o_weight_list)} participants')
print(f'o group has {len(np.where(o_sex_list==1)[0])} man')
print(f'o group has {len(np.where(o_sex_list==0)[0])} woman')
print(f'o group averaged age is {np.nanmean( o_age_list)} ')
print(f'o group averaged height is {np.nanmean(o_height_list)}')
print(f'o group averaged weight is {np.nanmean( o_weight_list)}')

