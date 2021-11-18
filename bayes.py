import numpy as np
import csv
import math
from scipy.stats import norm


#处理数据得到训练和测试的特征和标签 返回四个列表
def load_file(filename):
    with open(filename) as f:
        read_csv=csv.DictReader(f)
        lableName =list(read_csv.fieldnames)
        csv_mat = []
        for row in read_csv.reader:
            csv_mat.append(row)
        # 男1女0
        for i in range(len(csv_mat)):
            if csv_mat[i][20] == 'male':
                csv_mat[i][20]=1
            else:
                csv_mat[i][20]=0
        # 将数据集打乱
        np.random.shuffle(csv_mat)

        # 用平均值填充数据缺失处 
        data_mat = np.array(csv_mat).astype(float)
        nonzero_cnt = np.count_nonzero(data_mat, axis=0)
        nonzero_sum = np.sum(data_mat, axis=0)
        average = nonzero_sum / nonzero_cnt
        for row in range(len(data_mat)):
            for col in range (0,20):
                if data_mat[row][col] == 0.0:
                    data_mat[row][col] = average[col]

        #划分训练集和测试集
        train_mat = []
        test_mat = []
        train_lables = []
        test_lables = []
        num = int(len(data_mat) / 10 * 7)
        for i in range(num):
            train_mat.append(data_mat[i][:20])
            train_lables.append(data_mat[i][20])
        for i in range(num,len(data_mat)):
            test_mat.append(data_mat[i][:20])
            test_lables.append(data_mat[i][20])
        train_lables=np.array(train_lables).astype(int)
        test_lables=np.array(test_lables).astype(int)
        return train_mat,test_mat,train_lables,test_lables

filename = "/Users/liuhaozhe/Desktop/学习/大三/大三下/机器学习/结题项目/voice.csv"
train_mat,test_mat,train_lables,test_lables = load_file(filename)
# 训练集按性别分类
male_mat=[]
female_mat=[]
for i in range(len(train_lables)):
    if train_lables[i]==1: 
        male_mat.append(train_mat[i])
    else: 
        female_mat.append(train_mat[i])

# 训练集性别先验概率
p_male = np.count_nonzero(train_lables) / len(train_lables)
p_female = 1 - p_male

# 高斯分布的均值和极差
male_avg=np.mean(male_mat,axis=0)
Female_avg=np.mean(female_mat,axis=0)
male_var=np.sqrt(np.var(male_mat,axis=0))
Female_var=np.sqrt(np.var(female_mat,axis=0))

# 取对数计算概率
male_correct = 0
Female_correct = 0
for i in range(len(test_lables)):
    male_percent = math.log2(p_male)
    female_percent = math.log2(p_female)
    for j in range(0,20):
        male_percent+=math.log2(norm.cdf(test_mat[i][j],male_avg[j],male_var[j]))
        female_percent+=math.log2(norm.cdf(test_mat[i][j],Female_avg[j],Female_var[j]))
    if male_percent > female_percent :
        sex = 1
    else :
        sex = 0
    if sex == test_lables[i]:
        if sex == 1: 
            male_correct+=1
        else: 
            Female_correct+=1
male_accuracy = male_correct/np.count_nonzero(test_lables)
female_accuracy = Female_correct/(len(test_lables)-np.count_nonzero(test_lables))
print("男声正确率：",male_accuracy,"\t男声错误率：",1-male_accuracy)
print("女声正确率：",female_accuracy,"\t女声错误率：",1-female_accuracy)
