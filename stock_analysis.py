# coding: UTF-8
from sklearn import svm

# ファイルの読み込み
stock_data = []
stock_data_file = open("stock_price.txt", "r")
for line in stock_data_file:
    line = line.rstrip()
    stock_data.append(float(line))
stock_data_file.close()

# # データの確認
# print stock_data
count_s = len(stock_data)
# print count_s

# 株価上昇率の算出,おおよそ-1.0-1.0の範囲に収まるように調整
modified_data = []
for i in range(1, count_s):
    # 説明　(前日の株価　- 当日の株価) / 当日の株価　* 係数β　　# 係数は　-1.0 - 1.0の範囲に収まるよう20
    modified_data.append(float(stock_data[i] - stock_data[i-1])/float(stock_data[i-1]) * 20)
# print modified_data
count_m = len(modified_data)
print count_m
