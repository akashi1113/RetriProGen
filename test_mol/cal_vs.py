import pandas as pd


# 读取CSV文件，假设没有列名
df = pd.read_csv(r'/home/aichengwei/home/wangyarong/ResGen-main/ResGen-main/test_mol/new_gen.csv', header=None)

# 过滤第三列大于-4.5的行vina_scores_allfe_retrieval_100.csv
df_filtered = df[df.iloc[:, 2] < -4.8]

# 按第二列相同的值，保留第三列最小的那一行
df_filtered = df_filtered.sort_values(by=[1, 2]).drop_duplicates(subset=[1], keep='first')
count_below_minus_7_5 = df_filtered[df_filtered.iloc[:, 2] < -7.5][2].count()
total_count = df_filtered[2].count()
percentage_below_minus_7_5 = (count_below_minus_7_5 / total_count) * 100 if total_count > 0 else 0

# 计算第三列的平均值
average_score = df_filtered[2].mean()

print("第三列的平均值:", average_score)
print("剩下的第三列中小于-7.5的占比: {:.1f}%".format(percentage_below_minus_7_5))