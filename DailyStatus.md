## 2016.11.10
1. 又采集了第三批10个人的签名，并且进行了实验。使用weka for Android的RandomForest来测试，(X, Y, VX, VY), DTWMethod=1, (Min, Med, Temp)的组合得到了0.917, 0.95, 0.933的结果。FRR有提升，FAR下降明显。

## 2016.11.9
### Done
1. 添加签名持久化
### TODO
1. 再收集签名，训练

## 2016.11.2
### Done
1. 在Android上跑通

### TODO
1. 再收集签名，训练

## 2016.10.17
### Done
1. 使用weka替换java-ml
2. 使用weka的persistence方法，可以序列化到文件
3. 使用weka的RandomForest来测试，(X, Y, VX, VY), DTWMethod=1, (Min, Med, Temp)的组合得到了0.89, 1, 0.945的结果。目前最好。

## TODO
1. 收集第三批人的签名数据
2. 尝试降低FalseRejectRate

## 2016.10.11
### Done
1. 继续测试，使用(X, Y, VX), DTWMethod=1, (Min, Med, Temp)的组合得到了0.87, 0.99, 0.93的结果
2. 继续测试，使用(X, Y, VX, VY), DTWMethod=1, (Min, Med, Temp)的组合得到了0.897, 0.977, 0.937的结果。目前最好。

### TODO
1. 收集第三批人的签名数据。
2. 集成到Android计划

## 2016.10.9
### Done
1. 收集第二批10个人的伪造签名。
2. 第二批数据整理。进行训练测试。结果有所提高，但仍然不是很好。现在结果大概(80, 90)。
3. 使用DTWMethod=1, (X,Y), (Min, Med, Temp)的组合，得到了85, 95的结果。目前最好。
   使用DTWMethod=1, (Y), (Min, Med, Temp)的组合，得到了0.876, 0.955的结果。目前最好。

### TODO:
1. 收集第三批人的签名数据。

## 2016.10.8
### Done
1. 收集第二批10个人的真实签名。
2. 增加了Median的特征计算方法。

### TODO
1. 收集第二批10个人的伪造签名。
2. 第二批的数据一起整理，然后进行训练测试。观察结果。
3. 第三批10个人的数据。
