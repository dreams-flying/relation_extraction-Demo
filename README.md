# relation_extraction-Demo
# 模型展示
![image](https://github.com/dreams-flying/relation_extraction-Demo/blob/master/images/demo.png)
# 所需环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.10.6</br>
Flask==1.1.2</br>
# 项目目录
├── static &emsp;&emsp;存放网页相关前端配置</br>
│ ├── js &emsp;&emsp;js文件</br>
├── templates &emsp;&emsp;存放html文件</br>
├── utils &emsp;&emsp;存放模型相关文件</br>
│ ├── bert4keras</br>
│ ├──  data &emsp;&emsp;存放schemas数据</br>
│ ├──  pretrained_model &emsp;&emsp;存放预训练模型</br>
│ ├──  save &emsp;&emsp;存放已训练好的模型</br>
│ ├── findTriple.py &emsp;&emsp;三元组预测及处理函数</br>
├── app.py&emsp;&emsp;Flask程序主入口</br>
├── README.md
# 使用说明
1.安装相关库</br>
2.切换到主目录，运行flask</br>
```
python app.py
```
3.打开浏览器，输入
```
localhost:5000/
```
# 模型训练
关系抽取模型训练参考[relation_extraction-baseline](https://github.com/dreams-flying/relation_extraction-baseline)
