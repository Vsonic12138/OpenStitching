使用Opencv-4.10.0的C++接口内置的Stitching库进行的图像拼接实验

# 一、使用方法
## 1、修改工程的 “包含目录” 和 “库目录”
使用Visual Studio 2022打开工程，右击项目，在 配置属性 -> VC++目录 当中修改 “包含目录” 和 “库目录” 为项目在你电脑上的本工程的include和lib文件夹

## 2、修改工程编译版本
需要将Visual Studio的编译版本设置为“release”模式，否则会出现链接错误。

## 3、编译运行
编译运行，即可看到检测结果。输出在imgs->Barcodes文件夹下。

## 4、效果展示
**原始图像**
![1](https://github.com/user-attachments/assets/12036e34-91dc-4c9f-b38c-c06408571b3d)

![2](https://github.com/user-attachments/assets/52593a00-8d74-42da-8bf7-15715ad80a0e)

![3](https://github.com/user-attachments/assets/289a722e-c2d4-4ec1-a3c1-4b0827d54fe4)

![4](https://github.com/user-attachments/assets/3b3059df-826b-4d24-88af-d79e07152c59)

![5](https://github.com/user-attachments/assets/6cd667d2-d95a-4074-9bad-41c981fb6b64)


----------
**拼接图像结果：**
![stitched_output](https://github.com/user-attachments/assets/1c08693d-372c-435d-9a7d-93e1c2ba853e)
