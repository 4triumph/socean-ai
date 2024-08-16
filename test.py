# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image
# # 定义模型
# def build_model():
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(4, activation='softmax')  # 假设有4个类别
#     ])
#     return model
#
# # 加载数据并进行预处理
# def load_data(train_dir):
#     train_datagen = ImageDataGenerator(rescale=1./255)  # 将像素值缩放到 [0, 1] 区间
#     train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode='categorical'
#     )
#     return train_generator
#
# # 构建模型
# model = build_model()
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# # 加载数据并进行预处理
# def load_data(train_dir):
#     train_datagen = ImageDataGenerator(rescale=1./255)  # 将像素值缩放到 [0, 1] 区间
#     train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode='categorical'
#     )
#     return train_generator
#
# # 加载数据
# train_dir = 'E:/inputdata/'
# train_generator = load_data(train_dir)
#
# # 训练模型
# model.fit(train_generator, epochs=10)
# model.save("model.h5")
# export_path = 'E:/fish'
# # tf.saved_model.save(model, export_path)
# model=tf.saved_model.load(export_path)
# # 定义类别标签
# class_labels = ['Abactochromis_labrosus', 'Abalistes_stellaris', 'Ablabys_taenianotus', 'Ablennes_hians']  # 根据你的实际类别进行替换
#
# # 加载待预测的图像
# img_path = "E:/fish/Ablabys_taenianotus/Ablabys_taenianotus_0057.jpg" # 替换为待预测图像的文件路径
# img = image.load_img(img_path, target_size=(64, 64))  # 加载图像并调整大小
# img_array = image.img_to_array(img)  # 将图像转换为数组
# img_array = np.expand_dims(img_array, axis=0)  # 扩展维度以匹配模型的输入尺寸
# # 获取模型的输入和输出签名
# inference = model.signatures["serving_default"]
# # print(inference)
#
# # 创建一个输入张量（input tensor），这里假设 img_array 是你的输入数据
# input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
#
# # 执行推理
# output = inference(input_tensor)
#
# # 获取推理结果
# predictions = output['output_0']  # 假设模型的输出名称为 'output'
#
# # # 进行预测
# # predictions = model(img_array)
# predicted_class_index = np.argmax(predictions[0])  # 获取预测结果中概率最高的类别索引
# predicted_class_label = class_labels[predicted_class_index]  # 获取预测类别的标签
#
# print("Predicted class:", predicted_class_label)