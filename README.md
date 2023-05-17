# oneAPI Intel® 
Intel OneAPI serves as a comprehensive solution for heterogeneous application development, offering developers the flexibility, performance, and ease of use required to take full advantage of the increasingly diverse computing landscape. It enables developers to write high-performance code that can seamlessly target multiple hardware architectures, accelerating the pace of innovation and unlocking new possibilities in the field of computing.
OneAPI offers a unified programming model that allows developers to write code that can seamlessly target various computing devices, including CPUs, GPUs, FPGAs, and other accelerators. This eliminates the need for separate code bases for different hardware architectures, simplifying the development process and reducing maintenance efforts.

## Adantages of using oneDAL and oneDNN 
Using both oneDAL (Intel Data Analytics Library) and oneDNN (Deep Neural Network Library) models together provides several advantages:

1. **Performance Optimization**: Both oneDAL and oneDNN are optimized libraries that leverage Intel processors and architectures to deliver high-performance computations. They are designed to efficiently utilize the available hardware resources, such as multi-core processors, vector instructions (e.g., Intel Advanced Vector Extensions - AVX), and specialized acceleration features (e.g., Intel Deep Learning Boost - DL Boost). By utilizing these optimizations, oneDAL and oneDNN models can achieve faster execution times and improved overall performance, enabling efficient data analytics and deep learning tasks.

2. **Versatility**: oneDAL covers a wide range of data analytics tasks, including machine learning, data preprocessing, and mathematical computations. On the other hand, oneDNN focuses on deep neural networks and provides optimized primitives for deep learning operations, such as convolutions, pooling, and activation functions. By combining the two libraries, you gain versatility in tackling both traditional data analytics tasks and deep learning tasks within a unified framework.

3. **Seamless Integration**: oneDAL and oneDNN can be seamlessly integrated with popular programming languages and frameworks. oneDAL supports integration with Python and popular data science frameworks like scikit-learn and NumPy. Similarly, oneDNN integrates well with deep learning frameworks such as TensorFlow and PyTorch. This integration allows you to leverage the capabilities of oneDAL and oneDNN while benefiting from the familiar APIs and ecosystems of these frameworks, simplifying the development process and enabling efficient utilization of the libraries' functionality.

4. **Cross-Platform Compatibility**: Both oneDAL and oneDNN are designed to be cross-platform compatible, enabling you to develop applications that can run on various operating systems, including Windows, Linux, and macOS. This flexibility ensures compatibility and ease of use across different platforms, allowing you to deploy your models in diverse environments without major modifications.

5. **Scalability**: oneDAL and oneDNN models are built with scalability in mind. They provide efficient parallel processing capabilities, allowing you to distribute computations across multiple cores or even across distributed computing environments. This scalability enables you to handle large datasets, perform efficient model training, and process high-volume data in a timely manner.

6. **Intel Hardware Optimizations**: Intel processors offer advanced hardware features that can accelerate data analytics and deep learning tasks. Both oneDAL and oneDNN take advantage of these hardware optimizations, such as AVX instructions and DL Boost. Leveraging these features can significantly boost performance, providing faster execution and reducing training and inference times.

In summary, utilizing both oneDAL and oneDNN models provides advantages in terms of performance optimization, versatility, seamless integration with popular frameworks, cross-platform compatibility, scalability, and leveraging Intel hardware optimizations. This combination empowers developers and data scientists to efficiently tackle a wide range of data analytics and deep learning tasks while harnessing the power of Intel's hardware architectures.

### About the Project: Attendance management system using face recognition

Utilizing a **CNN with oneDNN** for attendance management brings accuracy, real-time processing, efficiency, scalability, and integration capabilities to your system. It enables you to develop a robust and high-performance attendance management solution that can handle the complexities of facial recognition and streamline attendance tracking processes.

1. **Accurate Attendance Recognition**: CNNs are well-suited for image-based tasks, including face recognition. By training a CNN model with oneDNN, you can achieve accurate and reliable attendance recognition. The model can be trained on a large dataset of face images, enabling it to learn intricate facial features and patterns, resulting in robust attendance identification.

2. **Real-time Processing**: CNNs implemented with oneDNN leverage the performance optimizations and parallel processing capabilities of Intel processors. This enables real-time processing of attendance data, making it suitable for scenarios where attendance needs to be tracked and updated in real-time, such as in classrooms, workplaces, or events.

3. **High Efficiency and Speed**: oneDNN is designed to deliver high performance and efficiency in deep learning computations. It leverages Intel's advanced hardware optimizations, such as **Intel Advanced Vector Extensions (AVX)**, to accelerate computations and improve throughput. This results in faster inference times and efficient attendance management.

The model architecture is given below,

<img width="670" alt="Screenshot 2023-05-17 at 10 55 21 AM" src="https://github.com/Ragzoid/oneAPI-Intel-Face_Recognition/assets/90862154/7d98439c-a74d-4c35-8b65-15870371ce57">


This model recognizes the face and adds the attendance status of the individual to the excel sheet. A sample output of the model is shown below.

![Output2](https://github.com/Ragzoid/oneAPI-Intel-Face_Recognition/assets/90862154/e7e62aac-66fb-425c-adab-1399fad96492)



further we are currently working on incoprating emotion detection to the project and the architecture diagram of the proposed model under implementation is shared below.

<img width="414" alt="Screenshot 2023-05-17 at 11 03 14 AM" src="https://github.com/Ragzoid/oneAPI-Intel-Face_Recognition/assets/90862154/c4919265-be71-47e5-a297-6252d1134bf1">

