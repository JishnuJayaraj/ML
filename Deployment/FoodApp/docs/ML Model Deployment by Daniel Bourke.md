# Machine Learning Systems Design

[link to course](https://stanford-cs329s.github.io/syllabus.html)



## Intro to machine learning systems design



![image-20210409002621352](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409002621352.png)

![image-20210409123852066](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409123852066.png)





#### What’s machine learning systems design?

The process of defining the **interface, algorithms, data**, **infrastructure**, and **hardware** for a machine learning system to satisfy **specified requirements**.

specific requirements: reliable, scalable, maintainable, adaptable



##### Batch prediction vs. online prediction

**Batch prediction**: - 

- Generate predictions periodically. 
- Predictions are stored somewhere (e.g. SQL tables, CSV files)
- Retrieve them as needed
- Allow more complex models

**Online prediction**:

- Generate predictions as requests arrive
- Predictions are returned as responses

![image-20210409163354210](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409163354210.png)

##### Batch vs Online Prediction

![image-20210409163451532](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409163451532.png)

##### Offline vs Online learning

![image-20210409163811020](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409163811020.png)

##### ML in production: iterative deployment

![image-20210409164242400](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409164242400.png)



> Machine learning is an approach to learn complex patterns from existing data and use these patterns to make predictions on unseen data.



## Data Engineering

![image-20210409211105901](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409211105901.png)

![image-20210409211325725](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409211325725.png)

![image-20210409211558962](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409211558962.png)

![image-20210409211640536](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409211640536.png)

![image-20210409212447474](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409212447474.png)

![image-20210409212609275](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210409212609275.png)





## ML Model Deployment by Daniel Bourke



### Ingredients

- Data
- Pytorch model
- Python scripts
- Make file/ Docker file

### Utensils

- Streamlit (build and share data apps)
- Google cloud sdk
  - project,storage, AI platform, docker, container registry, app engine

### Method

- get app working locally

- deploy the model to AI platform

- deploy to app engine

  

![image-20210408182124467](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210408182124467.png)

![image-20210408182625266](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210408182625266.png)



### Deploying a model in cloud



### Outline

![image-20210408183133957](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210408183133957.png)



![image-20210408183313841](C:\Users\MODEL\AppData\Roaming\Typora\typora-user-images\image-20210408183313841.png)

### Get the app working locally



1. Create env
2. install reqirements