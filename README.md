# co-gnss-rtk
This project aimed at completing the internship of Sadel. The title is 'Enhancement Algorithms for Railway Localization based on GNSS with RTK'
Internship Final Report

Corso di laurea di appartenenza: Laurea Magistrale di Informatica 8028
Nome, Cognome del tirocinante: Zhiguang Li
Matricola del tirocinante (per gli studenti):  0001029938
Inizio: 20 Giugno 2023
Fine: XX XX 2023
Nome della struttura presso cui si è svolta l’attività (azienda o Dipartimento): Sadel S.p.A.
Nome del Tutor Università di Bologna: Prof. Alessandro Vanelli-Coralli
Nome del Tutor operativo Università di Bologna: Dr.ssa Carla Amatetti
Nome del Supporto operativo Università di Bologna: Dr. Riccardo Campana
Nome del tutor aziendale (per i tirocini in azienda): Ing. Davide Amato

Oggetto del Tirocinio: Enhancement Algorithms for Railway Localization based on GNSS with RTK
Descrizione Progetto:
Global navigation satellite systems (GNSSs) enable receivers with the capability of determining their position, velocity, and time (PVT) using measurements based on propagation delay of transmitted radio carrier signals. A receiver can determine navigation coordinates in either point positioning or differential positioning configuration. Currently, differential positioning with observed carrier phases provides the highest accuracy for user coordinates. The carrier-phase observations are, however, ambiguous by an unknown integer number of cycles initially. Sub-centimeter level of positioning can be achieved only when this ambiguity is resolved. The process of resolving the unknown cycle ambiguities of carrier phase data as integers, termed as ambiguity resolution, is the key to highly accurate positioning. Historically, differential positioning was only possible by data post-processing.
Nowadays, real-time data transfer for short-to medium baselines is possible, which enables real-time computation of dynamic user coordinates, leading to real-time kinematic (RTK) positioning. A frequent source of error during RTK is the temporary loss of GNSS signals, owing to several factors such as signal blockage due to obstacles and vehicle dynamics.
Obiettivi:
Evaluation of the accuracy of the GNSS positioning measurements variance
Implementation of the chosen algorithms by simulation (Python)

La struttura ospitante (verifica Davide)
The internship was carried out at Sadel.S.p.A., located in Via M. Serenari, 1 40013 Castel Maggiore (BO), under the supervision of the academic tutors Prof. Alessandro Vanelli-Coralli, Dr. Carla Amatetti, Dr. Riccardo Campana and the corporate tutors Eng. Davide Amato PHD, in a mixed mode of on-site and remote work.
This internship is supported by SAF-la Scuola di Alta Formazione. Promoted by the Department of Electrical Energy and Information Engineering "Guglielmo Marconi" together with SADEL S.p.A., Alstom Ferroviaria S.p.A., Ferrovie dello Stato Italiane S.p.A., SITE S.p.A., TPER S.p.A., Systems Engineering "Seasonal" Higher School of Education (Season School) Integrated Mobility.At the end of the school, students will be able to get in touch with partner companies and have the possibility of organizing ad hoc sessions for writing a dissertation or a postgraduate internship.
The aim of the internship is to research and develop a novel sensor fusion algorithm, to integrate the collected IMU data with GNSS and RTK, and to use the algorithm to solve the measurement errors caused by various factors in the case of combining GNSS and RTK. This internship will carry out a preliminary classification of the problems in the data, use machine learning and neural network methods to classify the causes of the dynamic test results using GNSS and RTK sensor fusion on the railway lines of Formigine and Modena.
Before starting the internship, you will have to study Rail Transit Management System (ERTMS), European Train Control System (ETCS), Global Navigation Satellite System (GNSS), Inertial Measurement Unit (IMU), Real Time Kinematic Positioning (RTK), Machine Learning (ML), Supervised Learning (SL), Neural Network (NN), Python and Matlab programming languages, as well as the use of U-center, Google Earth.
Internship by analyzing the data results of the dynamic test using GNSS and RTK sensor fusion on the railway lines of Formigine and Modena, comparing the number of satellites, accuracy, latitude and longitude position information of U-center, RTK correction information and road environment conditions of Google Earth, to classify the errors of accuracy in the data. Through the method of artificial intelligence, the system can automatically identify and classify. The classification results of the current research are the following three situations: a. Clear view; b. Multipath correction; c. Tunel - the situation where the train is running in a tunnel or underground.
Il contesto  (verifica Davide)
Based on the research of the paper "Towards the Future Generation of Railway Localization Exploiting RTK and GNSS", we have come to the following conclusions after completing the performance tests of GNSS and RTK under static and dynamic conditions:
In static conditions, centimeter-level accuracy can be achieved using RTK if there is a clear view of the sky, while in urban environments multipath has a significant impact on positioning accuracy, limiting or even degrading the achievable accuracy. In the static test of the metal plate with signal shielding, under the premise of keeping the number of GNSS satellite signals during reception constant, we verified that the positioning accuracy decreases when the view of the sky is blocked. The floating range of accuracy is described in detail in the paper.
Under the conditions of the dynamic test, the researchers conducted a dynamic test on the railway line between Formigine and Modena in Emilia-Romagna, Italy, using a custom sensor node to collect GNSS, RTK and IMU measurement data in different environments. At the same time, the horizontal accuracy and satellite coverage of the GNSS data with RTK will be evaluated. In an environment without long bridges and tunnels and with a clear view of the sky, the accuracy of the RTK and GNSS modules is on average within 16 cm. However, with long bridges and tunnels, the number of GNSS satellites changes. Especially in tunnels, the number of available GNSS satellites is close to 0-1.
Using the U-centre to analyze the data set of this dynamic test, when GNSS is not available or the angle of accuracy is poor, the additional RTK sensor brings some help for accurate positioning. At the same time, there are limitations and the latter can even reduce the positioning accuracy compared to GNSS alone in unfavorable environmental conditions.
Therefore, the development of an algorithm that integrates sensors and navigation and positioning systems and combines collected IMU data, RTK data and GNSS data will become the basis for future work.
The programme of this study is a preliminary fusion algorithm, which aims to combine the test data, including the number of GNSS satellites and positioning accuracy, by using neural network and supervised learning algorithms, and classify the data according to the situation. It can be divided into three situations: clear sky, multi-path GNSS positioning and long bridge or tunnel.

L’attività svolta
During the meeting and continuous communication with Eng. Davide Amato PHD, Dr.ssa Carla Amatetti, Dr. Riccardo Campana, we have defined the internship plan, the report is as follows：
Reading and understanding the paper "Towards the Future Generation of Railway Localization Exploiting RTK and GNSS";(+author)
Consultation of information to learn: GNSS, RTK, IMU, ACCURACY, PRECISION and GNSS receivers and to master the working principle of the Real Time Navigation and Positioning system; and answering the test questions given by the tutor Davide Amato.
We used U-center and Google Earth to analyze the different scenarios and reasons for GNSS ERRORS; further study the positioning accuracy, frequency and quantity of different GNSS navigation systems, as well as the algorithm and process deduction of triangulation positioning; the working principle of RTK, and analysis of the problems with different accuracy of different RTK equipment, analysis of the situation of RTK generating error data; Learning the measurement error of IMU/INS in the railway system and analysis of the cause of the error; the principle and the principle of the error generated by the Omberay sensor in the auxiliary navigation and positioning Scenario analysis; and completing the in-depth test questions given by the tutor Davide Amato.
Learn supervised learning, neural networks, algorithmic models and examples of GNSS with RTK algorithms in Matlab.
Determine the requirements for the implementation of the algorithm. With the design of a neural network algorithm, we can identify when GNSS cannot work properly due to the decrease of the GNSS accuracy. It is then possible to identify thresholds to have different conditions: when the position reading is higher than a certain threshold.
The general steps to realize the algorithm: 
The starting datasets have been created in U-Center (version 22.07) with .UBX data format. The data could be analyzed also in U-Center;
The datasets are exported in different formats to have the possibility to process them. The first format is .CSV to have the possibility to be imported in Python and viewed also in Excel. The second format is .KML to have the possibility to view the data with Google Earth to check the specific situation (gallery, tunnel, below a bridge, below highway, near high buildings, near trees, open-field conditions, …)
Library in Python to import data from datasets;
Combine the input data with labels; 
Train the neural network; 
Find a way to optimize the error; 
Problems of the current algorithm related to what situation.
It took us about 2.5 weeks to complete 1, 2 and 3, during which I met with Ing. Davide Amato PHD 2-3 times a week at SADEL. In the middle of the second week, We had another online meeting with Dr.ssa Calra Amatetti, mainly to determine two things: first, the proposal for the algorithm part of the project, and second, the time and details of the next face-to-face meeting. On the Monday of the third week we met Dr Carla Amatetti and Dr Riccardo Campagna face to face to determine the requirements of the algorithm and the details of the algorithm, and to determine the content and requirements of the presentation for the next face to face meeting.
Later I spent more time on learning and implementing the algorithms.
GNSS
GNSS (Global Navigation Satellite System) It is a technology system that utilizes satellite signals to provide global positioning and navigation services. It can achieve accurate positioning and navigation for location on Earth by involving satellite constellations and receiver devices working together. 
GNSS constellations list:
GPS (Global Positioning System) is developed and operated by the US government. (1970-now), has 24 satellites.
BeiDou Compass System.  is developed and operated by China. (2000-2018), has 35 satellites.
GLONASS is developed and operated by Russia. (1970-now) , has 24 satellites.
Galileo is developed and operated by ESA. Europe (2000-2020) , has 22 satellites.
The accuracy of GNSS:
In fact, when we want to discuss the accuracy of GNSS, we need to consider it follow these elements:
The performance of the device receiver or antenna. Different receivers have different performance, and can use different bands; they have different capabilities for the number of GNSS satellites that can be received. So accuracy will vary.
Multi constellations and Multi bands. Some devices can support multi bands, it means that these devices  can receive from different satellites constellations. For example, ZED-F9P series u-blox F9 high precision GNSS modules can get many receiver types and its position accuracy is 0.01m + 1 ppm CEP. But the LEA-M8F module u-blox M8 GNSS time & frequency reference module can get not too many receiver types and its position accuracy is 2.5m - 4.0m CEP.

the method of messuare. Such as single-point positioning or use differential positionings or use other senior of correction positions (like GNSS with RTK)	
What does CEP stand for?
CEP named Circular Error Probable，represents the positioning accuracy of a positioning technology or device.
CEP → 50% of the samples are inside the CEP radius value.
Example: S1:0,1m S2:0,5m S3:2,5m S4:4.0m. CEP=0,5m 
Accuracy for GPS, DGPS, GPS with RTK-Correction
DGPS stands for differential GPS, and represents use of ground reference situations to send differential correction data to correct data from GPS.
GPS with RTK-Correction stands use real time kinematic technico to correct the data from GPS.	
The CEP accuracy of standard GPS positioning  in general is 5-10 meters.
The CEP accuracy of standard DGPS positioning  in general is 0.5-3 meters.
The CEP accuracy of GPS with RTK is 1-5 centimeters. 
The way to improve the accuracy of GNSS localization
the smaller Geometric Dilution of Precision (DoP) value and the higher accuracy
receiver performance: the better performance of the chip, and the higher accuracy.
the method of messuare: combine other sensors like RTK, IMU.
RTK
Real-time kinematic positioning (RTK) is the application of surveying to correct for common errors in current satellite navigation (GNSS) systems. It uses measurements of the phase of the signal's carrier wave in addition to the information content of the signal and relies on a single reference station or interpolated virtual station to provide real-time corrections, providing up to centimeter-level accuracy.
RTK uses a fixed base station and a rover to reduce the rover's position error. The base station transmits correction data to the rover.

What is the improvement with the RTK correction method to the standard GNSS measure?
Improved positioning accuracy.Standard GNSS positioning accuracy is generally 0.5m - 10m. (Depending which satellite constellations and which device we used. Standard GPS generally is 5-10 meters, DGPS generally is 0.5 - 3 meters. When we use RTK positioning accuracy is centimeter.)
High real-time performance. RTK can obtain high-precision positioning results in real time.
Enhanced anti-interference capability. RTK technology has a strong ability to correct atmospheric effects, satellite clock errors.
Low cost. GNSS receivers can get position centimeters and need at least 5 satellites. And the cost of the device is more expensive. We can use RTK through software and algorithms to improve accuracy.
Continuous measurement possible. RTK can use the moving environment, even if the atmospheric effects signal of GNSS, and RTK still run normally.
IMU / INS
An inertial measurement unit (IMU) is an electronic device that measures and reports a body's specific force, angular rate, and sometimes the orientation of the body, using a combination of accelerometers, gyroscopes, and sometimes magnetometers. In a navigation system, the data reported by the IMU is fed into a processor which calculates altitude, velocity and position.

An inertial navigation system (INS) is a navigation device that uses motion sensors (accelerometers), rotation sensors (gyroscopes) and a computer to continuously calculate by dead reckoning the position, the orientation, and the velocity (direction and speed of movement) of a moving object without the need for external references.INSs contain Inertial Measurement Units (IMUs) which have angular and linear accelerometers (for changes in position); some IMUs include a gyroscopic element (for maintaining an absolute angular reference).

What can be observed by IMU? altitude or position?
IMU itself cannot directly measure absolute altitude or position. It can only measure: Linear acceleration and angular velocity. While IMU itself cannot directly measure absolute altitude or position, it can provide an estimate of the relative change by repeatedly integrating acceleration and angular velocity measurements. 
Which is the relation between acceleration and velocity (speed)?
Acceleration and velocity: Velocity is the accumulation of acceleration. This means acceleration is the derivative of velocity. So the relationship between acceleration (a) and velocity (v) is:  v = ∫ a dt 
Integrating acceleration gives velocity. This also indicates that by measuring the acceleration of an object, we can obtain the change of its velocity.
Which is the relation between speed and position?
Speed and position: Position is the accumulation of speed. This means speed is the derivative of position. So the relationship between speed (v) and position (s) is: s = ∫ v dt 
Integrating speed gives position. This also indicates that by measuring the speed of an object, we can obtain the change of its position.
Odometry
Odometry is the use of data from motion sensors to estimate change in position over time. It is used in robotics by some legged or wheeled robots to estimate their position relative to a starting location. This method is sensitive to errors due to the integration of velocity measurements over time to give position estimates. Rapid and accurate data collection, instrument calibration, and processing are required in most cases for odometry to be used effectively.

Positioning sensors on train

Accuracy
Level of position accuracy with different measures (GNSS, IMU/INS, Odometry) and their problems?

Measures
position accuracy
Problems
GNSS (RTK with GNSS )
0.01m - 10m
1. Poor satellite 	
2. Signal obstructions and multipath interference
IMU 
1m - 100m / hour
1. Sensor signal noise 
2. Mechanical imperfections 
INS
0.1m - 1m
1. The issue of accumulated errors 
2. Sensitivity to technical requirements
Odometry
0.01% - 5%
1. Wheel slippage and uneven rolling    
2. Calibration errors


When will there be poor satellites?
There are a few situations where GPS satellites may provide poor coverage:
High latitudes
Urban canyons 
Signal blockage: Objects such as tall buildings, trees and tunnels can block satellite signals, resulting in fewer satellites available.
Poor satellites on the railway, there are some situations like trees and tunnels and under the bridge.
What is signal obstruction?
Signal obstruction refers to when an obstacle blocks or interferes with a signal. 
Some common causes of signal obstruction include:
Buildings and other structures - They can block line of sight between a transmitter and receiver.
Vegetation - Trees, bushes, crops, etc. can absorb radio signals especially at lower frequencies.
Topography - Hills and mountains can obstruct signals.
Atmospheric Effects - Fog, clouds, precipitation can attenuate optical and radio signals.
In this paper, the static test is performed using a metal plate as a signal obstacle.
What is multipath interference?
In radio communication, multipath is the propagation phenomenon that results in radio signals reaching the receiving antenna by two or more paths. Causes of multipath include atmospheric ducting, ionospheric reflection and refraction, and reflection from water bodies and terrestrial objects such as mountains and buildings. When the same signal is received over more than one path, it can create interference and phase shifting of the signal. Destructive interference causes fading; this may cause a radio signal to become too weak in certain areas to be received adequately. For this reason, this effect is also known as multipath interference.
In this picture, the position of the left car will reflect two or three radios to sensors, so it will be an accuracy error.  

Why does IMU accuracy increase over time?
Heading error accumulation
The IMU accumulates heading errors over time. But correcting with other sensors can reduce the heading error and increase accuracy.
Temperature stabilization
The IMU needs time to stabilize in temperature after starting up. When temperature is stable, the performance of components becomes stable, leading to higher accuracy.
Explain why there are odometry problems?
When the locomotive brakes, the wheels slip, or the locomotive derails, resulting in an incorrect steering angle.
The linear and angular velocity measurements are disturbed by the effect of the contour in different directions, such as when changing lanes, or by the effect of the gradient and flatness of the track when going up and down slopes.
The wheel has been used for a long time, resulting in wear of the wheel, changes in the contour radius and errors in the displacement calculated from the steering angle and linear velocity.
GNSS Receiver
A GNSS receiver is a device that receives and tracks signals from Global Navigation Satellite Systems (GNSS) like GPS, GLONASS, Galileo, and BeiDou. It then computes the receiver's location and time information.
In these pictures, we give some different GNSS Receiver:

Latitude and longitude and distance with two points
What  is the resolution of the latitude? What  is the resolution of the longitude? 

The radius of the semi-major axis of the Earth at the equator is 6,378,137.0 meters (20,925,646.3 ft) resulting in a circumference of 40,075,016.7 meters (131,479,714 ft). The equator is divided into 360 degrees of longitude, so each degree at the equator represents 111,319.5 meters (365,221 ft). As one moves away from the equator towards a pole, however, one degree of longitude is multiplied by the cosine of the latitude, decreasing the distance, approaching zero at the pole. The number of decimal places required for a particular precision at the equator is:


A value in decimal degrees to a precision of 4 decimal places is precise to 11.1 meters (36 ft) at the equator. A value in decimal degrees to 5 decimal places is precise to 1.11 meters (3 ft 8 in) at the equator. Elevation also introduces a small error: at 6,378 meters (20,925 ft) elevation, the radius and surface distance is increased by 0.001 or 0.1%. Because the earth is not flat, the precision of the longitude part of the coordinates increases the further from the equator you get. The precision of the latitude part does not increase so much, more strictly however, a meridian arc length per 1 second depends on the latitude at the point in question. The discrepancy of 1 second meridian arc length between equator and pole is about 0.3 meters (1 ft 0 in) because the earth is an oblate spheroid.  From: https://en.wikipedia.org/wiki/Decimal_degrees 
At which value in mm corresponds the variation of the LSN (least significant number) of the latitude? 
latitude from 44.59371430 to 44.59371431, the distance is 1.11mm.
latitude from 44.59371 to 44.59372, the distance is 1.11m.
At which value in mm corresponds the variation of the LSN (least significant number) of the longitude?  
Longitude: from 10,8566013 to  10,8566012, the distance of E/W  is 10.2 mm, the distance of W/E  is 4.35 mm 
Longitude: from 10,8566 to  10,856, the distance of E/W  is 1.02 m, the distance of W/E  is 0.435 m 
SOME PARAMETERS IN THE DATASET(U-centre)

Longitude - The east-west position in degrees (-180 to 180)    
Latitude - The north-south position in degrees (-90 to 90)
Altitude [msl] - The height above mean sea level in meters       
Altitude ellipsoidal - the elevation above a mathematical model that approximates the shape of the earth.
TTFF - Time to First Fix - The time taken to get the initial position fix  
Fix Mode - The type of position fix. RTK Fixed or No Fixed, FLOAT-RTK FLOAT
3D Acc. - The 3D position accuracy in meters 
2D Acc. - The 2D position (horizontal) accuracy in meters
PDOP - Position Dilution of Precision - A measure of satellite geometry  
HDOP - Horizontal Dilution of Precision   
Satellites - The number of satellites used in the position fix


Supervised Learning
What is Supervised Learning
Supervised learning is a machine learning approach where an algorithm learns from labeled training data to make predictions or take actions based on input data. In this learning paradigm, the training data consists of input-output pairs, where the inputs are the features or attributes of the data, and the outputs are the corresponding labels or target values. The goal of supervised learning is to learn a mapping function that can accurately predict the output for new, unseen input data.
How to use Supervised Learning 
To use supervised learning, you typically follow these steps:
Data Collection: Gather a labeled dataset where you have both the input features and their corresponding correct outputs or labels.
Data Preprocessing: Clean and preprocess the data to handle missing values, outliers, or any other data quality issues. This step may involve tasks such as data normalization, feature scaling, or feature selection.
Model Selection: Choose a suitable supervised learning algorithm or model that best fits your problem and data characteristics. There are various algorithms available, including linear regression, logistic regression, support vector machines (SVMs), decision trees, random forests, and neural networks.
Training: Feed the labeled training data into the selected model and let it learn the underlying patterns and relationships between the input features and the output labels. The model adjusts its internal parameters based on the training data to minimize the prediction errors.
Evaluation: Assess the performance of the trained model using evaluation metrics such as accuracy, precision, recall, F1-score, or mean squared error (MSE) depending on the problem type (classification or regression).
Prediction: Once the model is trained and evaluated, you can use it to make predictions or classify new, unseen data by feeding the input features to the trained model, which then produces the predicted output.
What problems it solved
Supervised learning is applicable to a wide range of problems, including but not limited to:
Classification: Predicting discrete, categorical labels. For example, email spam detection, image recognition, sentiment analysis, or fraud detection.
Regression: Predicting continuous numerical values. For example, predicting house prices based on features like location, size, and number of rooms, or predicting stock prices based on historical data.
Recommendation Systems: Generating personalized recommendations for users based on their past behavior or preferences.
Natural Language Processing: Analyzing and understanding textual data, such as sentiment analysis, text classification, or machine translation.
Supervised learning algorithms have been successful in solving various real-world problems by leveraging the labeled data to learn patterns and make accurate predictions on new, unseen data.

Neuarl Network
what is Neural networks
Neural networks, also known as artificial neural networks (ANNs), are a class of machine learning models inspired by the structure and functioning of the human brain. They are composed of interconnected nodes, called neurons, organized into layers. Each neuron takes input, applies a transformation, and produces an output. Neural networks are capable of learning complex patterns and relationships in data through a process called training.
How to use Neural networks
To use a neural network, you typically follow these steps:
Data Preparation: Gather and preprocess the data for training and testing. This step involves tasks such as data cleaning, normalization, and feature scaling.
Network Architecture: Determine the architecture of the neural network, including the number and size of layers, the number of neurons in each layer, and the activation functions to use.
Training: Initialize the network's weights and biases randomly, and then iteratively feed the training data through the network. The network adjusts its weights and biases through a process called backpropagation, which involves calculating gradients and updating the parameters to minimize the prediction errors.
Evaluation: Assess the performance of the trained network on a separate validation or test set. Common evaluation metrics include accuracy, precision, recall, and F1-score, depending on the problem type.
Prediction: Once the network is trained and evaluated, you can use it to make predictions on new, unseen data by feeding the input through the network and obtaining the corresponding output.
What problem it solved
Neural networks have been successfully applied to various problems, including:
Image and Object Recognition: Neural networks can learn to recognize objects, identify patterns, and classify images. They have been used for tasks like image classification, object detection, and facial recognition.
Natural Language Processing: Neural networks can process and understand human language, enabling applications such as sentiment analysis, language translation, and chatbots.
Speech Recognition: Neural networks have been effective in speech recognition tasks, such as converting spoken words into written text or enabling voice commands in various applications.
Recommendation Systems: Neural networks can learn user preferences and make personalized recommendations for products, movies, or music.
Now, let's take a code example using Python and the Keras library to create a simple neural network for image classification:


My Code of GNSS-RTK
1. Data processing
- Read in the training set and test set CSV data
	- Use the function named mount in the library named drive connecting  the dataset.
	- Use the function named read_csv in the library named pandas importing the dataset.
- The training set data is marked, and the three situations of good view, multipath and tunnel are judged according to the two characteristics of PACC H and SVs Used
	- the rule is: 
a. acc <= 3 good view	
b. acc > 3 && num >= 18 multipath 	
c. acc > 3 && num < 18 tunnel
	- two features: PACC H and SVs Used
	- three classifications: good view, multipath and tunnel.
- Conversion from string labels to one-hot encoding is done using apply and map
	- mapping = {'good view': [1,0,0], 'multipath': [0,1,0], 'tunnel': [0,0,1]}
- create labels and blend these labels or write labels into the dataset..
2. Model construction
- Constructed a simple fully connected neural network, including 4 Dense layers
Dense layer is a common fully connected layer in a neural network.
The so-called fully connected layer means that each neuron in this layer is connected to all neurons in the previous layer.
The Dense layer has the following characteristics:
- Each neuron is fully connected
- The mapping relationship between the parameters of the input layer and the Dense layer follows the linear relationship y=Wx+b. Among them, x is the input, which represents the output of the neuron in the front layer; W is the weight matrix, which represents the connection weight of each input to the current neuron, and b is the bias vector.
- The Dense layer usually adds a non-linear activation function, such as ReLU, to introduce non-linearity. The output of the activation function is the final output of the current layer.
- The parameters W and b of the Dense layer will be updated through the backpropagation algorithm of the loss function to minimize the Loss.
- Dense layers can be spliced together to form a fully connected neural network to achieve complex nonlinear mapping.
- Each layer uses the ReLU activation function, and the last layer uses softmax output
- ReLU is a linear rectification function, mainly used in the hidden layer of the neural network. It can introduce non-linearity and help the model learn complex function mappings.
- softmax is mainly used in the output layer of classification problems. It converts the output values of multiple neurons into probability distributions for multi-classification problems.
- ReLU is usually used in the hidden layer to extract features.
- Softmax is usually used in the output layer for classification.
- ReLU is widely used in various neural network models.
- softmax is more used in classification problems.
- The input layer accepts 2 features, and the output layer outputs the probability of 3 categories

3. Model training
- Use categorical_crossentropy as loss function
- How to determine our loss function:
First of all, according to our needs, we can know that our scenario is a classification problem, so we exclude the use of loss functions for other regression problems;
Secondly, according to the classification results, we divide the final results into three categories, so we exclude the use of binary_crossentropy, we use categorical_crossentropy;

- Categorical_crossentropy is a common loss function for multi-classification tasks.
It is suitable for multi-category problems that are mutually exclusive between categories, that is, each sample only belongs to one category, and does not belong to multiple categories at the same time.
The specific role of categorical_crossentropy is:
- It can measure the distance between the model's predicted class distribution and the true class distribution in multi-classification problems.
- The true distribution is usually represented using a one-hot encoding, where the position of the true category is 1 and the rest are 0.
- categorical_crossentropy computes the cross-entropy between the predicted probability distribution and the one-hot encoding distribution.
- The smaller the cross entropy value, the closer the two distributions are, indicating that the model predicts more accurately.

- Use accuracy as an evaluation indicator
- Set 100 epochs, batch_size is 32
1. epochs
- epochs refers to the number of times all training data is fed into the model training.
- Generally set to dozens to hundreds of times, so that the model iteratively learns multiple times on the training data.
- The more epochs, the longer the model training time, but often the better the effect.

2. batch_size
- batch_size refers to the number of samples for each input model training.
- Smaller batch training changes more frequently and is closer to stochastic gradient descent; larger batch learning is more stable.
- But too small a batch will lead to slow convergence, generally set between 8-256.

3. set epochs and batch_size
- The larger the epoch, the smaller the batch_size can be. The training times for different combinations are the same, but the effect may be different.
- Multiple experiments are required to find the best combination, so that the model can both fully learn and underfit.

4. Prediction and result processing
- Make predictions on the test set and get the probability that each sample belongs to 3 categories
	9.9999815e-01 = 9.9999815 * 10 ^ -1 = 0.99999815
- Convert probabilities to one-hot encoded predicted labels
	‘’‘python
preds = []
for row in y_pred:
pred = [0, 0, 0]
pred[row.argmax()] = 1
preds.append(pred)
test_df['preds'] = preds
‘’’
1. y_pred is an array of num_samples x num_classes, each row represents the predicted probability of a sample.
2. Through row.argmax(), you can get the index of the maximum value of each row, that is, the predicted category.
3. Using this index, create an array pred with all 0s, and set the corresponding index position to 1.
4. Add pred to the list preds, preds contains all one-hot encoding results.
5. Finally, assign preds to the preds column of the test set test_df to complete the conversion.

The content of y_pred, each row has 3 values, representing the predicted probability of the sample belonging to 3 categories.
Take the index where the maximum probability is located, and set it to 1 is the corresponding one-hot encoding.
For example, the argmax of [0.99,0.0001,0.0001] is 0, and it is [1,0,0] after conversion.
The whole process realizes the transition from softmax probability to one-hot encoding.

- Write the prediction results back to the test set DataFrame
- Display and export test set results to CSV file


5. Overall Evaluation
- The code has a certain modular structure, step by step completes a classification project
- The logic of the data processing part is relatively clear
- The model structure is relatively simple and may need to be tuned to improve the generalization ability
- The training process did not take measures to prevent overfitting
- Prediction and result processing are basically reasonable



La capacità acquisite(技能  )
U-center
Google Erath












Data


Firma del tirocinante



Firma del tutor aziendale (per i tirocini in azienda) 



