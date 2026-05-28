---
title: "Improvement in Forecasting Short-Term Tropical Cyclone Intensity Change and Their Rapid Intensification Using Deep Learning"
collection: publications
category: manuscripts
permalink: /publication/paper008
excerpt: '<u><b>J. H. Kim</b></u>, Y. G. Ham*, D. Kim, T. Li, C. Ma. <b><i>Artificial Intelligence for the Earth Systems</i></b>'
date: 2024-04-25
venue: 'Artificial Intelligence for the Earth Systems'
paperurl: 'https://journals.ametsoc.org/view/journals/aies/3/2/AIES-D-23-0052.1.xml'
doi: '10.1175/AIES-D-23-0052.1'
---

Forecasting the intensity of a tropical cyclone (TC) remains challenging, particularly when it undergoes rapid changes in intensity. This study aims to develop a convolutional neural network (CNN) for 24-h forecasts of the TC intensity changes and their rapid intensifications over the western Pacific. The CNN model, the DeepTC, is trained using a unique loss function, an amplitude focal loss, to better capture large intensity changes, such as those during rapid intensification (RI) events. We showed that the DeepTC outperforms operational forecasts, with a lower mean absolute error (8.9%–10.2%) and a higher coefficient of determination (31.7%–35%). In addition, the DeepTC exhibits a substantially better skill at capturing RI events than operational forecasts. To understand the superior performance of the DeepTC in RI forecasts, we conduct an occlusion sensitivity analysis to quantify the relative importance of each predictor. Results revealed that scalar quantities such as latitude, previous intensity change, initial intensity, and vertical wind shear play critical roles in successful RI prediction. Additionally, the DeepTC utilizes the three-dimensional distribution of relative humidity to distinguish RI cases from non-RI cases, with higher dry–moist moisture gradients in the mid-to-low troposphere and steeper radial moisture gradients in the upper troposphere showed during RI events. These relationships between the identified key variables and intensity change were successfully simulated by the DeepTC, implying that the relationship is physically reasonable. Our study demonstrates that the DeepTC can be a powerful tool for improving RI understanding and enhancing the reliability of TC intensity forecasts.
