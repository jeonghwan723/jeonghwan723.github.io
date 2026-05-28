---
title: "Understanding machine learning weather prediction by designing a cost-efficient model with knowledge-oriented modules"
collection: publications
category: manuscripts
permalink: /publication/paper012
excerpt: 'M. Cheon, <u><b>J. H. Kim</b></u>, Y. Choi, S. Y. Kang, J. G. Lee, Y. G. Ham, J. Y. Kim, D. Kang*. <b><i>Scientific Reports</i></b>'
date: 2025-12-15
venue: 'Scientific Reports'
paperurl: 'https://www.nature.com/articles/s41598-025-32366-3'
---

Deep learning-based models are gaining prevalence in global weather forecasting, surpassing the performance of existing numerical models. However, training these models with high-resolution global weather data requires massive computational resources, making it difficult to conduct extensive experiments to understand the model processes. In addition, the reason for region- or variable-dependent accuracy in the machine learning models, along with the extra predictability provided by each component, remains unknown. Therefore, we propose a novel data-driven model named KARINA, which combines Geocyclic Padding and SENet modules with the ConvNeXt backbone to enhance weather forecasting while minimizing training resources. Despite its much lower training cost, KARINA achieved competitive performance compared to the recently developed data-driven models such as Pangu-Weather and GraphCast, while surpassing the numerical weather prediction of ECMWF IFS at a lead time of up to 10 days. The efficient training process and KARINA’s modular structure allow us to demonstrate the effectiveness of Geocyclic Padding and SENet through comprehensive trials. Geocyclic Padding significantly improves the modeling of horizontal advection, while SENet particularly captures the dynamics of atmospheric convection. These findings suggest that incorporating knowledge-oriented techniques can lead to reliable performance. This paper presents a framework for gaining a deeper understanding of the model mechanism and proposes ways to improve machine learning weather prediction models.
