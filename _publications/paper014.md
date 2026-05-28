---
title: "A super-resolution framework for downscaling machine learning weather prediction toward 1-km air temperature"
collection: publications
category: manuscripts
permalink: /publication/paper014
excerpt: 'H. Park, S. Park\*, D. Kang\*, <u><b>J. H. Kim</b></u>. <b><i>npj Climate and Atmospheric Science</i></b>'
date: 2026-01-26
venue: 'npj Climate and Atmospheric Science'
paperurl: 'https://www.nature.com/articles/s41612-026-01328-5#article-info'
doi: '10.1038/s41612-026-01328-5'
---

Artificial intelligence has improved the accuracy and efficiency of weather forecasting, surpassing traditional numerical weather prediction models. However, the coarse spatial resolution of global weather forecasting systems limits their ability to capture fine-scale surface heterogeneity and localized extremes, particularly in regions with complex terrain or urban heat island effects. Here, we introduce SR-Weather, a deep learning-based super-resolution framework that converts coarse 0.25° forecasts into 1-km surface air temperature fields using MODIS-derived temperature targets and high-resolution auxiliary inputs. SR-Weather outperforms existing super-resolution methods by explicitly incorporating spatial context, such as topography, impervious surface fraction, and seasonal climatology maps of air temperature. When SR-Weather was applied to the FuXi global weather forecast, the 7-day forecast error in South Korea decreased by more than 20%, which was comparable to the 1-day forecast error from low-resolution prediction using simple spatial interpolation. In addition, SR-Weather effectively reconstructs missing pixels in MODIS-derived air temperature maps under heavy cloud contamination by leveraging auxiliary variables and climatologically smoothed fields. Although validated over South Korea, the framework relies on globally available MODIS products and minimal auxiliary inputs, making it feasible to retrain for other regions. These results indicate that SR-Weather is a scalable and high-fidelity tool for enhancing machine learning-based weather forecasts at fine spatial scales.
