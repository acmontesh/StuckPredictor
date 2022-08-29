# Pipe Sticking Predictor
----------------------------------------------------------------
By Abraham Montes. 2021

##Summary
The proposed methodology to build the predictor is summarized in the following schematic: <br>
<img src="img/methodology.png" /> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; *Figure 1. Methodology schematic*
<br>
This methodology uses CRISP-DM as a general framework, but was built specifically for this project. 


<p>Figure 1 shows the schematic of the methodology proposed in this paper. It parts from a rigorous study of the problem of interest, i.e., the anomalous event sought to be predicted. Then a sequence of modules logically ordered allows building the pre-dictor, with some iterative recursion to the starting points. </p><br>
    <p>
The construction of the predictor machine is based on the problem division approach (prediction of nominal streams and classification of contrasts). In summary, it classi-fies time windows into positive (anomalous) or negative (nominal) classes based on their contrast with the predicted nominal windows. The contrast measures are also called outliers scores, or anomaly grade. An actual nominal window is expected to produce low contrast with the predicted window. Conversely, an anomalous window is expected to generate high contrast. </p><br>
    <p>

These windows should be selected near the actual positive cases, so the predictor machine as a whole learns to identify patterns that warn the driller of the imminent occurrence of such events. This window will be referred to as forecast window. Next, these modules will be explained.
    </p>
###Stages
<ol>
    <strong><li>E.D.A.: </strong><p>
    This module involves the foundational task for the entire process. It focuses on un-derstanding the problem and seeks to obtain the dimensions of the feature vectors that will feed the nominal streams predictor. It is important to account for horizontal and vertical correlations. i.e., correlation across time on each channel (or sensor read-ing), and correlation between channels, as suggested by [4]. </p><br><p>
To explore vertical and horizontal correlations, it is useful to build sparse matrices with Pearson correlation factors. Figure 3 shows an instance for a set of selected channels. It is evident (both physically and statistically) that to build a nominal pre-dictor for the pressure channel (SPP), it is necessary to include the flow rate (GPM) as one of the dimensions of the feature vector. 
</p><br><p>
A sparse matrix with the adjacent time steps in a channel allows to visually find autocorrelations. However, autoregression methods need to follow such an initial approach.  Carrying over a principal component analysis (PCA) can be also useful, as it would allow to find the dimensions with the major fraction of the variance.
</p><br><p>
This module also seeks to select the most relevant channels (RC) for the anoma-lous event prediction. Over these channels the contrast will be measured. This can be done empirically, exploring the data surrounding the anomalous events and identify-ing anomalies per channel, or automatically, …. It is noteworthy that the methodolo-gy on figure 2 proposes an iteration back to this stage after the nominals prediction and the contrast classification modules. If the performance of the system is poor, a new set of RC can be selected. Hence, this selection of RC becomes part of a global search space.
</p><br><p>

Finally, it is recommended to include derived features. In the case of anomalous events in drilling operations, especially when dealing with collective anomalies, it is important to account for the trends among the whole set of channels, more than the time dependency of each channel alone. One way to capture this, is to calculate the discrete derivatives by using the left truncated version of the Taylor series. For in-stance, for channel i, the first discrete derivative at time t can be calculated as:</p><br>

$$
\left[ \frac{d\bar{S}^{(k)}}{dx} \right]_t = \frac{3{s_t}^{(k)} - 4{s_{t-1}}^{(k)} + {s_{t-2}}^{(k)}}{2h}
$$

<br>

Where $h$ is the time step. e.g., in the case of a data stream with 0.1 Hz frequency, then $h=10$ seconds.

<br>

</li>
<strong><li>Time window sliding: </strong>
<p>
This module is intended to build the instances for training the nominal predictors. For this, it is necessary to select a forecasting method. The literature reports 3 different methods for doing so: One step ahead, several steps ahead, and several steps by pre-dicting with predictions. In the first one, the model captures the time dependency of the variable and predicts only the channel’s value one time step ahead. In the sec-ond, the model learns to predict several steps ahead, i.e., a forecast window. In the latter, the model predicts one step ahead at first, and then that prediction becomes part of the feed for a next prediction. When the model predicts the next value, it be-comes again part of the feed stream and so forth. This type of forecasting is com-monly used in stocks applications [13]. </p><br><p>

It is noteworthy that the first approach would not be useful if the stream frequen-cy is high (more than 0.03 Hz). This is because the forecast window would not be sufficiently wide for the crew to react and prevent the anomalous event from hap-pening. Moreover, the third option usually produces unstable models in which only a few steps ahead are correctly predicted and then the model diverges [14]. Hence, the second approach is suggested. </p><br><p>

To build the instances set, given the second forecasting approach, 2 window sizes are selected (the feed window, T, and the forecast window, F) and slid over the da-taset. These windows are contiguous. Each sliding step extracts each of the instances. Hence, the resultant tensor has dimensions N×T×D, where T is the size of the feed window, D is the number of channels, and N is the number of times the windows were slid and is equal to X-T-F, where X is the number of time stamps in the dataset, and F the size of the forecast window. It is worth to mention that each RC has its own ten-sor, since each has different features. Figure 4 illustrates this procedure.</p><br>

<img src="img/windowsliding.jpg" /> 

*Figure 2. Time windows sliding to build the instances tensor for the prediction module. The feed window is slid over the feature channels for the RC (in this case the channel 0). The forecast window accounts for the RC alone. Once the sliding is finished, a tensor of size $N\times T\times D$ and a matrix $N\times F$ are obtained.*

<br>

</li>
<strong><li>Nominals prediction: </strong>
<p>
This module seeks to find the best nominal predictor per RC and well. It is worth to mention that it also finds the best sizes of feed and forecast window, since the nomi-nal predictor’s architecture depends on both. </p><br><p>

Since this module is intended to reproduce the nominal streams, the data must be free of “contaminant” events. i.e., anomalous events that could bias the predictor. </p><br><p>

For forecasting, regression models can be used to capture the horizontal and verti-cal correlation of each channel and with other channels. For time series with mono-tonic trends, or some degree of seasonality (i.e., exhibiting cycles over long periods of time), linear regression is sufficient and accurate to reproduce the time series of each channel under nominal conditions. However, in highly complex contexts, prediction models that capture nonlinear regressions must be used. An approximator of func-tions by excellence is neural networks. In addition, recurrent networks are particularly suitable for data with temporal dependency thanks to the ”recurrence” within the graph.  </p><br><p>

To use recurrent networks as nominal predictors in the context of anomalous events detection in drilling operations, the topology must be as shown in figure 5. The input instances will be matrices of size $T×D$, and there will be F output neurons, repre-senting the selected size of the forecast window.  </p><br><p>

Recurrent networks are not the only way of capturing such temporal dependency. However, their use is strongly recommended. The suitability of each predictor is measured on randomly selected nominal windows by calculating the mean absolute percentage error (MAPE). As shown in figure 2, if the MAPE score shows that the predictor is poor, then the method proposes the iteration back to E.D.A. </p><br><p>

To summarize, in this module a search is carried out, being the search space: </p><ul>
<li> Prediction model (recurrent networks are strongly recommended given their suitability to deal with temporal adjacency). </li>
<li>	Hyperparameters of the prediction model.</li>
<li>	Feed window size.</li>
<li>	Forecast window size.</li>
</ul><br>
This search can be executed exhaustively, which would imply considerably high computation time, especially for large datasets. There are 2 proposed alternatives to deal with time complexity: <ul> <br>
<li>	Parallel or distributed processing. </li>
<li>	Use of heuristics (such as genetic algorithms). </li>
</ul>
<br>

<img src="img/runit.jpg" /> 

*Figure 3. Topology of the recurrent predictor to be used for predicting nominal streams on each RC. The input instances are $T \times D$, where $T$ is the size of the feed window selected and $D$ is the number of channels. The number of output neurons is $F$, which is the selected size of the forecast window.*

<br>

<br>
</li>
<strong><li>Contrast classification: </strong>
<p>
This module parts from the best nominal predictor for each channel and each well. With this, the selected outlier scores are calculated for sampled positive (anomalous) and negative (nominal) windows, on the RC. Finally, with all the measured samples, a classifier is trained to find the decision boundary between anomalous and nominal in terms of the RC.  </p><br><p>
It is noteworthy that the outlier scores are part of the search space. If the classifier performs poorly, then an iteration back to E.D.A. needs to be done so continuously search through different RC, different features for their predictions, or different outli-er scores. In this methodology, we recommend 3 basic scores: </p><br>
<ul>
<li>

<strong>Mean square error (MSE): </strong> It allows capturing the deviation from the nominal character in the window, highlighting peaks and collective anomalies by squared powering. For a predicted forecast window  $\bar{F}$  and, the MSE is calculated as: <br>
$$
MSE^{(k)} = \frac{1}{F} \sum_{t=1}^{F} \left (\delta_t^{(k)} \right)^2
$$

<br>
</li> 
<br>

Where  $\delta_t^{(k)} = {c_{t,True}}^{(k)}-{\hat{c}_{t}}^{(k)}$  ,  ${c_{t,True}}$  is the real data point of channel  $k$  at time  $t$  and  ${\hat{c}_{t}}^{(k)}$  is the predicted value of channel $k$ at time $t$. <br><br>

<li>
<strong>Standard deviation of the squared deviations (σ): </strong>It allows to measure the dispersion of the deviations with respect to the nominal condition forecasting. i.e., it allows to capture the presence of abrupt jumps of the signal in the forecast window. As it can be noted, a collective anomaly would not have any relevant contrast in this predictor. Here is how to calculate this predictor for a forecast window F ̅:
</li><br>
<li>
Kurtosis of squared deviations (ϑ): This is a measure of the "heaviness" of the tail in the data distribution, assuming normality —and therefore of the amount of noise. However, it is also a measure of contrast applicable to other distributions, and still useful for detecting contextual anomalies [15]. As re-ferred in the literature, this outlier score is highly handful in many anomaly de-tection applications [4] because the fourth power allows to further highlight deviations from the mean in the forecast window.
</li>
</ul>
<br><p>
The resultant matrix contains the contrast measures for each channel and the la-bel column with a binary encoding of the anomalous condition, as shown in table 1. The table contains random numbers only to illustrate the structure of the matrix.</p><br><p>

Finally, the problem of detecting anomalous events in time series is generally linked to a pronounced imbalance of the classes, i.e., a very small number of positive cases (anomalies) compared to the number of negative (nominal) cases. There are different solutions proposed in the literature to perform data “augmentation”, from very simple ones (such as replication, noise injection, position inversion, among oth-ers) to more complex ones, such as the use of generative models (e.g. generative ad-versarial networks, GANs) [16].
</p><br><p>
We propose to stick to the simplest at first. In many occasions the resampling or replication of the positive cases results in classifiers with good performance, without incurring in pronounced biases [4]. 
</p><br>
</li>
</ol>
##Code Structure