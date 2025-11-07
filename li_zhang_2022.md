### Journal Pre-proof

Cross-feature fusion speech emotion recognition based on attention mask residual network and Wav2vec
2.

Xiaoke Li and Zufan Zhang

PII: S2352-8648(24)00129-

DOI: https://doi.org/10.1016/j.dcan.2024.10.

Reference: DCAN 799

To appear in: _Digital Communications and Networks_

Received date: 21 December 2023

Revised date: 5 August 2024

Accepted date: 14 October 2024

Please cite this article as: X. Li and Z. Zhang, Cross-feature fusion speech emotion recognition based on attention mask residual network and Wav2vec2.0,
_Digital Communications and Networks_ , doi:https://doi.org/10.1016/j.dcan.2024.10.007.

This is a PDF file of an article that has undergone enhancements after acceptance, such as the addition of a cover page and metadata, and formatting for
readability, but it is not yet the definitive version of record. This version will undergo additional copyediting, typesetting and review before it ispublished in its
final form, but we are providing this version to give early visibility of the article. Please note that, during the production process, errors may be discovered which
could affect the content, and all legal disclaimers that apply to the journal pertain.

©2024 Published by Elsevier.


```
Digital Communications and Networks(DCN)
```
```
journal homepage: http://www.elsevier.com/locate/dcan
```
## Cross-feature fusion speech emotion recognition based on attention mask

## residual network and Wav2vec 2.

#### Xiaoke Li∗, Zufan Zhang

School of Communications and Information Engineering, Chongqing University of Posts and Telecommunications, Chongqing 400065, China

Abstract

Speech Emotion Recognition (SER) has received widespread attention as a crucial way for understanding human emotional states. However,
the impact of irrelevant information on speech signals and data sparsity limit the development of SER system. To address these issues, this paper
proposes a framework that incorporates the Attentive Mask Residual Network (AM-ResNet) and the self-supervised learning model Wav2vec 2.
to obtain AM-ResNet features and Wav2vec 2.0 features respectively, together with a cross-attention module to interact and fuse these two features.
The AM-ResNet branch mainly consists of maximum amplitude difference detection, mask residual block, and an attention mechanism. Among
them, the maximum amplitude difference detection and the mask residual block act on the pre-processing and the network, respectively, to reduce
the impact of silent frames, and the attention mechanism assigns different weights to unvoiced and voiced speech to reduce redundant emotional
information caused by unvoiced speech. In the Wav2vec 2.0 branch, this model is introduced as a feature extractor to obtain general speech features
(Wav2vec 2.0 features) through pre-training with a large amount of unlabeled speech data, which can assist the SER task and cope with data sparsity
problems. In the cross-attention module, AM-ResNet features and Wav2vec 2.0 features are interacted with and fused to obtain the cross-fused
features, which are used to predict the final emotion. Furthermore, multi-label learning is also used to add ambiguous emotion utterances to deal
with data limitations. Finally, experimental results illustrate the usefulness and superiority of our proposed framework over existing state-of-the-art
approaches.

©c2022 Published by Elsevier Ltd.

KEYWORDS:
Speech emotion recognition, Residual network, Mask, Attention, Wav2vec 2.0, Cross-feature fusion

1. Introduction

Emotion is generally understood as a complex and specific subjective
response that people have to external objective things or events [1]. The
ways of conveying emotions are diverse, including text, speech signal,
facial expression and human body posture. Among them, the speech
signal is one of the most immediate, natural, and convenient ways to
express emotions. It has always been people’s desire to communicate
naturally with various machines through speech. Moreover, the tech-
nology of recognizing emotions from speech signals can be applied to
online teaching, mental health management, and in-vehicle communi-
cation systems [2, 3, 4]. Therefore, SER technology is of great signif-
icance for establishing human-machine emotional interaction in intelli-
gent systems, which can greatly promote the development of emotion
computing [5].
In the past few years, researchers have devoted themselves to de-
veloping better speech emotion classification methods for recognizing
the emotional state. Unfortunately, there are still some problems to be
solved in the previous work. Firstly, a speech signal is generally com-
posed of voiced speech, unvoiced speech, and silent frames [6]. Due
to the periodicity, the voiced speech can be recognized and extracted,

```
∗Xiaoke Li (Corresponding author) (email: D200101015@stu.cqupt.edu.cn).
```
(^1) Zufan Zhang (email: zhangzf@cqupt.edu.cn).
while the unvoiced speech and silent frames may increase the computa-
tional complexity and decrease the emotion recognition accuracy [7, 8].
Second, large-scale annotated speech emotion data is relatively scarce,
which makes the training of supervised deep learning models challeng-
ing. Without data of sufficient diversity and size, a model’s ability to
generalize may be limited [9]-[10].
For the first problem, the researchers mainly employ the Voice Ac-
tivity Detection (VAD) technology on speech signals to discard silent
frames or unvoiced speech [11, 12, 13]. For example, Harar et al. [11]
applied the VAD to preprocess the original speech signals for removal of
silent frames or unvoiced speech, thereby improving the results of SER.
However, using VAD to remove silent frames on the speech signal will
destroy the continuity of the timing signal, which helps to recognize
emotions. In addition, the proper fusion between the unvoiced speech
and voiced speech, instead of directly removing the unvoiced speech,
can improve the classification accuracy [14].
For the second problem, two methods have been provided to deal
with the problem of data scarcity. One of them is self-supervised learn-
ing models, which leverage pre-training on extensive unlabeled speech
data and fine-tuning to suit specific tasks like SER [10, 15, 16]. Inspired
by the success of self-supervised learning models in natural language
processing, models such as Wav2vec [17], Wav2vec 2.0 [18], HuBERT
[19], and WavLM [20] have emerged in the field of speech processing.
Pepino et al. [21] discovered that the combination of eGeMAPS with


2 Xiaoke Li, et al.

```
Speech signals
```
```
Cross
```
```
Cross
```
```
Feed
forward
```
```
Feed
forward
```
```
Concat
```
```
Cross
```
```
Cross
```
```
Feed
forward
```
```
Feed
forward
```
```
Concat
```
```
Cross-attention
```
```
Wav2vec 2.
```
```
......CNNCNN TransformerTransformer ......
```
```
AM-ResNet
```
```
B
N R
```
```
M M
```
```
C C
```
```
Classifiter
```
```
Avgpool
```
```
Linear
```
```
AM-ResNet features
```
```
Wav2vec 2.0 features
```
```
Fig. 1.An overview of the proposed SER framework using AM-ResNet, Wav2vec 2.0 and cross-attention.
```
the Wav2vec 2.0 features can improve the performance of SER. Xia et
al. [22] investigate multiple acoustical and learned features in SER and
find that the Wav2vec features are superior to the traditional features,
especially with a large temporal context. Yue et al. [23] obtained the
general speech representations for SER by leveraging the Wav2vec 2.
model as a feature extractor and achieved a promising result. The work
[24] proposes a joint ASR-SER training model where the Wav2vec 2.
representations are encoded into text and used as linguistic features to
support the SER task. The other is multi-label learning [25, 26], which
considers the multi-label assignment based on the annotators and gen-
erates an emotional proportion label for each utterance. According to
multi-labels, each utterance can be classified as clear or ambiguous.
According to [27], theclearutterance is the only emotion that has a
majority vote in multi-labels, and vice versa is an ambiguous utterance.
Unlike single-label learning, which focuses on the majority agreement
of annotators [5, 28], where the ambiguous emotion utterances are dis-
carded due to the lack of majority consensus, multi-label learning is
more suitable for the disagreement of emotion annotations and can han-
dle ambiguous emotion utterances. In addition, the work [27] proved
through experiments that ambiguous emotion utterances contain emo-
tional cues, which can improve the emotion recognition rate. Conse-
quently, this paper applies the self-supervised learning Wav2vec 2.
model as a feature extractor to obtain the general speech representa-
tions and employs multi-label learning to enable the network to also
use ambiguous emotion utterances to address the data limitations and
improve the recognition rate.

Based on the above ideas, this paper presents a framework for SER
to effectively identify emotions, which incorporates the attentive mask
residual network and the self-supervised learning model Wav2vec 2.0 to
obtain AM-ResNet features and Wav2vec 2.0 features respectively, to-
gether with a cross-attention module to interact and fuse these two fea-
tures. AM-ResNet mainly consists of maximum amplitude difference
detection, mask residual block, and an attention mechanism. Specifi-
cally, the two modules of the maximum amplitude difference detection
and the mask residual block work together to reduce the effect of the
silent frame: the former sets the silent frame in the speech signal to 0
in the preprocessing stage, and the latter keeps the value of the silent
frame unchanged during network learning. In addition, the mask resid-
ual block can also mask invalid information caused by zero padding.
The attention mechanism is responsible for assigning different weights
to unvoiced and voiced speech for the appropriate fusion of unvoiced
and voiced speech. The Wav2vec 2.0 model is introduced as a feature
extractor to obtain the general speech representations, named Wav2vec
2.0 features. Finally, the AM-ResNet and Wav2vec 2.0 features are in-
teracted with and fused in the cross-attention module to obtain the cross-
fused features, which are used to predict emotion in the classifier. Most

```
importantly, our proposed framework can handle the disagreement of
emotion annotation through multi-label learning. Furthermore, related
experiments are performed to verify the effectiveness of the proposed
framework, including comparisons with other recent methods.
The rest of this paper is organized into three sections. Section 2 de-
scribes the proposed framework for cross-feature fusion SER in detail.
In Section 3, some experiments and the experimental results are ana-
lyzed and discussed. Finally, conclusions are provided in Section 4.
```
2. The proposed method

```
2.1. Framework overview
To reduce the effect of irrelevant information on speech signals and
data sparsity, a framework is proposed, that incorporates the attentive
mask residual network and the self-supervised learning model Wav2vec
2.0 to obtain AM-ResNet features and Wav2vec 2.0 features respec-
tively, together with a cross-attention module to interact and fuse these
two features.
Fig. 1 depicts the total structure of the proposed framework, which
mainly comprises two stages: feature extraction, feature fusion and
classifier. A detailed description is given below.
Stage 1: feature extraction.First, the AM-ResNet features are de-
rived by feeding the original speech signal into the AM-ResNet, where
the AM-ResNet consists of the M-ResNet and an attention mechanism.
Second, the original speech signal is fed into the Wav2vec 2.0 base
model, which is obtained through 960 hours of pre-training and fine-
tuning on the Librispeech dataset of 16 kHz-sampled speech audio.
The last hidden state output by the Wav2vec 2.0 model is used as the
Wav2vec 2.0 features.
Stage 2: feature fusion and classifier.The cross-attention mod-
ule dynamically interacted and fused the AM-ResNet features and the
Wav2vec 2.0 features to acquire the cross-fused features. Finally, the
classifier is used to predict the final emotion based on these cross-fused
features.
```
```
2.2. AM-ResNet features
Inspired by the silent frames and unvoiced speech, which can in-
crease the computational complexity and decrease the emotion recog-
nition accuracy, this paper proposes AM-ResNet to improve the per-
formance of SER. In AM-ResNet, the maximum amplitude difference
detection and the mask residual block act on preprocessing and the net-
work, respectively to reduce the effect of silent frames. Moreover, the
attention mechanism assigns different weights to unvoiced and voiced
speech to reduce the redundancy of emotional information caused by
unvoiced speech as much as possible.
```

Cross-feature fusion speech emotion recognition based on attention mask residual network and Wav2vec2.0 3

```
Mask residual network Attention mechanism
```
```
Attention
```
```
Masks
```
```
Speech signals
```
```
Input
```
```
MADD C
```
```
BN R
```
```
M
```
```
C
```
```
BN R BN BN R
```
```
M M
```
```
C
```
```
R AMfeatures -ResNet
```
```
M
```
```
C
```
```
Fig. 2.The flowchart to obtain AM-ResNet features.
```
Fig. 2 sketches the flowchart for obtaining the AM-ResNet features
using the proposed model AM-ResNet. Firstly, we use the Maximum
Amplitude Difference Detection (MADD) module to preprocess the
original speech signal before entering the mask residual network (M-
ResNet), which is mainly composed of mask residual blocks. Mean-
while, the masks are generated based on the preprocessed speech sig-
nals. Second, M-ResNet is utilized to extract high-level features by
using these processed speech signals and corresponding masks as in-
put. Finally, the attention mechanism is introduced to acquire the AM-
ResNet features of the speech signals.

2.2.1. Maximum amplitude difference detection
In general, the silent frames contained in speech signals will affect
the learning of the network to some extent [7, 8]. Aiming at the prob-
lem that VAD removing the silent frame destroys the continuity of the
speech signal, this paper develops a preprocessing method, named Max-
imum Amplitude Difference Detection (MADD). Specifically, the Max-
imum Amplitude Difference (MAD) obtained by subtracting the mini-
mum amplitude from the maximum amplitude in a sliding window is
used as the detection feature to detect the speech presence region. For
theithsliding window with a length ofsand step size 1, theithMAD
can be denoted as

```
di=maxamp
```
##### (

```
segi
```
##### )

```
−minamp
```
##### (

```
segi
```
##### )

##### (1)

wheresegi=(segi, 1 ,segi, 2 ,···,segi,s) represents theithspeech segment
with a length ofsin a speech signal;maxamp(·) andminamp(·) are the
maximum and minimum amplitude value functions, respectively.
Through the above method,hMADs of the speech signal can be
obtained, defined asd=(d 1 ,d 2 ,···,dh).
If there is a silent region in the speech signal, then the silent region
can be represented byftimes the minimum valuemin(d) in MADs,
meaning that there is no sound in that region. Therefore, the amplitude
value of the corresponding speech region is set to 0. The above process
can be expressed as

```
segi=
```
##### 

##### 

##### 

##### 

```
segi, di≥f×min(d)
```
```
0 , di<f×min(d)
```
##### (2)

2.2.2. Mask residual block
In deep learning, the network used to automatically capture features
is critical to the whole network learning process. Before introducing
the mask residual block, the structure and characteristics of the original
residual block are first introduced. Residual Network (ResNet) [29], as
one of the CNNs, has been widely used in various fields and mostly con-
sists of residual blocks. Compared with the conventional CNN struc-
ture, the residual block has a shortcut connection and element-wise ad-
dition operation. Assuming that the input vector of each residual block
is denoted asxi, the output vector can be expressed as follows:

```
xi+ 1 =xi+F(xi,Hi) (3)
```
wherexi+ 1 represents the output vector of theithresidual block.F(·)
represents the residual function.Hi={hi,j| 1 ≤j≤k}represents the

```
weight set of theithresidual block, andkrepresents the number of layers
in theithresidual block.
As shown in Fig. 3, the figure on the left corresponds to the original
res-block [30]. This block is made up of convolutional layers, Batch
Normalization (BN) layers, and Rectified Linear Unit (ReLU). There-
fore, equation (3) can be expanded as:
```
```
xi+ 1 =x ̆i+F(x ̆i,Hi)
x ̆i+ 1 =σ(xi+ 1 )
```
##### (4)

```
whereσ(·)represents the function of BN layer and ReLU;x ̆iis the
output ofσ(·).
Let us consider the BN layer shown in Fig.3, and assuming that it
has ac-dimensional input denoted asZ=(z 1 ,···,zc)∈Rc×n, where
nis the number of utterances. For each dimensionzi=
```
##### (

```
zi, 1 ,···,zi,n
```
##### )T

##### ,

```
the output of the BN layerzˆi=
```
##### (

```
ˆzi, 1 ,···,zˆi,n
```
##### )T

```
can be described in the
following form [31]:
```
```
zˆi,j=ai×
```
```
zi,j−E(zi)
√
Var(zi)+
```
```
+bi (5)
```
```
whereVar(·) andE(·) represent variance and mean functions, respec-
tively.ai=
```
##### √

```
Var(ˆzi) andbi=E(zˆi) with initial conditiona 1 = 1 ,b 1 =
```
0. Besides,is introduced as the minimum value to prevent division
by 0 in the equation. The BN layer produces non-zero output when its
input is zero.
    Consequently, the silent region of the speech signal will be affected
by the above BN layer. Furthermore, although zero padding solves the
problem of variable length data input to the network, the zero padding
region will also be affected by the BN layer. According to equation
(5), the silent region and the zero padding part have non-zero values
after passing through the BN layer, which is shown in Fig. 4 (c). As a
result, invalid information is generated and continues to accumulate in
the deep network, which interferes with the learning of the network and
affects the recognition accuracy of the model.
    To cope with this problem, this paper designs a mask res-block
that keeps the value of the silent region and speech regions with zero
padding as 0. The mask is used to mark each sampling point in the
speech signal, indicating whether the feature is valid. The criterion for
feature validity is whether there is an area of voice activity in its re-
ceptive field. This is because the mask has only two values, 1 and 0,
indicating whether the feature is valid at that location. If the receptive
field at the corresponding position has only one filled region and one
silent region, its feature is invalid and inappropriate. In addition, the
mask changes with the length of the input feature.
    As shown in Fig. 3, the image on the right corresponds to the
mask res-block, which adds a trunk for inputting the mask based on
the original res-block. In this trunk, multiple max-pooling layers are
stacked, and each max-pooling layer has the same kernel size, stride,
and padding as the corresponding convolutional layer in each mask res-
block, which is represented by the double-headed dotted arrow in Fig.
3 (b). The mask goes through the max pooling layer and is multiplied
by the BN layer. This keeps the values of the silent region and speech
regions with zero padding as always at 0, avoiding the introduction of


4 Xiaoke Li, et al.

```
ReLU
```
```
Batch normalization
```
```
ReLU
```
```
Batch normalization
```
```
Conv7- 64
```
```
Conv7- 64
```
```
xi
```
```
xi+ 1
```
```
(a) Original res-block
```
```
ReLU
```
```
ReLU
```
```
Conv7- 64 Max-pooling
```
```
Conv7- 64
```
```
Batch normalizationBatch normalization
```
```
Batch normalizationBatch normalization
```
```
Max-pooling
```
```
Maski
```
```
Maski+ 1
```
```
xi
```
```
xi+ 1
```
```
ReLU
```
```
ReLU
```
```
Conv7- 64 Max-pooling
```
```
Conv7- 64
```
```
Batch normalization
```
```
Batch normalization
```
```
Max-pooling
```
```
Maski
```
```
Maski+ 1
```
```
xi
```
```
xi+ 1
```
```
Same kernel size,
stride,padding
```
```
(b) Mask res-block
```
```
Fig. 3.The structure of the original res-block and mask res-block.
```
```
0 0.2 0.4 0.6 0.8 1 1.
```
```
−0.
```
```
0
```
```
0.
```
```
1
```
```
Time (s)
```
```
Amplitude
```
```
(a) Original speech signal
```
```
0 0.2 0.4 0.6 0.8 1 1.
```
```
−0.
```
```
−0.
```
```
−0.
```
```
0
```
```
0.
```
```
0.
```
```
0.
```
```
Time (s)
```
```
Amplitude
```
```
(b) Fig. (a) after convolution
```
```
0 0.2 0.4 0.6 0.8 1 1.
```
```
−0.
```
```
−0.
```
```
0
```
```
0.
```
```
0.
```
```
Time (s)
```
```
Amplitude
```
```
(c) Fig. (a) after original res-block
```
```
0 0.2 0.4 0.6 0.8 1 1.
```
```
−0.
```
```
−0.
```
```
0
```
```
0.
```
```
0.
```
```
Time (s)
```
```
Amplitude
```
```
(d) Fig. (a) after mask res-block
```
```
Fig. 4.A comparison of the original speech signal under different operations.
```
invalid information that degrades model performance. Specifically, the
mask that marks each sampling point in the speech signal is defined as

```
mi=
```
##### (

```
mi, 1 ,mi, 2 ,···,mi,j
```
##### )

```
,mi,j∈{ 0 , 1 } (6)
```
wheremi,j=0 or 1 represents that the amplitude value of thejthsam-
pling point is 0 or not in the utterance.
In each mask res-block, the change of mask denotes as

```
mi+ 1 =MPL(l)(mi) (7)
```
wheremiandmi+ 1 represent the input and output mask vectors of the
ithmask residual block, respectively;MPL(l)(·) representsltimes max-
pooling operations.

```
The above process has described the core idea and structure of mask
res-block in detail. To further demonstrate the effectiveness of mask
res-block, Fig. 4 illustrates a comparison of the original speech signal
under different operations. Specifically, Fig.4 (a) is the original speech
signal, and Fig.4 (b) is the original speech signal after the convolution
operation. Based on Fig.4 (b), Figs.4 (c) and (d) are obtained by in-
putting the original speech signal into the original res-block and mask
res-block, respectively. The former is shifted upward by a certain dis-
tance as a whole, especially when the value of the silent regions changes
from 0. In contrast, the latter shows that the silent regions of the original
speech signal remain in an unchanged state, revealing the superiority of
mask res-block.
```

Cross-feature fusion speech emotion recognition based on attention mask residual network and Wav2vec2.0 5

2.2.3. Attention mechanism
Considering that there are not only silent frames but also unvoiced
speech in the speech signal, and unvoiced speech may cause information
redundancy, proper fusion by assigning different weights to unvoiced
and voiced speech may assist improve classification accuracy. Indeed,
the work [32] squeezed the spatial dimension by using global average
pooling on each feature map, which indicates that there is still a lot of
redundant information in CNNs. Besides, projection to low dimensional
space can effectively reduce redundant information and make the net-
work extract information more accurately. Inspired by this, an attention
mechanism that mainly consists of the convolution layers is designed to
assign different weights to the unvoiced and voiced speech. The main
strategy is to reduce the feature dimension to 1 without changing the
time dimension to focus on the importance of different time steps. As
depicted in Fig. 5, the attention mechanism is applied for the high-level
features extracted from the front multiple mask res-block to generate a
weighted vector. Then the weighted vector element-wise products high-
level features to obtain AM-ResNet features.
For an utterance, letE= (e 1 ,···,eI) ∈ RU×Irepresent theU-
dimensional high-level features extracted from an utterance by the front
multiple mask res-block, andIrepresents the total number of theU-
dimensional high-level features. In the attention mechanism, the fea-
ture dimension is reduced fromUto 1 after a convolution layer, and
then each attentive weight in a corresponding weighted vectorw =
(w 1 ,···,wI) undergoes normalization through a sigmoid layer, con-
straining the values within the range [0, 1]. The features with a cor-
responding attentive weight of 0 are deemed redundant and filtered out.
Conversely, the features with high attentive weight are significant and
occupy a high proportion during the subsequent classification. Then,
the weighted vectorwis calculated as

```
w=S igmoid(O(E,Watt)) (8)
```
whereO(·) represents convolution operator,Wattdenotes the weight
matrix of the convolution, andS igmoid(·) indicates the sigmoid acti-
vation function.
After calculating the weighted vectorw, the AM-ResNet features are
obtained by the following formula:

```
E ̄=E⊗w (9)
```
where⊗denotes the element-wise product.

```
Attention mechanism
```
```
High-level features AM-ResNet features
```
```
Weighted vector
Conv1d
```
```
Sigmoid
```
```
Conv1d
```
```
Sigmoid
```
```
Fig. 5.The architecture of the attention mechanism.
```
2.3. Wav2vec 2.0 features

The Wav2vec 2.0 model [18] is a self-supervised learning method
developed by Facebook AI Research (FAIR) for speech representation.
This model undergoes training in two stages. Initially, it is pre-trained
on a large-scale unlabeled speech dataset, followed by fine-tuning on
labeled data to be applied for downstream speech recognition tasks. Fig.
6 displayed the flowchart to obtain Wav2vec 2.0 features.
Specifically, the Wav2vec 2.0 model composes three major compo-
nents. The first component is the speech feature encoder, which consists
of several 1d convolutions, followed by layer normalization and the
GELU activation function. This component encodes the 1-D original

```
speech signal into latent speech representationsZ. The second compo-
nent is the context network, which consists of a convolutional layer and
Transformer, where the latent speech representationsZare randomly
masked at certain positions and then input to the context network to
obtain the context representationsC. The third component is the quan-
tization module, where the latent speech representationsZare mapped
to quantized latent speech representationsQvia the Gumbel softmax.
For self-supervised pre-training, the Wav2vec 2.0 model employs a
contrastive learning approach to quantify loss at the masked position.
The equation for comparative learning loss functionLmis specifically:
```
```
Lm=−log
```
```
exp(similarity(ct,qt)/κ)
∑
̃q∼Qt
```
```
exp(similarity(ct,q ̃)/κ)
```
##### (10)

```
wheresimilarity(ct,qt)=ctTqt/‖ct‖‖qt‖represents the cosine similar-
ity between context feature vectorctand quantized latent speech rep-
resentationsqt;tis the time step,κis the temperature,QtincludesK
distractors andqt.
To average the use ofVfeatures in each codebook, the diversity loss
functionLdis also applied in the learning target of the Wav2vec 2.
mode. This loss function can avoid the problem of mode collapse. Here,
the diversity loss functionLdcan be expressed as follows:
```
```
Ld=
```
##### 1

##### GV

##### ∑G

```
g= 1
```
```
−H(pg)=
```
##### 1

##### GV

##### ∑G

```
g= 1
```
##### ∑V

```
v= 1
```
```
pg,vlogpg,v (11)
```
```
wherepgrepresents the probability of theg-th group, andpg,vrepresents
the probability of selecting thev-th codebook entry in theg-th group.
As a result, the final loss functionLis described as follows:
```
```
L=Lm+βLd (12)
```
```
whereβrepresents a tuned hyperparameter, which controls the model
ability to distinguish negative samples.
In this paper, we mainly use the Wav2vec2-base-960h, which is built
through 960 hours of pre-training and fine-tuning on the Librispeech
dataset of 16 kHz sampled speech audio, to obtain the Wav2vec 2.
features by fetching the model’s last hidden state.
```
```
Transformer
... ...
```
```
CNN Wav2vec 2.features^0
```
```
Speech signals
```
```
Fig. 6.The flowchart to obtain Wav2vec 2.0 features.
```
```
2.4. Cross-attention module
Cross-attention is a common attention mechanism in deep learning
that plays an important role in processing sequence-to-sequence tasks
or tasks with strong correlation between sequences. Therefore, inspired
by the work of using cross-attention to fuse different modality [33], a
cross-attention module is applied to capture the interactions between
AM-ResNet features and Wav2vec 2.0 features for obtaining cross-
fused features with complementary properties.
To learn the interaction between the AM-ResNet featuresFaand the
Wav2vec 2.0 featuresFw, we first need to apply down-sampling to the
sequence dimension of the Wav2vec 2.0 featuresFwbecause the two
features have different lengths in the sequence dimension. Therefore, an
adaptive average pooling layer is used on the Wav2vec 2.0 featuresFwto
```

6 Xiaoke Li, et al.

```
Cross-fused
Concat features
Softmax
```
```
Softmax
```
```
norm
```
```
Feed
forward
```
```
norm
FC
```
```
Wav2vec 2.0 features FC
```
```
AM-ResNet
features
```
```
Concat norm
```
```
Concat norm forwardFeed
```
```
Avg
pool
```
```
Vw
Kw
Qa
```
```
Qw
Ka
Va
```
```
Fig. 7.The architecture of the cross-attention.
```
make the sequence dimensions of the two features equal, the Wav2vec
2.0 featuresFwafter the adaptive average pooling layer are expressed
as:
F∗w=avgpool(Fw) (13)

where avgpool(·) represents the 1d adaptive average pooling operation.
Second, linear projection is needed to convert the AM-ResNet fea-
turesFaand the Wav2vec 2.0 featuresF∗winto three terms, namely
query, key, and value:

```
Qa,Qw=WQaFaWQwF∗w (14)
```
```
Ka,Kw=WKaFaWKwF∗w (15)
```
```
Va,Vw=WVaFaWVwF∗w (16)
```
whereQa,Ka,VaandQw,Kw,Vware the query, key, and value of the
AM-ResNet featuresFaand the Wav2vec 2.0 features after the adaptive
average pooling layerF∗wrespectively;WQa,WKa,WVaandWQw,WwK,WVw
are the projection matrices.
Third, the cross-computation of dot products between the query and
key is executed for both the AM-ResNet featuresFaand the Wav2vec
2.0 featuresFwto estimate the correlations between two features. The
computed results are then scaled by division with

##### √

dand normaliza-
tion through the application of the softmax function to obtain weights
on the value. The interaction information between two features can be
expressed as follows:

```
Fa→w=so ftmax(QwKTa/
```
##### √

```
d)Va (17)
```
```
Fw→a=so ftmax(QaKTw/
```
##### √

```
d)Vw (18)
```
whereFa→wandFw→arepresent the interaction information from the
AM-ResNet featuresFa to the Wav2vec 2.0 featuresFw and the
Wav2vec 2.0 featuresFwto the AM-ResNet featuresFa, respectively;d
is the feature dimension.
Fourth, we update one feature with the interaction information from
the other feature, which can be expressed as follows:

```
Fa+w→a=LN(Concat(Fa,Fw→a)) (19)
```
```
Fw+a→w=LN(Concat(F∗w,Fa→w)) (20)
```
whereLN(·) represents layer normalization operation.
Then a fully connected feed-forward layer [34] is applied behind the
cross-attention layer to make the computations more efficient, which
can be represented as

```
F ́a=LN(Fa+w→a+FeedForward(Fa+w→a)) (21)
```
```
F ́w=LN(Fw+a→w+FeedForward(Fw+a→w)) (22)
```
whereFeedForward(·) represents the fully connected feed-forward
layer.

```
Ultimately, the cross-fused features, denoted asFc, are obtained by
concatenating the AM-ResNet features containing interaction informa-
tion from Wav2vec 2.0 featuresF ́aand the Wav2vec 2.0 features con-
taining interaction information from AM-ResNet featuresF ́w, which
can be expressed as
Fc=Concat(F ́aF ́w) (23)
The above cross-fused featuresFcobtained form the cross-attention
module are used in classifier to predict the underlying emotion through
the integration of an adaptive average pooling layer and a fully con-
nected layer within the classifier.
```
3. Experiments

```
To assess the effectiveness of the proposed framework, we conduct
speaker-independent experiments on the Interactive Emotional Dyadic
Motion Capture (IEMOCAP) dataset [35] by applying a 10-fold cross-
validation approach (selecting one speaker per fold as the test set and
the remaining 9 speakers as the training set). Multi-label learning is
used to add ambiguous emotional utterances for dealing with data lim-
itations. Furthermore, we conduct the comparative analyses between
the proposed framework and the latest research approaches in SER. Fi-
nally, we systematically analyze the advantages of each module of the
proposed framework.
```
```
3.1. Experimental details
3.1.1. Dataset and preprocessing
This paper uses the IEMOCAP dataset to perform experiments to val-
idate the effectiveness of the proposed method. The IEMOCAP dataset
[35] is an English emotion dataset published by the University of South-
ern California. It is collected from the performances of 10 professional
actors, half of whom are male. The whole dataset contains 5 sessions,
and each session consists of a performance of one actor and one actress.
In this dataset, each utterance is labeled with categorical emotion labels
by at least three annotators, who retain the flexibility to select multi-
ple emotion labels per utterance. For example, an utterance might be
simultaneously labeled as “anger” by two annotators and as having a
“neutral” emotion by a third annotator, thereby forming the multi-label
combination of “anger, neutral”. Due to the inherently subjective na-
ture of human emotion expression and perception, inconsistencies in
emotion annotation arise. To address this problem, Ando et al. [27]
introduced the multi-label assignment approach, which involves defin-
ing happiness, neutral, anger, and sadness as target emotions, and then
categorizing the dataset based on whether the target emotion has a dom-
inant or secondary status. Following this approach, this paper divides
the IEMOCAP dataset into clear and ambiguous subsets. In the clear
subset, an utterance is labeled as clear if the target emotions have a
dominant status, indicated by a consensus of at least two annotators on
a common label. Conversely, the ambiguous subset indicates that the
target emotions are secondary, either a lack of consensus among three
annotators or an agreement between two annotators on a label that does
```

Cross-feature fusion speech emotion recognition based on attention mask residual network and Wav2vec2.0 7

not correspond to the target emotions. Furthermore, since the speech
signals are not of equal length, we split the utterance into segments of
equal duration of 4s and perform zero-padding on the utterance that
lasts less than 4s.

Table 1
Hyperparameter configurations in stages 1 and 2.

```
Stage 1 Stage 2
Hyperparameter Value Hyperparameter Value
Batch size 32 Batch size 64
Epoch 400 Epoch 300
Weight decay 0.00001 Weight decay 0
Momentum 0.9 Betas (0.93, 0.98)
```
3.1.2. Experimental settings
In this paper, the overall training process mainly consists of two
stages: Stage 1 focuses on the training of AM-ResNet, and Stage 2
focuses on the feature interaction and fusion between two features, as
well as the classifier. Table 1 lists the detailed configuration of the two
stages.
Stage 1: The proposed AM-ResNet is implemented on PyTorch,
which is a scientific computing package based on Python. The learn-
ing rate is kept constant at 0.01 for the first 40 epochs and divided by 10
and 100 for the next 80 and 280 epochs. Besides, Stochastic Gradient
Descent (SGD) algorithm is employed to optimize AM-ResNet.
Stage 2: The AM-ResNet features and the Wav2vec 2.0 features
are first interacted with and concatenated in the cross-attention module,
culminating in the generation of cross-fused features. Subsequently,
these features undergo classification to discern and predict the under-
lying emotion. The PyTorch package is also used in this process. The
learning rate is 0.001, and the Adaptive Moment Estimation (Adam)
algorithm is applied to optimize feature fusion and classifier.

3.1.3. Evaluation metrics
To measure the accuracy of SER, the experiments employ two
popular evaluation indicators within the field of emotion recognition:
Weighted Accuracy (WA) and Unweighted Average Recall (UAR).
WA is applied to balance the total performance of the SER system. It
is calculated by counting the number of correctly classified samples and
then dividing by the number of total samples. For a dataset containing
Ncategories, it can be expressed as

##### WA=

##### ∑N

```
i= 1 TPi
∑N
i= 1 (TPi+FPi)
```
##### (24)

Since the distribution of samples in each category of emotion datasets
is often unbalanced, using only WA as an evaluation indicator will lead
to categories with a larger number of samples dominating. Therefore,
UAR is proposed to comprehensively measure the recognition perfor-
mance of all categories. UAR first calculates the recallRecalliof each
category, and then divides the sum of the accuracy rates by the number
of categoriesN:

```
Recalli=
```
```
TPi
TPi+FNi
```
##### (25)

##### UAR=

##### ∑N

```
i= 1 Recalli
N
```
##### (26)

3.2. Comparison results

The evaluation of recognition accuracy involves comparative analy-
ses between our proposed framework and various competitive methods,
encompassing the methods only utilizing clear emotion utterances [21],

```
Table 2
Performance comparisons with existing approaches on the IEMOCAP dataset.
The ‘-’ implies the lack of this measure, and the best results are labeled in bold
.
Method Training set WA (%) UAR (%)
Chen et al. [36] (2018) clear - 64.
Chou et al. [37] (2019) clear - 61.
Pepino et al. [21] (2021) clear - 67.
Li et al. [24] (2022) clear 63.40 -
Yue at al. [23] (2022) clear 68.29 -
Pastor et al. [38] (2023) clear - 65.
AM-ResNet clear 67.35 65.
AM-ResNet+W2V2+CA clear 68.36 67.
Etienne et al. [26] (2018) clear+ambiguous 64.50 -
Ando et al. [27] (2018) clear+ambiguous 62.60 63.
Chou et al. [37] (2019) clear+ambiguous - 61.
Upadhyay et al. [39] (2024) clear+ambiguous - 63.
AM-ResNet clear+ambiguous 67.50 66.
AM-ResNet+W2V2+CA clear+ambiguous 70.79 72.
```
```
Table 3
The comparison of cross-validation accuracy between ResNet and M-ResNet on
the IEMOCAP dataset. The best results are labeled in bold.
```
```
Fold Testing set
```
```
ResNet M-ResNet
UAR (%) WA (%) UAR (%) WA (%)
1 Session1F 72.57 73.12 73.95 75.
2 Session1M 68.53 66.25 69.93 68.
3 Session2F 70.59 69.09 74.41 71.
4 Session2M 73.01 66.12 72.31 63.
5 Session3F 64.34 68.78 64.87 66.
6 Session3M 64.38 63.31 63.94 67.
7 Session4F 63.48 73.80 62.17 68.
8 Session4M 66.10 66.59 68.20 66.
9 Session5F 62.87 64.37 64.32 65.
10 Session5M 59.71 63.59 63.29 65.
Average 66.56 67.50 67.74 67.
```
```
[23]-[24], [36]-[38] and the methods incorporating both clear and am-
biguous emotion utterances [26, 27]. The detailed results of this com-
parative assessment are presented in Table 2.
In all approaches, our proposed framework achieves more robust per-
formance both in UAR and WA when trained with both clear and am-
biguous utterances. Remarkably, substantial improvements in UAR are
observed. Compared with the methods that only use clear emotion ut-
terances [21], [23]-[24], [36]-[38], AM-ResNet only uses clear emotion
utterances as the training set and surpasses the methods described in
[24], [36], [37] in UAR or WA. In addition, AM-ResNet+W2V2+CA
uses only clear emotion utterances, as the training set outperforms all
methods in either UAR or WA. This indicates that the proposed AM-
ResNet+W2V2+CA can leverage the self-supervised learning capabil-
ities of Wav2vec 2.0 to address data sparsity concerns, and the cross-
attention module provides a suitable way to fuse the features learned
using supervised and self-supervised methods.
In addition, compared with the methods [26, 27] using both clear
and ambiguous emotion utterances, the proposed AM-ResNet using
both clear and ambiguous emotion utterances also exceeds them. In
```

8 Xiaoke Li, et al.

```
Happiness Neutral Anger Sadness
```
```
Happiness
```
```
Neutral
```
```
Anger
```
```
Sadness
```
```
52.61 16.71 8.06 22.
```
```
14.15 55.50 9.45 20.
```
```
8.20 9.13 77.88 4.
```
```
5.54 12.14 2.08 80.
```
```
(a) The confusion matrix of speaker-independent experiments on
the IEMOCAP dataset using ResNet (Average UAR=66.56%)
```
```
Happiness Neutral Anger Sadness
```
```
Happiness
```
```
Neutral
```
```
Anger
```
```
Sadness
```
```
54.80 17.57 9.54 18.
```
```
14.20 58.16 11.12 16.
```
```
7.83 8.96 79.97 3.
```
```
5.06 13.07 3.86 78.
```
```
(b) The confusion matrix of speaker-independent experiments on
the IEMOCAP dataset using M-ResNet (Average UAR=67.74%)
```
```
Fig. 8.ResNet VS. M-ResNet.
```
detail, the above methods apply ambiguous emotion utterances to im-
prove classification performance, but the effect of silent frames and un-
voiced speech on network learning is ignored. In contrast, AM-ResNet
employs maximum amplitude difference detection and mask residual
blocks acting on preprocessing and network respectively to reduce the
influence of silent frames. Meanwhile, the attention mechanism reduces
redundancy in emotional information caused by unvoiced speech to the
utmost extent.
In particular, compared with AM-ResNet+W2V2+CA using only
clear emotion utterances as the training set, AM-ResNet+W2V2+CA
using both clear and ambiguous emotion utterances obtained improve-
ments in both UAR and WA, especially in UAR. This phenomenon can
also be observed on AM-ResNet. This means that the ambiguous emo-
tion utterances contain emotional cues, and the proposed framework can
utilize the ambiguous emotion utterances, which alleviates data limita-
tion problems and supports the increase of UAR and WA.
The above comparative analyses of the proposed framework and the
existing methods have shown the advancement of our proposed frame-
work. In the subsequent experiments, AM-ResNet, Wav2vec 2.0 and
the cross-attention module will be analyzed.

3.3. Analysis of AM-ResNet

3.3.1. Analysis of mask res-block
The silent region and the zero padding part have non-zero values after
passing through the BN layer, which may interfere with the learning of
the network and affect the recognition accuracy of the model. In mask
res-block, the mask records the valid and invalid regions by marking
each sampling point in the speech signal. This can keep the value of
the silent region and speech region with zero padding always 0. The
analysis of the mask res-block is given to verify its effectiveness on the
SER task.
To demonstrate the effectiveness of the proposed mask res-block,
comparative experiments are conducted on the IEMOCAP dataset be-
tween ResNet utilizing the original res-block and M-ResNet employing
the mask res-block. All experimental parameters and the training set
(clear+ambiguous) remain constant, except forthe different res-blocks
(original res-block and mask res-block) used. And the cross-entropy
loss function is applied in both cases.
As illustrated in Table 3, the average performance of the proposed
M-ResNet integrating the mask res-block, surpasses that of ResNet by
1.18% in UAR and 0.14% in WA. This indicates that the mask res-block

```
Table 4
The comparison of cross-validation accuracy between M-ResNet and AM-
ResNet on the IEMOCAP dataset. The best results are labeled in bold.
```
```
Fold Testing set
```
```
M-ResNet AM-ResNet
UAR (%) WA (%) UAR (%) WA (%)
1 Session1F 73.95 75.27 72.19 73.
2 Session1M 69.93 68.97 71.22 69.
3 Session2F 74.41 71.43 72.32 71.
4 Session2M 72.31 63.79 74.05 68.
5 Session3F 64.87 66.25 69.33 71.
6 Session3M 63.94 67.11 64.20 63.
7 Session4F 62.17 68.45 64.85 74.
8 Session4M 68.20 66.35 67.07 66.
9 Session5F 64.32 65.75 63.82 66.
10 Session5M 63.29 65.44 60.35 66.
Average 67.74 67.88 67.90 68.
```
```
can mitigate the impact of the silent frames and zero-padded areas of
the speech signal after the BN layer is no longer 0 on the model recog-
nition accuracy. Furthermore, Fig. 8 shows the speaker-independent
classification performance per emotion of ResNet and M-ResNet on the
IEMOCAP dataset. Comparative analysis with Fig. 8 (a) reveals a rela-
tive improvement in the recognition accuracy of happiness, neutral and
anger in Fig. 8 (b). Specifically, the number of happiness misclassifica-
tions as sadness decreased by 4.54%, indicating that the mask res-block
can reduce the effect of invalid information on the SER task.
```
```
3.3.2. Analysis of attention mechanism
The unvoiced speech in speech signals can affect the performance
of the model due to its aperiodicity, so it is essential to assign differ-
ent weights to the unvoiced and voiced speech by dynamically creating
weight for each sampling point in the time dimension. In AM-ResNet,
an attention mechanism is designed to discriminate the unvoiced and
voiced speech and help to fuse it properly for better classification ac-
curacy. The analysis of the attention mechanism is given to verify its
effectiveness on the SER task.
```

Cross-feature fusion speech emotion recognition based on attention mask residual network and Wav2vec2.0 9

```
Happiness Neutral Anger Sadness
```
```
Happiness
```
```
Neutral
```
```
Anger
```
```
Sadness
```
```
54.80 17.57 9.54 18.
```
```
14.20 58.16 11.12 16.
```
```
7.83 8.96 79.97 3.
```
```
5.06 13.07 3.86 78.
```
```
(a) The confusion matrix of speaker-independent experiments on
the IEMOCAP dataset using M-ResNet (Average UAR=67.74%)
```
```
Happiness Neutral Anger Sadness
```
```
Happiness
```
```
Neutral
```
```
Anger
```
```
Sadness
```
```
55.06 18.94 7.80 18.
```
```
11.54 60.37 8.38 19.
```
```
7.77 9.76 77.93 4.
```
```
3.96 14.99 2.64 78.
```
```
(b) The confusion matrix of speaker-independent experiments on
the IEMOCAP dataset using AM-ResNet (Average UAR=67.93%)
```
```
Fig. 9.M-ResNet VS. AM-ResNet.
```
```
Happiness Neutral Anger Sadness
```
```
Happiness
```
```
Neutral
```
```
Anger
```
```
Sadness
```
```
55.06 18.94 7.80 18.
```
```
11.54 60.37 8.38 19.
```
```
7.77 9.76 77.93 4.
```
```
3.96 14.99 2.64 78.
```
```
(a) Confusion matrix of speaker-independent experiments on the
IEMOCAP dataset using AM-ResNet (Average UAR=67.93%)
```
```
Happiness Neutral Anger Sadness
```
```
Happiness
```
```
Neutral
```
```
Anger
```
```
Sadness
```
```
58.81 18.61 14.44 8.
```
```
10.84 67.07 9.98 12.
```
```
6.92 8.18 81.92 2.
```
```
4.68 10.29 3.74 81.
```
```
(b) The confusion matrix of speaker-independent experiments on
the IEMOCAP dataset using AM-ResNet+W2V2+CA (Average
UAR=72.21%)
```
```
Fig. 10.AM-ResNet VS. AM-ResNet+W2V2+CA.
```
To evaluate the effect of the attention mechanism, comparative ex-
periments are conducted on the aforementioned dataset, presenting the
performance of M-ResNet with the Attention Mechanism (AM-ResNet)
in contrast to M-ResNet without the attention mechanism. All param-
eter configurations and training set (clear+ambiguous) are consistent
across the experiments, with the sole variation being the presence or
absence of the attention mechanism. The cross-entropy loss function is
employed in both cases.

The result is shown in Table 4, in which one can see that the atten-
tion mechanism improves both in WA and UAR, as it reduces the re-
dundant information about unvoiced speech and focuses on the voiced
speech. Besides, the confusion matrixes of speaker-independent exper-
iments on the IEMOCAP dataset using M-ResNet and AM-ResNet are
displayed in Fig. 9. A comparison between Fig. 9 (a) and (b) highlights
that Fig. 9 (a) show a relative improvement in the recognition accuracy
of happiness, neutral and sadness, while reducing the misclassification

```
of happiness as anger by 1.74%, further indicating that the attention
mechanism can provide a solution for the redundant information of the
speech signals.
```
```
3.4. Analysis of the Wav2vec 2.0 features and cross-attention
```
```
In this ablation experiment, we demonstrate the impact of the
Wav2vec 2.0 features and cross-attention module on the proposed
framework by comparing the classification performance of the proposed
AM-ResNet, AM-ResNet+W2V2+FC, and AM-ResNet+W2V2+CA.
All parameter configurations and training set (clear+ambiguous) are
consistent across the experiments. The Adam algorithm is applied to
optimize both cases.
As displayed in Table 5, the average performance of AM-
ResNet+W2V2+CA is degraded in both UAR and WA compared to
```

10 Xiaoke Li, et al.

Table 5
The comparison of cross-validation accuracy between AM-ResNet, AM-ResNet+W2V2+FC, and AM-ResNet+W2V2+CA on the IEMOCAP dataset. The best
results are labeled in bold.

```
Fold Testing set
```
```
AM-ResNet AM-ResNet+W2V2+FC AM-ResNet+W2V2+CA
UAR (%) WA (%) UAR (%) WA (%) UAR (%) WA (%)
1 Session1F 72.19 73.12 68.94 68.75 77.37 77.
2 Session1M 71.22 69.81 69.12 65.53 73.02 71.
3 Session2F 72.32 71.43 71.01 70.69 79.29 80.
4 Session2M 74.05 68.69 67.14 62.92 77.77 74.
5 Session3F 69.33 71.31 62.29 63.22 68.61 68.
6 Session3M 64.20 63.88 59.87 59.62 64.49 62.
7 Session4F 64.85 74.06 64.08 65.91 70.10 70.
8 Session4M 67.07 66.83 63.51 60.04 72.30 70.
9 Session5F 63.82 66.13 66.72 63.39 72.79 70.
10 Session5M 60.35 66.13 59.71 57.30 66.39 63.
Average 67.90 68.87 65.24 63.74 72.21 70.
```
AM-ResNet indicating that model overfitting can occur when only com-
bining the features of AM ResNet and Wav2vec 2.0 without additional
regularization mechanisms. Conversely, the average performance of
the proposed AM-ResNet+W2V2+CA compared to AM-ResNet im-
proves by 4.37% in UAR and 1.92% in WA, and UAR improves more
than WA. This improvement implies that the integration of the cross-
attention module facilitates the sparseness of combined features from
AM-ResNet and Wav2vec 2.0 thereby reducing the influence of noise
and irrelevant information, enhancing the model’s robustness against
noise perturbations and ultimately augmenting its generalization capa-
bilities.
To further investigate the recognition rate for each emotion, the
speaker-independent classification performance per emotion of the pro-
posed AM-ResNet and AM-ResNet+W2V2+CA on the IEMOCAP
dataset is depicted in Fig. 10. Notably, both Fig. 8 (a) and (b)
have higher recognition accuracy for sadness, anger, and neutral emo-
tions, while happiness demonstrates a comparatively lower accuracy.
This consistent trend in recognition rates across both AM-ResNet and
AM-ResNet+W2V2+CA. The low recognition rate of happiness is at-
tributed to the small number of happy utterances compared to the
other three categories of emotions, and happiness is often confused
with anger and neutral. Besides, compared with AM-ResNet, AM-
ResNet+W2V2+CA obtains the greatest improvement in the recogni-
tion rate of neutral, followed by sadness, happiness, and anger. In
particular, the proposed AM-ResNet+W2V2+CA alleviates the phe-
nomenon of happiness being misclassified as sadness in the AM-
ResNet. This means the addition of the Wav2vec 2.0 features and cross-
attention module increases the ability to distinguish different emotional
states of the model, which is essential for the accuracy of models, espe-
cially in applications such as emotion recognition.

4. Conclusions

In this paper, we present a framework that combines AM-ResNet and
Wav2vec 2.0 to obtain AM-ResNet features and Wav2vec 2.0 features
respectively, and a cross-attention module to interact and fuse these two
features, which are used to detect the emotional state of speech. First,
AM-ResNet is a better solution to the problem that silent frames and
unvoiced speech can increase computational complexity and decrease
emotion recognition accuracy. To reduce the effect of the silent frames
from the original speech signals, the maximum amplitude difference
detection and the mask residual block have been designed. The mask
residual block can keep the value of silent regions and speech regions
with zero padding unchanged. Then, the attention mechanism assigns
different weights to unvoiced and voiced speech to reduce the redun-

```
dant emotional information caused by unvoiced speech. Second, the
self-supervised learning model Wav2vec 2.0 and multi-label learning
are introduced to address the data sparsity in SER. Finally, extensive
experiments on the publicly available dataset IEMOCAP have been im-
plemented. The classification results have demonstrated the superiority
of our proposed framework over existing state-of-the-art methods.
```
```
Acknowledgements
```
```
This work is supported by Chongqing University of Posts and
Telecommunications Ph.D. Innovative Talents Project (Grant No.
BYJS202106), Chongqing Postgraduate Research Innovation Project
(Grant No. CYB21203).
```
```
References
```
```
[1] P. E. Ekman, R. J. Davidson (Eds.), The Nature of Emotion: Fundamen-
tal Questions, Series in Affective Science, Oxford University Press, New
York, 1994.
[2] D. Wu, S. Si, S. Wu, R. Wang, Dynamic trust relationships aware data
privacy protection in mobile crowd-sensing, IEEE Internet Things J. 5 (4)
(2018) 2958–2970.
[3] D. Wu, H. Shi, H. Wang, R. Wang, H. Fang, A feature-based learning
system for internet of things applications, IEEE Internet Things J. 6 (2)
(2019) 1928–1937.
[4] A. B. Nassif, I. Shahin, I. Attili, M. Azzeh, K. Shaalan, Speech recognition
using deep neural networks: A systematic review, IEEE Access 7 (2019)
19143–19165.
[5] S. Zhang, S. Zhang, T. Huang, W. Gao, Speech emotion recognition us-
ing deep convolutional neural network and discriminant temporal pyramid
matching, IEEE Trans. Multimedia 20 (6) (2018) 1576–1590.
[6] M. B. Akc ̧ay, K. Oguz, Speech emotion recognition: Emotional mod- ̆
els, databases, features, preprocessing methods, supporting modalities, and
classifiers, Speech Commun. 116 (2020) 56–76.
[7] Ch. Rakesh, R. R. Rao, S. R. Krishna, A comparative study of silence and
non silence regions of speech signal using prosody features, in: Proceed-
ings of the 2016 International Conference on Communication and Elec-
tronics Systems (ICCES), IEEE, Coimbatore, India, 2016, pp. 1–4.
[8] Y. Ma, Y. Hao, M. Chen, J. Chen, P. Lu, A. Kosir, Audio-visual emotionˇ
fusion (avef): A deep efficient weighted approach, Inf. Fusion 46 (2019)
184–192.
[9] A. Satt, S. Rozenberg, R. Hoory, Efficient emotion recognition from speech
using deep learning on spectrograms, in: Proceedings of Interspeech 2017,
ISCA, 2017, pp. 1089–1093.
[10] Y. Gao, C. Chu, T. Kawahara, Two-stage finetuning of wav2vec 2.0 for
speech emotion recognition with asr and gender pretraining, in: Proceed-
ings of Interspeech 2023, ISCA, 2023, pp. 3637–3641.
```

Cross-feature fusion speech emotion recognition based on attention mask residual network and Wav2vec2.0 11

[11] P. Harar, R. Burget, M. K. Dutta, Speech emotion recognition with deep
learning, in: Proceedings of the 2017 4th International Conference on Sig-
nal Processing and Integrated Networks (SPIN), IEEE, Noida, Delhi-NCR,
India, 2017, pp. 137–140.
[12] I. J. Tashev, Zhong-Qiu Wang, K. Godin, Speech emotion recognition
based on gaussian mixture models and deep neural networks, in: Proceed-
ings of the 2017 Information Theory and Applications Workshop (ITA),
IEEE, San Diego, CA, USA, 2017, pp. 1–4.
[13] M. Pandharipande, R. Chakraborty, A. Panda, S. K. Kopparapu, An unsu-
pervised frame selection technique for robust emotion recognition in noisy
speech, in: Proceedings of the 2018 26th European Signal Processing Con-
ference (EUSIPCO), IEEE, Rome, 2018, pp. 2055–2059.
[14] L. Tian, C. Lai, J. D. Moore, Recognizing emotions in dialogues with
disfluencies and non-verbal vocalisations, in: Proceedings of FInterdis-
ciplinary Workshop on Laughter and Other Non-Verbal Vocalisations in
Speech 2015, IEEE, Institute of Electrical and Electronics Engineers, 2015,
pp. 39–41.
[15] Y. Li, Y. Mohamied, P. Bell, C. Lai, Exploration of a self-supervised speech
model: A study on emotional corpora, in: Proceedings of the 2022 IEEE
Spoken Language Technology Workshop (SLT), IEEE, Doha, Qatar, 2023,
pp. 868–875.
[16] L.-W. Chen, A. Rudnicky, Exploring wav2vec 2.0 fine tuning for improved
speech emotion recognition, in: Proceedings of the 2023 IEEE Interna-
tional Conference on Acoustics, Speech and Signal Processing (ICASSP),
IEEE, Rhodes Island, Greece, 2023, pp. 1–5.
[17] S. Schneider, A. Baevski, R. Collobert, M. Auli, Wav2vec: Unsupervised
pre-training for speech recognition, in: Proceedings of Interspeech 2019,
ISCA, 2019, pp. 3465–3469.
[18] A. Baevski, Y. Zhou, A. Mohamed, M. Auli, Wav2vec 2.0: A framework
for self-supervised learning of speech representations, Advances in neural
information processing systems 33 (2020) 12449–12460.
[19] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, A. Mo-
hamed, Hubert: Self-supervised speech representation learning by masked
prediction of hidden units, IEEE/ACM Trans. Audio, Speech, Lang. Pro-
cess. 29 (2021) 3451–3460.
[20] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda,
T. Yoshioka, X. Xiao, J. Wu, L. Zhou, S. Ren, Y. Qian, Y. Qian, J. Wu,
M. Zeng, X. Yu, F. Wei, Wavlm: Large-scale self-supervised pre-training
for full stack speech processing, IEEE J. Sel. Top. Signal Process. 16 (6)
(2022) 1505–1518.
[21] L. Pepino, P. Riera, L. Ferrer, Emotion recognition from speech using
wav2vec 2.0 embeddings, in: Proceedings of Interspeech 2021, ISCA,
2021, pp. 3400–3404.
[22] Y. Xia, L.-W. Chen, A. Rudnicky, R. M. Stern, Temporal context in speech
emotion recognition, in: Proceedings of Interspeech 2021, ISCA, 2021, pp.
3370–3374.
[23] P. Yue, L. Qu, S. Zheng, T. Li, Multi-task learning for speech emotion and
emotion intensity recognition, in: Proceedings of the 2022 Asia-Pacific
Signal and Information Processing Association Annual Summit and Con-
ference (APSIPA ASC), IEEE, Chiang Mai, Thailand, 2022, pp. 1232–
1237.
[24] Y. Li, P. Bell, C. Lai, Fusing asr outputs in joint training for speech emo-
tion recognition, in: Proceedings of 2022 IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP), IEEE, Singapore,
Singapore, 2022, pp. 7362–7366.
[25] E. Mower, A. Metallinou, C.-C. Lee, A. Kazemzadeh, C. Busso, S. Lee,
S. Narayanan, Interpreting ambiguous emotional expressions, in: Proceed-

```
ings of the 2009 3rd International Conference on Affective Computing and
Intelligent Interaction and Workshops, IEEE, Amsterdam, Netherlands,
2009, pp. 1–8.
[26] C. Etienne, G. Fidanza, A. Petrovskii, L. Devillers, B. Schmauch,
Cnn+lstm architecture for speech emotion recognition with data augmen-
tation, in: Proceedings of Workshop on Speech, Music and Mind (SMM
2018), ISCA, 2018, pp. 21–25.
[27] A. Ando, S. Kobashikawa, H. Kamiyama, R. Masumura, Y. Ijima, Y. Aono,
Soft-target training with ambiguous emotional utterances for dnn-based
speech emotion classification, in: Proceedings of the 2018 IEEE Interna-
tional Conference on Acoustics, Speech and Signal Processing (ICASSP),
IEEE, Calgary, AB, 2018, pp. 4964–4968.
[28] Q. Mao, M. Dong, Z. Huang, Y. Zhan, Learning salient features for speech
emotion recognition using convolutional neural networks, IEEE Trans.
Multimedia 16 (8) (2014) 2203–2213.
[29] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recog-
nition, in: Proceedings of the 2016 IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), IEEE, Las Vegas, NV, USA, 2016, pp.
770–778.
[30] K. He, X. Zhang, S. Ren, J. Sun, Identity mappings in deep residual net-
works, in: B. Leibe, J. Matas, N. Sebe, M. Welling (Eds.), Proceedings of
Computer Vision – ECCV 2016, Vol. 9908, Springer International Publish-
ing, Cham, 2016, pp. 630–645.
[31] S. Ioffe, C. Szegedy, Batch normalization: Accelerating deep network
training by reducing internal covariate shift, in: Proceedings of the 32nd
International Conference on Machine Learning, Vol. 37, PMLR, Lille,
France, 2015, pp. 448–456.
[32] J. Hu, L. Shen, G. Sun, Squeeze-and-excitation networks, in: Proceed-
ings of the 2018 IEEE/CVF Conference on Computer Vision and Pattern
Recognition, IEEE, Salt Lake City, UT, 2018, pp. 7132–7141.
[33] L. Sun, B. Liu, J. Tao, Z. Lian, Multimodal cross- and self-attention
network for speech emotion recognition, in: Proceedings of the 2021
IEEE International Conference on Acoustics, Speech and Signal Process-
ing (ICASSP), IEEE, Toronto, ON, Canada, 2021, pp. 4275–4279.
[34] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, I. Polosukhin, Attention is all you need, Advances in neural
information processing systems 30.
[35] C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N.
Chang, S. Lee, S. S. Narayanan, Iemocap: Interactive emotional dyadic
motion capture database, Lang. Resour. Eval. 42 (4) (2008) 335–359.
[36] M. Chen, X. He, J. Yang, H. Zhang, 3-d convolutional recurrent neural net-
works with attention model for speech emotion recognition, IEEE Signal
Process Lett. 25 (10) (2018) 1440–1444.
[37] H.-C. Chou, C.-C. Lee, Every rating matters: Joint learning of subjective
labels and individual annotators for speech emotion classification, in: Pro-
ceedings of the 2019 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), IEEE, Brighton, United Kingdom, 2019,
pp. 5886–5890.
[38] M. A. Pastor, D. Ribas, A. Ortega, A. Miguel, E. Lleida, Cross-corpus
training strategy for speech emotion recognition using self-supervised rep-
resentations, Appl. Sci. 13 (16) (2023) 9062.
[39] S. G. Upadhyay, W.-S. Chien, B.-H. Su, C.-C. Lee, Learning with rater-
expanded label space to improve speech emotion recognition, IEEE Trans.
Affective Comput. (2024) 1–15.
```

# Declaration of interests

## Dear editor,

## The authors declare that they have no conflicts of interest to this work.

## Best regards,

## Xiaoke Li, Zufan Zhang


