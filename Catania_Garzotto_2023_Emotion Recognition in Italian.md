## Speech Emotion Recognition in Italian

## Using Wav2Vec 2.0 and the Novel Crowdsourced

## Emotional Speech Corpus Emozionalmente

## Fabio Catania

## Franca Garzotto

Abstract—Speech emotion recognition (SER) relies on speech
corpora to collect emotional voices for analysis. However, emo-
tions may vary by culture and language, and resources in Italian
are scarce. To address this gap, we launched a crowdsourcing
campaign and obtained Emozionalmente, a corpus of 6902 sam-
ples produced by 431 non-professional Italian speakers verbaliz-
ing 18 sentences expressing the Big Six emotions and neutrality.
We conducted a subjective validation of Emozionalmente by
asking 829 humans to guess the emotion expressed in the audio
clips, achieving an overall accuracy of 66%. Additionally, we fine-
tuned the deep learning wav2vec 2.0 model on Emozionalmente
and achieved good performance, with an accuracy of around 81-
83%. In this paper, we describe the design choices, a descriptive
analysis of the corpus, and the methodology and results of the
behavioral and computational studies conducted on the dataset.
Our work provides an alternative and extensive resource for
linguistic and speech-processing research on the Italian language.

Index Terms—Affective computing, Speech emotion recogni-
tion, Emotional speech corpus, Behavioural study, Computational
study, Wav2Vec 2.0 deep learning model

### I. INTRODUCTION

# S

PEECH Emotion Recognition (SER) consists of recog-
nizing the emotional aspects of speech, regardless of its
semantic content [1]. It involves categorizing emotions into
discrete classes and associating labels with them [2]. The most
common classes in the field of affective computing are known
asthe Big Six[3], [4]: anger, fear, disgust, joy, sadness, and
surprise, which are universally experienced in all cultures [3].
Automatic SER has the potential to enable speech-based
technologies to recognize users’ emotions and respond empa-
thetically [5], such as in call center applications [6], in-car
stress monitoring systems [6], and emotion-aware therapeu-
tic interventions [7]. However, while humans can efficiently
perform SER as part of natural speech communication [8],
automated SER remains an open research question. Further-
more, emotions can vary across cultures and languages [9],
[10], [11], [12], and resources in some cultures and languages
are limited. Italian is underrepresented in speech emotion
research. To the best of our knowledge, the only current
Italian emotional speech corpus is Emovo [13]. It includes
588 audio recordings grabbed in laboratory conditions by 6

F. Catania is with Politecnico di Milano, Italy, MI 20133 ITA (e-mail:
fabio.catania@polimi.it)
F. Garzotto is with Politecnico di Milano, Italy, MI 20133 ITA (e-mail:
franca.garzotto@polimi.it)

```
professional actors acting out 14 sentences expressing the
Big Six emotions and neutrality. These numbers are not that
large considering that Italian is an official language in Italy,
Switzerland, San Marino, and Vatican City and is the second
most widely spoken native language in the European Union
with 67 million speakers (15% of the EU population) [14]. It
urges collecting new emotional resources in Italian to enable
extensive and robust linguistic and speech-processing research
in this language.
To address this gap, we present Emozionalmente, a crowd-
sourced emotional speech corpus containing 6902 samples
recorded by 431 non-professional actors speaking 18 Italian
sentences expressing the Big Six emotions and neutrality.
Emozionalmente is novel compared to Emovo as it includes
a greater variability of speakers, ways of expressing emotions,
and phonemes produced within the sentences. In addition,
Emozionalmente was recorded in more natural, out-of-lab
conditions using non-professional equipment. We report on
two initial studies: a behavioral study to validate the collected
data and measure the human ability to recognize emotions in
Emozionalmente (and Emovo for comparison), and a com-
putational study to measure the ability of a deep learning
model to recognize emotions in Emozionalmente (and Emovo
for comparison). These scores serve as a baseline for future
research and show which of the explored models are valuable
and worth further exploration for automatic SER.
The rest of the paper is organized as follows. Section II
reviews the state-of-the-art resources in the field of SER in
terms of approaches and emotional speech corpora. Given the
importance of Emovo in the context of this paper, we dedicated
a subsection to it. Section III describes the design choices, data
collection process, and the collected dataset of Emozional-
mente. Section IV describes the methodology and results of the
behavioral study conducted to validate Emozionalmente (and
Emovo). Section V details the methodology and results of the
computational study for automatic SER on Emozionalmente
(and Emovo). Section VI discusses the results and limitations
of the proposed work, and Section VII draws the conclusions
and outlines the next steps of our research.
```
### II. STATE OF THE ART

```
A. Emotional speech corpora
Emotional speech corpora are collections of audio samples
containing emotional content, generally labeled with one or
```

more emotional categories. These labels may be assigned by
the sample’s authors based on their intended emotions or by
external evaluators based on their recognition of emotions.
Emotional speech corpora are critical for evaluating and
training human emotional skills in psychology and sociology
studies and developing software applications that synthesize
and detect emotions in computer science. The quality of an
emotional corpus is crucial for ensuring the communicative
effectiveness of its samples and preventing incorrect research
conclusions in various areas. From the literature [15], [16],
[17], [18], it is evident that emotional speech corpora differ
in many aspects. Researchers should carefully consider the
characteristics of the emotional speech corpora when selecting
(and creating) emotional speech corpora to ensure the accuracy
and validity of their research results. The choice may depend
on the specific research questions and objectives. Specifically,
emotional speech corpora can

- include audio recordings with monolingual or multilin-
    gual sentences;
- collect different sets of emotions, such as the Big Six
    emotions [3];
- contain audio recordings uniformly distributed over emo-
    tions (or not), or include audio recordings with a set of
    phrases uniformly verbalized with different emotions (or
    not);
- be obtained through professional or amateurish recording
    tools;
- include speech recorded in a fully-setup environment
    without any noise or in a wild setting;
- contain information about the context where the speech
    was recorded, including a description of the situation
    (e.g., conversational context) or other complementary
    communication channels (e.g., video);
- collect audio recordings with simulated, induced, or nat-
    ural emotions;
- vary in size and include a different number of actors (with
    different ages and genders);
- contain audio recordings by professional or semi-
    professional actors or a generic audience with no acting
    experience.

Table I provides a non-exhaustive overview of some emotional
speech corpora commonly used in computer science research.
Emotional corpora collected into the wild are scarce [17].
Therefore, most of the corpora in this review are recorded in
a fully-setup environment without any noise and exploiting
professional tools to obtain as clean data as possible. Further-
more, some corpora include information from other sources
(e.g., video [19]) useful for understanding the context while
working with an audio signal.
Most existing emotional speech corpora are monolingual,
while few of them are multilingual [15]. Monolingual corpora
help understand how best to represent an emotion in speech
for a specific language and to search for patterns that can
help speech recognition and synthesis in that language. On the
other hand, multilanguage corpora enable researchers to inves-
tigate cross-cultural and cross-language features of emotional
speech. Psychologists have long debated whether emotions are

```
culture- or language-dependent, but results are still scarce and
inconsistent [12], [11].
Most existing corpora are organized by categorical emo-
tions, particularly the Big Six emotions [15]. In [20], [19],
[21], [22], [13], [23], [24], [25], [26], authors took into account
neutrality as a supplementary emotional state. In [20], [23],
[27], [26], [24], just a subset of the Big Six was considered.
Fewer authors used a dimensional model for describing emo-
tions in a continuous space to include emotional states that are
not archetypal but more controlled, weaker, and subtler [28].
Some existing corpora (e.g., [13], [19], [21]) include the
same number of samples for each emotion, although this
distribution does not reflect the actual frequency of emotions
in the real world. Still, these corpora are helpful for controlled
scientific analysis and experiments because balanced emotions
enable a fair consideration for each emotional category. Also,
it is common in many emotional corpora to see the same
sentence expressed with different emotional tones (e.g., [13],
[19], [21]). This approach aims to enable analyses of the
acoustic features of the speech that is purely based on the
emotional content of the sentence and not on its lexical one.
Another solution implies using non-sentences (as in [13]) or
even non-words in the speech data.
Regarding the actors, they are balanced in gender in most
corpora (e.g., [20], [19], [21], [22], [13], [27], [24], [26]), but
there are exceptions (e.g., [23], [25]). Their number varies
widely (e.g., they are 4 in [20] and 100 in [28]). In most cor-
pora, recruited actors are professionals to reduce the possibility
of collecting audio recordings that do not adequately represent
the expected emotions. However, other corpora are by semi-
professional actors (e.g., [20]) or even amateurs (e.g., [28]) to
be partially closer to real-world situations and avoid exagger-
ation in expressing emotions. Although acted emotions tend
to be more exaggerated than real ones [29], the relationship
between the acoustic correlates and the acted emotions does
not seem to contradict that between acoustic correlates and
real emotions [30]. For this reason, emotions in the corpora
are generally acted. In some studies, actors have been asked to
self-induce their emotions before acting to express them better
with their voices. A famous method is the one by the Russian
theatre actor Stanislavski [31], which works on the actor’s
conscious thought to activate sympathetically and indirectly
the other less-controllable psychological processes, such as
emotional experience and subconscious behavior.
Emovo:To the best of our knowledge, Emovo [32] is the
only existing emotional speech corpus in Italian. It consists of
the voices of 6 actors (3F, 3M) who played 14 sentences simu-
lating the Big Six emotions (anger, disgust, fear, joy, sadness,
surprise) plus the neutral state. The corpus exhibits a robust
balance in which every actor performed all 14 sentences across
all seven emotions, resulting in 98 samples per actor. Each
sentence was enacted 42 times, with each emotion expressed
84 times, resulting in a total of 588 audio recordings. The
entire corpus comprises approximately 1835 seconds (¡1 hour)
of recorded material, with an average duration of 3.12 seconds
(SD= 1.36 seconds) per audio recording. Before starting each
recording, actors employed the Stanislavsky method to self-
induce different emotional states recalling the situations in
```

```
TABLE I
OVERVIEW OF THE MAIN EXISTING EMOTIONAL SPEECH CORPORA
Name Language Actorsdemographics Actorsproficiency Emotions Truthfulnessof emotions Linguisticcontent Setting Tools
```
```
DES [20] Danish 2M, 2F Semi-professional
anger, joy,
neutrality, sadness,
surprise
simulated
```
```
2 scripted
single words,
9 scripted
sentences,
2 scripted
paragraphs
```
```
Setup
environment Professional
```
```
SAVEE [19] English 4M Amateur
anger, disgust,
fear, joy, neutrality,
sadness and surprise
simulated 120 scriptedsentences Setupenvironment Professional
```
```
EMO DB [21] German 5M, 5F Professional
anger, disgust,
fear, joy, neutrality,
surprise, sadness
simulated 10 scriptedsentences Setupenvironment Professional
```
```
INTERFACE [22]
```
```
English,
French,
Slovenian,
Spanish
```
```
1M, 1F
per language,
2M, 1F
in English
```
```
Professional
anger, disgust,
fear, joy, neutrality,
sadness, surprise
simulated 175 scriptedsentences Setupenvironment Professional
```
```
EMOVO [13] Italian 3M, 3F Professional
anger, disgust,
fear, joy, neutrality,
sadness, surprise
simulated 14 scriptedsentences Setupenvironment Professional
```
```
RUSLANA [23] Russian 12M, 49F Amateur
anger, fear,
joy, neutrality,
sadness, surprise
simulated 10 scriptedsentences Setupenvironment Professional
eNTERFACE [27] English 34M, 8F Amateur
anger, disgust,
fear, joy,
sadness, surprise
induced Spontaneouscontent Setupenvironment Professional
CASIA [24] Mandarin 4M, 4F Professional
anger, fear,
joy, neutrality,
sadness, surprise
simulated 200 scriptedsentences Setupenvironment Professional
```
```
Belfast
Database [28] English
```
```
100 people
(unspecified
gender)
Amateur
```
```
Emotions are not
set a-priori, but lately
tagged by external
evaluators through
dimensional (activation)
and categorical
(16-24 emotions)
models
```
```
induced Spontaneouscontent Setupenvironment Professional
```
```
Cmotion [25]
```
```
English,
French,
German,
Italian
```
```
39M 12 professionaland 27 amateur
anger, disgust,
fear, joy, neutrality,
sadness, surprise
```
```
simulated
and induced
1 scripted sentence
per language
Setup
environment Professional
```
```
IEMOCAP [26] English 5M, 5F Professional anger, joy,neutrality, sadness simulated
3 scripted plays
and spontaneous
content
```
```
Setup
environment Professional
```
their own life where they intensely felt the emotions. The
audio recordings were performed in a room of the laboratories
of the Fondazione Ugo Bordoni^1 in Rome using professional
equipment and with a sampling frequency of 48 kHz, 16
bits stereo, and .wav format. 9 of the acted sentences are
semantically meaningful and neutral, 3 are nonsense, and 2
are lists of objects. Altogether, they satisfy the following
conditions: (i) inclusion of all the phonemes of the Italian
language, and (ii) inclusion in every sentence of a fair balance
between voiced and unvoiced consonants.

B. Speech emotion recognition pipeline

The traditional approach for SER follows the pipeline
exemplified in Figure 1, consisting of audio pre-processing,
representation, and classification [6].

Fig. 1. Exemplification of the traditional pipeline for speech emotion
recognition: (i) audio pre-processing, (ii) audio representation, and (iii) audio
classification.

1) Audio pre-processing:Audio pre-processing can be con-
ducted to highlight or suppress specific characteristics of
the signal, ultimately facilitating classification. Performing

(^1) https://fub.it/en
zero, one, or multiple operations, in different orders, may
be appropriate, and there are no established guidelines in the
literature on the optimal operations to perform. Following, we
provide some examples, not claiming comprehensiveness.
Noise reduction, which enhances the subjective intelligi-
bility of recorded speech [33], has been found to improve
system performance for speech emotion recognition in noisy
conditions [34]. A typical noise reduction method involves
subtracting an averaged noise spectrum from the noisy signal
spectrum [35], [36].
Voice activity detection (VAD) extracts speech from au-
dio by filtering out unvoiced signals and silence [37]. VAD
systems typically utilize deep learning techniques, but basic
methods use the auto-correlation of the signal by exploiting
the periodic nature of the voice [38].
Data augmentation generates synthetic data from an existing
dataset, potentially improving the generalization capability of a
model [39]. Audio data augmentation includes injecting noise
[40], shifting time [41], changing pitch [42], and adjusting
speed [43] in the original recording.
2) Audio representation: To represent the information in
an audio recording, researchers typically employ a set of
parameters that compactly describe certain audio features and
contain only the necessary information for speech emotion
recognition while discarding extraneous information.
When features are extracted from each frame of the parti-
tioned signal, they describe the audio at the local level [15].
In contrast, if they are computed as statistics of the local
features across the entire audio recording, they are referred
to as global features [15]. Studies have shown that global


features generally outperform local ones regarding emotions
classification accuracy [44], [45], [46], [47]. However, some
researchers argue that global features completely lose the
temporal information present in signals [6] and are only
effective in distinguishing high-arousal emotions (such as
anger, fear, and joy) from low-arousal ones (such as sadness)
[48]. Overall, global features are often preferred due to their
lower computational requirements and time complexity than
local features.
Features used for speech emotion recognition can beexpert-
designedor obtained through deep neural networks. Expert-
designed features are typically a combination of prosodic and
spectral features [49]. Prosodic featurescharacterize large
units of speech, such as syllables, words, phrases, and sen-
tences, and include (i) pitch-related features, (ii) formant
features, (iii) energy-related features, (iv) timing features, and
(v) articulation features [50], [51], [17]. Previous studies have
shown that prosodic features provide a reliable marker of
basic emotions [52], [50], [53], [51], [54]. However, there
are contradictory reports on how they change depending on
emotions. For instance, while the authors in [51] suggest
that a high speaking rate is associated with the emotion
of anger, the authors in [54] have an opposite conclusion.
Spectral featuresare obtained by converting the time-based
audio signal into the frequency domain using the Fourier
Transform. The Mel Frequency Cepstral Coefficients (MFCC)
are the most widely used spectral features, representing the
short-term power spectrum of speech [55]. Spectral features
are also known to be reliable markers of emotions [48]. For
example, it has been reported that utterances with happiness
emotion have high energy at the high-frequency range, while
those with sadness emotion have low energy at the same
range [53], [56]. Many different sets of selected features have
been explored in the literature, such as the ComParE set [57]
and the extended Geneva Minimalistic Acoustic Parameter
Set (eGeMAPS) [58]. However, there is no consensus on
the best feature set for speech emotion recognition, as the
results appear to be data-dependent. Variability in acoustic
features, such as pitch and energy contours, caused by different
sentences, speakers, and speaking styles, can compromise the
generalization of emotion recognition results [53]. In recent
years, deep neural networks (DNNs) have gained consider-
able popularity in audio signal processing for representation
learning. It is about extracting parameters from the audio that,
despite lacking physical meaning and easy intelligibility by
humans, proved to be superior to expert-designed features in
a wide variety of tasks (including speech recognition [59], [60]
and music transcription [60], [61]). Still, the advantages of this
approach have been just scarcely explored for speech emotion
recognition and are to be fully confirmed, yet [62], [63], [64],
[65].
3) Audio classification: After extracting features from
speech data, models for emotion recognition are trained and
tested. Typically, classifier performance is evaluated based on
accuracy, defined as the number of correct predictions divided
by the total number of predictions [66], [6], [67], [15]. In this
subsection, we have chosen not to report on the performance
of the models in the reviewed studies to avoid confusing the

```
reader. Indeed, comparing the performance of different models
is challenging due to various factors such as the use of different
languages, datasets with different emotions, varied sets of
features, diverse model architectures, and training algorithms
[66], [6], [67], [15]. Additionally, speech emotion recognition
can be a speaker-dependent or speaker-independent problem
[68]. Speaker-dependent emotion recognition performs better
than the speaker-independent approach because prior knowl-
edge about the speaker is considered during the recognition
process.
The scientific community has not agreed on the most
suitable classifier for emotion classification [6], [66], [67],
[15]. Each classifier has its strengths and limitations, and some
studies have attempted to combine the potentials of multiple
classifiers [69], [70]. As found in [66], [67], [15], traditional
machine learning classifiers have been employed:
```
- K-nearest neighbor (KNN) [71], [72],
- Random forest [73], [74],
- Support Vector Machine (SVM) [75], [76],
- Linear Discriminant Analysis [77], [78], [79],
- Decision Tree [80],
- Bayes Classifier [81],
- Hidden Markov Model [82], [83], [84],
- Gaussian Mixture Model [83], [85],
- Artificial Neural Network [86], [87].
Moreover, Deep neural networks (DNN) have been used and
have shown to produce better results than classical machine
learning classifiers for emotion recognition [88], [89]. A
promising approach is transfer learning, which involves using
a model created for an auxiliary task (such as speech recog-
nition) that has been trained on large datasets. By replacing
some of its final layers, the model can be used as a feature
extractor or fine-tuned to the task of emotion recognition. This
approach has been successfully employed in studies such as
[62], [63], [64], [65].

### III. EMOZIONALMENTE:

### THE NEWITALIAN EMOTIONAL SPEECH CORPUS

```
A. Dataset design
We created a monolingual corpus of emotional speech in
Italian namedEmozionalmente(meaning ”Emotionally” in En-
glish). Our design choices were made to ensure comparability
with the state-of-the-art Emovo corpus [32].
Similar to Emovo, Emozionalmente:
```
- required actors to simulate emotions,
- included the Big Six emotions (anger, disgust, fear, joy,
    sadness, and surprise) as well as a neutral state, and
- used pre-determined sentences for actors to speak.
The sentences spoken in Emozionalmente are listed below.
Italian version:
S01 Gli operai si alzano presto
S02 La cascata fa molto rumore
S03 Vorrei il numero telefonico del Signor Piatti
S04 Non sapevo che fosse in citt`a
S05 L’ho incontrato oggi dopo 2 anni
S06 Zia Marta ha detto che devo stare a casa stasera


```
S07 Ho preso 6 nella verifica di matematica
S08 Tommaso ha detto che dovevo scegliere io cosa fare
S09 Il capo mi ha affidato un altro lavoro
S10 Torner`a a casa presto
S11 Vado in biblioteca
S12 E’ una notte stellata
S13 Oggi c’e una partita di basket`
S14 E’ impegnato in una riunione
S15 E’ andato a scuola dopo pranzo
S16 Il cane ha riportato qui la palla
S17 Giovanni parte per Roma domani
S18 Ieri un gatto ha bevuto dalla tazza
```
English translation:

S01 Workers get up early
S02 The waterfall makes a lot of noise
S03 I would like the telephone number of Mr. Piatti
S04 I didn’t know they were in town
S05 I saw him today after two years
S06 Aunt Marta said I should stay home tonight
S07 I got a 6 in the math test
S08 Tommaso said it was up to me
S09 The boss gave me another assignment
S10 She’ll be back soon
S11 I’m going to the library
S12 It is a star filled night
S13 There is a basketball game today
S14 He is in a meeting
S15 He went to school after lunch
S16 The dog brought the ball
S17 Tomorrow Giovanni goes to Rome
S18 Yesterday a cat drunk from the mug
The sentences have been scripted out inspired by three other
sets of sentences in the literature in English and Portuguese
[90], [91], which were constructed ad-hoc to be semantically
neutral and easily readable with different emotional tones. The
sentences include everyday vocabulary and all phonemes of
the Italian language. Indeed, we aimed to create a phonetically
comprehensive repository to ensure that all Italian sounds were
adequately represented in the corpus. On top of that, three sen-
tences from Emovo were also included in the Emozionalmente
sentence set.
In contrast to Emovo, in Emozionalmente

- the actors were not necessarily professionals, but a
    generic audience with no restriction of age or gender;
- the actors were not necessarily required to play all the
    sentences in the different emotions, but they could per-
    form as many sentence-emotion pairings as they wished
    until available;
- speech was not necessarily recorded with professional
    equipment, but with the instrumentation at the disposal
    of the actors;
- recordings were not performed in a noiseless laboratory,
    but wherever the actors wanted;
- actors were not paid for their performance but sponta-
    neously recorded their voice through a crowdsourcing
    web app, as commonly happens for the construction of

```
many other datasets for different application domains for
machine learning [92] (see Section III-B).
```
```
B. Data collection
We have been collecting the data of Emozionalmente from
the 9thof February 2021 to the 7thof June 2021. We used a
web app to collect it via crowdsourcing since these apps are
easy to share, and can be accessed from different commercial
devices (e.g., smartphones, tablets, laptops) and in different
contexts with the critical constraint of having an Internet
connection. In addition, we chose a web app because it does
not require any installation or configuration by the user. We
evaluated some existing crowdsourcing web platforms (i.e.,
Amazon Mechanical Turk^2 and Prolific^3 ), but we decided to
develop a custom solution to offer a personalized experience
to our users: https://emozionalmente.i3lab.group/. We shared
the link through social media and local and Italian national
press. Here is how the web app works. As shown in Figure
2A, on the app’s homepage, the user can find some insights
about our motivation for collecting emotional speech samples
and instructions on how to take part in the corpus creation.
The homepage provides also a link to a tutorial that explains
how to access each functionality in the web app. Before
starting the data collection process, users must accept informed
consent dealing with the experimental procedure and the use
of data. They are also asked to enter their age, gender, and
native language (see Figure 2B). Once the user enters their
demographic information, they can access the audio recording
page (see Figure 2C). Each user can record at most 126 audios
(18 sentences expressed in the seven emotional states). The
app assigns the user the sentence to verbalize and the emotion
to express at each turn. Sentences and emotions are chosen
randomly, preventing a user from acting out the same sentence-
emotion pair more than once and balancing the number of
audio recordings collected by the platform for each emotion.
The user is asked to self-induce the assigned emotional state,
recalling situations in their lives where they intensely felt that
emotion. When the user is ready, they can click or press
the recording button; the app starts recording their voice
through the microphone of the running device and continues
until they click or press it again when their performance is
finished. The user can listen to the audio they just recorded.
If they are satisfied with their acting performance, they can
submit it; otherwise, they can discard it and retry. In addition,
a section on the web app allows users to validate audio
recordings by other users (see Figure 2D). In other words,
every user can listen to the speeches by others (they cannot
listen to their own) and assign (i) a score for noisiness (i.e.,
cleanornoisy) and (ii) a label describing the emotion they
recognize.Noisyis selected when the words in the speech
do not match with the words in the sentence assigned to the
actor or when the audio is so noisy that it prevents a clear
understanding of the speech in the audio;cleanotherwise. We
are aware that a binary ”clean”/”noisy” selection may seem
rather limiting for some aspects, and we could have gone for
```
(^2) https://mturk.com/
(^3) https://prolific.co/


Fig. 2. A panel of the web pages for crowd-sourcing Emozionalmente. (A) The home page with the general information about the data collection and the
Emozionalmente project. (B) The web page for inserting the demographic information of the user. (C) The audio recording web page, where the users can
record their voice while verbalizing the required sentences. (D) The audio validation web page, where the users can listen to the audios by other participants,
assign a score for the noisiness of the recordings (i.e., ”clean” or ”noisy”), and guess the emotion they recognize in each audio. (E) The web page showing
the aggregated statistics and performance of the user. (F) The web page showing the aggregated statistics and performance of all the users.

a finer granularity evaluation scale, but we preferred not to
include other options to facilitate users’ decision-making and
speed up the validation process. Emotional labels consist of
the Big Six plus the neutral state. Although more than one
emotion may be perceived in speech, the user must select
the more significant one. Audio recordings to play are chosen

```
by the system so as to collect a balanced number of audio
evaluations in total. Users are not given any speaker-dependent
training. Users can listen to an audio recording as often as they
want if needed. Last, as depicted in Figure 2E and Figure 2F,
the user can check some aggregated data about their acting
performance and the data collected in total in Emozionalmente.
```

Fig. 3. Demographics of the actors of Emozionalmente Fig. 4. Demographics of the evaluators of Emozionalmente

C. Data cleaning

We cleaned the collected data to prepare it for the following
speech emotion recognition analysis. In particular, we filtered
out all the audio recordings

- played by an actor whose mother tongue is not Italian so
    that we can assume that there is none or little cultural
    difference among the actors and the evaluators in the
    corpus as suggested in [6];
- evaluated by a listener whose mother tongue is not Italian
    for the same reason as above;
- evaluated less than 5 times (since they received at most
    5 evaluations);
- flagged as noisy more than half (i.e., 3 or more) of the
    times;
- that were never evaluated with the same emotion that the
    actor was supposed to express and consequently – as our
    assumption - do not properly represent it.

Before data cleaning, the corpus consisted of 11404 samples.
The definitive version of the dataset is described in the next
section.

D. Data description

Emozionalmente counts 6902 samples. It is the second
speech emotional corpus in the Italian language by creation
date but the first by size. Audio recordings last 26297 seconds
(7+ hours) in total, which means that each audio recording
lasts on average 3.81 seconds (SD= 0.99). Recordings were
generally obtained with non-professional equipment and have
2 channels (stereo), a sample size of 16 bits, and a .wav format.
6839 audio recordings were obtained with a sampling rate of
48 kHz and 63 of them with 44.1 kHz, depending on the
characteristics of the recording device. Recordings have been
performed by 431 actors, whose mother language, gender, and
age are depicted in Figure 3. They are all Italian. 131 are
males, 299 females, e 1 listed themselves as “other”. They
have an average age of approximately 31 years old, with a
standard deviation of 12. The corpus is not much balanced

```
concerning the audio recordings by the actors: each actor
performed on average 16 sentences (SD= 22). Still, emotions
were expressed uniformly (986 times each), and every sentence
was verbalized 383 times on average (SD=15).
```
### IV. STUDY1: HUMAN SPEECH EMOTION RECOGNITION

### ONEMOZIONALMENTE

```
A. Methodology
1) Emozionalmente: Study 1 aims to evaluate the ef-
fectiveness of Emozionalmente in conveying emotions. To
achieve this, a subjective evaluation of the emotional content
of the audio recordings was conducted. The evaluation was
carried out through crowdsourcing, utilizing the ad-hoc web
application described in Section III-B. The particulars of the
participants involved in the study are explicated in Section
IV-B1. Participants were instructed to listen to select audio
recordings from Emozionalmente, evaluate them as either
noisyorclean, and predict the emotion conveyed by the actor’s
voice.
2) Emovo: To provide a basis for comparison, we have
also checked Emovo’s effectiveness in conveying emotions.
Five male Italian participants aged 24 to 29 were recruited
to accomplish this. Each participant listened to all the audio
recordings in Emovo, rated the perceived noisiness of the
clips, and guessed the emotions expressed by the actors
with the voice. Participants used the same web application
previously employed to collect the tags for validating the data
in Emozionalmente.
```
```
B. Results
1) Emozionalmente:The Emozionalmente evaluators from
the crowdsourcing campaign were 829 in total and are dis-
tributed as depicted in Figure 4. They are all Italian. 282 are
males, 540 are females, and 7 listed themselves as “other”.
They have an average age of 29 (SD= 10). A total of 34510
evaluations were completed, 5 per audio. Among these, 1310
categorized the audio as noisy and 33200 as clean.
```

Fig. 5. Heatmap describing the accuracy (as decimal representations of
percentages) by human evaluatorsrecognizing the emotions expressed in the
audio recordings of Emozionalmente

```
Fig. 6. Heatmap describing the accuracy (as decimal representations of
percentages) of human evaluatorsrecognizing the emotions expressed in the
audio recordings of Emovo
```
In addition, by considering the emotional intention of the
actors as the ground truth and by defining theaccuracyas
the number of correct predictions divided by the total number
of guesses, evaluators had an overall accuracy in recognizing
emotions of 66%. More details regarding the accuracy relative
to each individual emotion can be found in Figure 5.
2) Emovo:All audio recordings in Emovo were assessed as
“clean” by 5 out of 5 evaluators. In addition, evaluators had
an overall accuracy in recognizing emotions of 67%. More
details regarding the accuracy relative to each emotion can be
found in Figure 6.

### V. STUDY2: AUTOMATIC SPEECH EMOTION RECOGNITION

### ONEMOZIONALMENTE

A. Methodology

1) Emozionalmente: Study 2 is to measure the ability of
a deep learning model to recognize emotions on Emozional-
mente. We used the wav2vec 2.0 model [93] (initially created
for speech recognition) to extract contextualized representa-
tions from the raw audio signal, and we fine-tuned it for
speech emotion recognition. Wav2vec 2.0 is a transformer-
based model. Its architecture comprises three sub-modules:
feature encoder, quantization module, and transformer module.
The feature encoder is a multi-layer CNN that extracts low-
level features from the input signal. These features are fed
into the quantization module and the transformer module. The
quantization module produces multilingual quantized speech
units whose embeddings are then used as targets for the
transformer module. The transformer module creates contex-
tualized representations of the audio. Although wav2vec 2.
was originally created to recognize speech, it can be used for
other tasks by adding a simple neural network on the top of it
to perform the desired task (e.g., speaker verification [94],
mispronunciation detection [95], emotion recognition [62],
[63], [64], [65]) We adopted a pre-trained wav2vec 2.0 model
checkpoint from Hugging Face^4. It was fine-tuned on Italian
using the Common Voice dataset^5 showing a final word error
rate (WER) of 9.36%. Since wav2vec 2.0 cannot naturally

(^4) https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian
(^5) https://commonvoice.mozilla.org/it/datasets
form a sentence representation [93], we had to fine-tune it
on utterance-level classification tasks. We experimented with
different architectures to put as a final layer of the model.
Due to its simplicity, we finally opted for processing the
contextualized representation extracted by wav2vec 2.0 with a
global average pooling across the time dimension (to handle
audio recordings of any length), the ReLU activation, and a
single linear layer to predict the emotion categories as in [65].
First, we split Emozionalmente into training and test sets in
two different ways to run the speaker-dependent and speaker-
independent experiments. In the first case (speaker-dependent),
we split Emozionalmente into a training set (80%) and a test
set (20%), paying attention to keep the classes balanced. In
the second case (speaker-independent), we made sure that the
train and test sets had a comparable number of males and
females, that the test set included at least 20% of the total
number of samples, and that the actors in the training set were
not included in the test set. We performed hyperparameters
tuning using Optuna^6 to optimize the training rate and the
number of epochs to train our model. Once the tuning phase
was complete, we trained each classifier on the training set
with the best parameters found, and we tested them on the
test set.
2) Emovo: For a matter of comparison, we iterated the
same procedure described above on Emovo.
B. Results
C. Emozionalmente
After fine-tuning the wav2vec 2.0 model on the training set
of Emozionalmente, we assessed its accuracy in recognizing
emotions on the test set, both in the speaker-dependent (see
Figures 7) and speaker-independent (see Figures 8) cases. In
the speaker-dependent case, the model reached an accuracy
of 83% on the test set. In the speaker-independent case, the
model reached an accuracy of 81% on the test set.
D. Emovo
Likewise, we report the accuracy of the wav2vec 2.0 models
that we fine-tuned on Emovo both in the speaker-dependent
(^6) https://optuna.readthedocs.io/en/stable/


Fig. 7. Heatmap describing the accuracy (as decimal representations of
percentages) of the fine-tuned wav2vec 2.0 modelrecognizing the emotions
expressed in Emozionalmentein a speaker-independentcase.

```
Fig. 8. Heatmap describing the accuracy (as decimal representations of
percentages) of the fine-tuned wav2vec 2.0 modelrecognizing the emotions
expressed in Emozionalmentein a speaker-dependentcase.
```
Fig. 9. Heatmap describing the accuracy (as decimal representations of
percentages) of the fine-tuned wav2vec 2.0 modelrecognizing the emotions
expressed in Emovoin a speaker-independentcase.

```
Fig. 10. Heatmap describing the accuracy (as decimal representations of
percentages) of the fine-tuned wav2vec 2.0 modelrecognizing the emotions
expressed in Emovoin a speaker-dependentcase.
```
(see Figures 10) and speaker-independent (see Figures 9) case.
Specifically, in the speaker-dependent case, the model reached
an accuracy of 70% on the test set. In the speaker-independent
case, the model reached an accuracy of 30% on the test set.

### VI. DISCUSSION

Human evaluators had similar performance in recognizing
emotions in Emozionalmente (66%) and Emovo (67%). They
outperformed the accuracy of a random classifier (14%), and
their performance was comparable with that found in the lit-
erature on other datasets validated by humans as a seven-class
problem [96]. This result suggests that both datasets convey
emotions with good and similar communication effectiveness.
Also, it suggests that including non-professional actors does
not significantly impact the effectiveness of conveying the
emotions of the collected data. Still, we observed some diver-
gence in the recognition accuracy of different emotions. For
example, disgust was the least accurately recognized emotion
in both Emozionalmente (52%) and Emovo (54%). In both
corpora, the second-last position was taken by fear (54% in
Emozionalmente, 61% in Emovo). In Emozionalmente, the
most accurately recognized emotion was neutrality (84%).
In Emovo, anger was highest (88%), followed by neutrality
(81%). Different accuracy for different emotions might suggest
an intrinsic difficulty for humans to express and recognize

```
specific emotions. In this sense, disgust and fear were also
proven to be hard to distinguish in previous studies [68].
```
```
Regarding automatic speech emotion recognition, the
wav2vec 2.0 model we fine-tuned on Emozionalmente had
a significantly higher accuracy than the one fine-tuned on
Emovo. This difference in performance can be attributed to
the difference in size and variability offered by the two
datasets. Indeed, Emozionalmente includes many more audio
clips (6902 vs. 588) and actors (431 vs. 6) than Emovo
and thus ensures much more acoustic variety during training.
Deep-learning models are notoriously data-hungry; larger,
more heterogeneous data generally leads to more robust
models. Moreover, the wav2vec 2.0 model we fine-tuned
on Emozionalmente performed better in a speaker-dependent
scenario (83%) than in a speaker-independent one (81%). The
results align with the literature based on deep-learning rep-
resentations [62], [63], [64], [65]. Also, these results showed
once more the effectiveness of Emozionalmente in conveying
emotions and its potential as a corpus for linguistic and
speech processing research. Indeed, even the worst model
trained on Emozionalmente (the speaker-independent model)
showed high performance (higher than humans: 81% vs. 66%).
Moreover, the model had balanced accuracy on the different
classes (see Figure 8), and this was not the case with human
listeners 5. Results show that working on Emozionalmente for
```

training a speech emotion recognition model might produce
valuable classifiers that might be integrated into emotion-
aware technology. The disadvantage of using a deep learning
neural network to extract a representation of speech is that
the features generated do not have an apparent physical
meaning associated with them and, thus, are more difficult to
understand. Further investigation is required in this direction
to interpret them.

A. Limitations

Emozionalmente, the speech emotional corpus we designed
and collected, has some limitations that offer interesting av-
enues for improvement. Below, we discuss these limitations
and provide some future work directions.
As is true for most of the existing speech emotional corpora
(e.g., EMOVO [13], EMODB [21], SAVEE [19]), the biggest
limitation of Emozionalmente may be that we did not collect
speech in a natural context and emotions are acted out. We
know that simulated emotions tend to be more exaggerated
than real ones [29]. Thus, if we want to study emotion
synthesis and recognition in humans and machines based on
the analysis of Emozionalmente, the results could not be
directly generalized to a natural context. However, a further
study should be performed to investigate in that direction.
The dataset is somewhat unbalanced: most actors did not
perform all possible phrase-emotion pairings, and phrases were
not acted in perfectly balanced numbers.
In addition, the audio recordings in Emozionalmente are
generally dirtier than those of other corpora recorded in a
fully-setup environment without any noise and exploiting pro-
fessional tools (e.g., Emovo). While this represents a limitation
for the dataset, it is also closer to a real context.
Through crowdsourcing, we were able to recruit much more
actors than most of the emotional speech corpora in the
literature (see Table I) and had the opportunity to capture
many insights into the way emotions are expressed. However,
we noticed that the actors in Emozionalmente are not equally
distributed by gender and age. This issue paves the ground
for enlarging the dataset and including more people from the
groups that are less represented so far (e.g., youngsters, elderly,
males).
Another limitation of Emozionalmente is that, unlike most
corpora in the literature, actors are not professionals, which
increases the risk that the audio recordings do not properly
represent the labeled emotions. Still, our evaluators recorded
an overall accuracy in recognizing emotions of 66% with some
noticeable variance in recognizing different emotional states
(for example, disgust and fear were recognized with lower
accuracy than the other emotions). The achieved accuracy is
high compared to the chance level (100% / 7 classes = 14.29%)
and is in line with the accuracy of humans in recognizing
emotions in other monolingual and acted datasets (e.g., 66.5%
in [19]). This suggests that audio recordings generally convey
emotions in an acceptable suitable manner.
Regarding the studies’ procedural limitations, we assumed
that speakers could generally communicate emotions with
voice, and we considered their communicative intention (i.e.,

```
the emotion they wanted to convey in each audio) as ground
truth for our study. This assumption is critical in the case
of Emozionalmente, where the actors are amateurs, while it is
safer in the case of Emovo, where the actors are professionals.
To overcome this issue, in Emozionalmente, we filtered out
the audio recordings that were never evaluated with the same
emotion that the actor was supposed to express. Still, this
measure might not be enough. Indeed, another limitation of the
study is that we evaluated the baseline of humans with only
five evaluations per audio. Moreover, we have no evidence that
participants are generallygoodlisteners regarding emotions.
To address these limitations, we plan to have more people
validating Emovo and Emozionalmente and repeat the study,
including only the audio clips in which the actor and at
least 50% of the listeners agree on the emotion conveyed.
Meantime, the results obtained can be considered valid but
still preliminary.
Last, the audio recordings in Emozionalmente were not
obtained under ideal laboratory conditions as those of Emovo,
and we did not address this issue in this study. Data cleaning
(e.g., removing the background noise) and data augmentation
(e.g., injecting different background noises) should be further
explored as future work to improve the accuracy of the
classifiers trained on Emozionalmente. In our defense, we can
say that looking at the results, the noise in Emozionalmente
does not seem to impact the models’ effectiveness in terms of
performance.
```
### VII. CONCLUSION

```
In conclusion, Emozionalmente represents a valuable re-
source for research in spoken language communication and
emotion recognition. The corpus provides a diverse range
of emotional expressions and was collected in natural, non-
laboratory conditions, making it a more ecologically valid
resource than previous emotional speech corpora for Italian.
The accuracy of the subjective evaluations by human listeners
(66%) and the fine-tuned wav2vec 2.0 deep learning model
(81-83%) shows that the dataset is reliable and valid for future
research in automatic speech emotion recognition. Further-
more, our findings suggest that crowdsourcing can be a viable
and cost-effective method for collecting emotional speech
data in other languages. The wav2vec 2.0 model fine-tuned
on Emozionalmente outperformed humans in speech emotion
recognition, indicating the potential for its integration into
conversational agents for greater awareness of the surrounding
emotional context. Future work could investigate the use
of Emozionalmente in other applications, such as affective
computing and human-robot interaction.
```
### REFERENCES

```
[1] B. W. Schuller, “Speech emotion recognition: Two decades in a nutshell,
benchmarks, and ongoing trends,”Commun. ACM, vol. 61, no. 5, p.
90–99, apr 2018. [Online]. Available: https://doi.org/10.1145/
[2] S. PS and G. Mahalakshmi, “Emotion models: a review,”International
Journal of Control Theory and Applications, vol. 10, pp. 651–657, 2017.
[3] P. Ekman, “An argument for basic emotions,”Cognition & emotion,
vol. 6, no. 3-4, pp. 169–200, 1992.
[4] Q. Yao, “Multi-sensory emotion recognition with speech and facial
expression, 2014. copyright-proquest,”Graduate School, p. 133, 2014.
```

[5] R. W. Picard,Affective computing. MIT press, 2000.
[6] M. El Ayadi, M. S. Kamel, and F. Karray, “Survey on speech emotion
recognition: Features, classification schemes, and databases,”Pattern
recognition, vol. 44, no. 3, pp. 572–587, 2011.
[7] C. Lisetti, R. Amini, U. Yasavur, and N. Rishe, “I can help you change!
an empathic virtual agent delivers behavior change health interventions,”
ACM Transactions on Management Information Systems (TMIS), vol. 4,
no. 4, pp. 1–28, 2013.
[8] D. A. Sauter, F. Eisner, P. Ekman, and S. K. Scott, “Cross-cultural recog-
nition of basic emotions through nonverbal emotional vocalizations,”
Proceedings of the National Academy of Sciences, vol. 107, no. 6, pp.
2408–2412, 2010.
[9] B. Mesquita and N. H. Frijda, “Cultural variations in emotions: a
review.”Psychological bulletin, vol. 112, no. 2, p. 179, 1992.
[10] A. P. Fiske, S. Kitayama, H. R. Markus, and R. E. Nisbett, “The cultural
matrix of social psychology.”The handbook of Social Psychology, 1998.
[11] M. D. Pell, L. Monetta, S. Paulmann, and S. A. Kotz, “Recognizing
emotions in a foreign language,”Journal of Nonverbal Behavior, vol. 33,
no. 2, pp. 107–120, 2009.
[12] H. A. Elfenbein and N. Ambady, “On the universality and cultural speci-
ficity of emotion recognition: a meta-analysis.”Psychological bulletin,
vol. 128, no. 2, p. 203, 2002.
[13] G. Costantini, I. Iaderola, A. Paoloni, and M. Todisco, “Emovo corpus:
an italian emotional speech database,” inInternational Conference on
Language Resources and Evaluation (LREC 2014). European Language
Resources Association (ELRA), 2014, pp. 3501–3504.
[14] E. C. Eurobarometer, “Special eurobarometer 386 - europeans and
their languages report,” 2012. [Online]. Available: https://europa.eu/
eurobarometer/surveys/detail/
[15] M. B. Akc ̧ay and K. O ̆guz, “Speech emotion recognition: Emotional
models, databases, features, preprocessing methods, supporting modali-
ties, and classifiers,”Speech Communication, vol. 116, pp. 56–76, 2020.
[16] R. P. Gadhe, R. Shaikh Nilofer, V. Waghmare, P. Shrishrimal, and
R. Deshmukh, “Emotion recognition from speech: a survey,”Interna-
tional journal of scientific & engineering research, vol. 6, no. 4, pp.
632–635, 2015.
[17] C. M. Lee and S. S. Narayanan, “Toward detecting emotions in spoken
dialogs,”IEEE transactions on speech and audio processing, vol. 13,
no. 2, pp. 293–303, 2005.
[18] N. Campbell, “Databases of emotional speech,” inISCA Tutorial and
Research Workshop (ITRW) on Speech and Emotion, 2000.
[19] S. Haq and P. J. Jackson, “Multimodal emotion recognition,” inMachine
audition: principles, algorithms and systems. IGI Global, 2011, pp.
398–423.
[20] I. S. Engberg and A. V. Hansen, “Documentation of the danish emo-
tional speech database des,”Internal AAU report, Center for Person
Kommunikation, Denmark, vol. 22, 1996.
[21] F. Burkhardt, A. Paeschke, M. Rolfes, W. F. Sendlmeier, B. Weisset al.,
“A database of german emotional speech.” inInterspeech, vol. 5, 2005,
pp. 1517–1520.
[22] V. Hozjan, Z. Kacic, A. Moreno, A. Bonafonte, and A. Nogueiras,
“Interface databases: Design and collection of a multilingual emotional
speech database.” inLREC, 2002.
[23] V. Makarova and V. A. Petrushin, “Ruslana: A database of russian
emotional utterances,” inSeventh international conference on spoken
language processing, 2002.
[24] J. T. F. L. M. Zhang and H. Jia, “Design of speech corpus for mandarin
text to speech,” inThe Blizzard Challenge 2008 workshop, 2008.
[25] A. Origlia, V. Galat`a, and B. Ludusan, “Automatic classification of emo-
tions via global and local prosodic features on a multilingual emotional
database,” inSpeech Prosody 2010-Fifth International Conference, 2010.
[26] C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N.
Chang, S. Lee, and S. S. Narayanan, “Iemocap: Interactive emotional
dyadic motion capture database,”Language resources and evaluation,
vol. 42, no. 4, pp. 335–359, 2008.
[27] O. Martin, I. Kotsia, B. Macq, and I. Pitas, “The enterface’05 audio-
visual emotion database,” in22nd International Conference on Data
Engineering Workshops (ICDEW’06). IEEE, 2006, pp. 8–8.
[28] E. Douglas-Cowie, R. Cowie, and M. Schroder, “A new emotion ̈
database: considerations, sources and scope,” inISCA tutorial and
research workshop (ITRW) on speech and emotion, 2000.
[29] J. Wilting, E. Krahmer, and M. Swerts, “Real vs. acted emotional
speech.” inInterspeech, vol. 2006, 2006, p. 9th.
[30] C. E. Williams and K. N. Stevens, “Emotions and speech: Some
acoustical correlates,”The Journal of the Acoustical Society of America,
vol. 52, no. 4B, pp. 1238–1250, 1972.

```
[31] S. Moore,The Stanislavski system: The professional training of an actor.
Penguin, 1984.
[32] G. Costantini, I. Iaderola, A. Paoloni, and M. Todisco, “Emovo corpus:
an italian emotional speech database,” inInternational Conference on
Language Resources and Evaluation (LREC 2014). European Language
Resources Association (ELRA), 2014, pp. 3501–3504.
[33] P. C. Loizou,Speech enhancement: theory and practice. CRC press,
2007.
[34] J. Pohjalainen, F. Fabien Ringeval, Z. Zhang, and B. Schuller, “Spectral
and cepstral audio noise reduction techniques in speech emotion recog-
nition,” inProceedings of the 24th ACM international conference on
Multimedia, 2016, pp. 670–674.
[35] M. Berouti, R. Schwartz, and J. Makhoul, “Enhancement of speech cor-
rupted by acoustic noise,” inICASSP’79. IEEE International Conference
on Acoustics, Speech, and Signal Processing, vol. 4. IEEE, 1979, pp.
208–211.
[36] S. Boll, “Suppression of acoustic noise in speech using spectral subtrac-
tion,”IEEE Transactions on acoustics, speech, and signal processing,
vol. 27, no. 2, pp. 113–120, 1979.
[37] N. H. Tandel, H. B. Prajapati, and V. K. Dabhi, “Voice recognition
and voice comparison using machine learning techniques: A survey,”
in2020 6th International Conference on Advanced Computing and
Communication Systems (ICACCS). IEEE, 2020, pp. 459–465.
[38] J. K. Lee and C. D. Yoo, “Wavelet speech enhancement based on
voiced/unvoiced decision,”Proceedings of the Korean Society of Noise
and Vibration Engineering International Conference, pp. 4149–4156,
2003.
[39] A. Chatziagapi, G. Paraskevopoulos, D. Sgouropoulos, G. Pantazopou-
los, M. Nikandrou, T. Giannakopoulos, A. Katsamanis, A. Potamianos,
and S. Narayanan, “Data augmentation using gans for speech emotion
recognition.” inInterspeech, 2019, pp. 171–175.
[40] R. Pappagari, T. Wang, J. Villalba, N. Chen, and N. Dehak, “x-vectors
meet emotions: A study on dependencies between emotion and speaker
recognition,” inICASSP 2020-2020 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp.
7169–7173.
[41] B. A. Prayitno and S. Suyanto, “Segment repetition based on high
amplitude to enhance a speech emotion recognition,”Procedia Computer
Science, vol. 157, pp. 420–426, 2019.
[42] J. Salamon and J. P. Bello, “Deep convolutional neural networks and
data augmentation for environmental sound classification,”IEEE Signal
processing letters, vol. 24, no. 3, pp. 279–283, 2017.
[43] Z. Aldeneh and E. M. Provost, “Using regional saliency for speech emo-
tion recognition,” in2017 IEEE international conference on acoustics,
speech and signal processing (ICASSP). IEEE, 2017, pp. 2741–2745.
[44] D. Ververidis and C. Kotropoulos, “Emotional speech classification using
gaussian mixture models and the sequential floating forward selection
algorithm,” in2005 IEEE International Conference on Multimedia and
Expo. IEEE, 2005, pp. 1500–1503.
[45] H. Hu, M.-X. Xu, and W. Wu, “Fusion of global statistical and segmental
spectral features for speech emotion recognition.” inINTERSPEECH,
2007, pp. 2269–2272.
[46] M. T. Shami and M. S. Kamel, “Segment-based approach to the recog-
nition of emotions in speech,” in2005 IEEE International Conference
on Multimedia and Expo. IEEE, 2005, pp. 4–pp.
[47] R. W. Picard, E. Vyzas, and J. Healey, “Toward machine emotional in-
telligence: Analysis of affective physiological state,”IEEE transactions
on pattern analysis and machine intelligence, vol. 23, no. 10, pp. 1175–
1191, 2001.
[48] T. L. Nwe, S. W. Foo, and L. C. De Silva, “Speech emotion recognition
using hidden markov models,”Speech communication, vol. 41, no. 4,
pp. 603–623, 2003.
[49] K. Sailunaz, M. Dhaliwal, J. Rokne, and R. Alhajj, “Emotion detection
from text and speech: a survey,”Social Network Analysis and Mining,
vol. 8, no. 1, pp. 1–26, 2018.
[50] R. Cowie, E. Douglas-Cowie, N. Tsapatsoulis, G. Votsis, S. Kollias,
W. Fellenz, and J. G. Taylor, “Emotion recognition in human-computer
interaction,”IEEE Signal processing magazine, vol. 18, no. 1, pp. 32–80,
2001.
[51] I. R. Murray and J. L. Arnott, “Toward the simulation of emotion in
synthetic speech: A review of the literature on human vocal emotion,”
The Journal of the Acoustical Society of America, vol. 93, no. 2, pp.
1097–1108, 1993.
[52] R. Cowie and E. Douglas-Cowie, “Automatic statistical analysis of the
signal and prosodic signs of emotion in speech,” inProceeding of Fourth
International Conference on Spoken Language Processing. ICSLP’96,
vol. 3. IEEE, 1996, pp. 1989–1992.
```

[53] R. Banse and K. R. Scherer, “Acoustic profiles in vocal emotion
expression.”Journal of personality and social psychology, vol. 70, no. 3,
p. 614, 1996.
[54] A. Oster and A. Risberg, “The identification of the mood of a speaker
by hearing impaired listeners,”SLT-Quarterly Progress Status Report,
vol. 4, pp. 79–90, 1986.
[55] S. Kuchibhotla, H. D. Vankayalapati, R. Vaddi, and K. R. Anne,
“A comparative analysis of classifiers in emotion recognition through
acoustic features,”International Journal of Speech Technology, vol. 17,
no. 4, pp. 401–408, 2014.
[56] L. Kaiser, “Communication of affects by single vowels,”Synthese, pp.
300–319, 1962.
[57] B. Schuller, S. Steidl, A. Batliner, J. Hirschberg, J. K. Burgoon, A. Baird,
A. Elkins, Y. Zhang, E. Coutinho, K. Evaniniet al., “The interspeech
2016 computational paralinguistics challenge: Deception, sincerity &
native language,” in17TH Annual Conference of the International
Speech Communication Association (Interspeech 2016), Vols 1-5, 2016,
pp. 2001–2005.
[58] F. Eyben, K. R. Scherer, B. W. Schuller, J. Sundberg, E. Andr ́e, C. Busso,
L. Y. Devillers, J. Epps, P. Laukka, S. S. Narayananet al., “The geneva
minimalistic acoustic parameter set (gemaps) for voice research and
affective computing,”IEEE transactions on affective computing, vol. 7,
no. 2, pp. 190–202, 2015.
[59] G. Hinton, L. Deng, D. Yu, G. E. Dahl, A.-r. Mohamed, N. Jaitly,
A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainathet al., “Deep neural
networks for acoustic modeling in speech recognition: The shared views
of four research groups,”IEEE Signal processing magazine, vol. 29,
no. 6, pp. 82–97, 2012.
[60] H. Lee, P. Pham, Y. Largman, and A. Ng, “Unsupervised feature learn-
ing for audio classification using convolutional deep belief networks,”
Advances in neural information processing systems, vol. 22, pp. 1096–
1104, 2009.
[61] N. Boulanger-Lewandowski, Y. Bengio, and P. Vincent, “Modeling
temporal dependencies in high-dimensional sequences: Application
to polyphonic music generation and transcription,” arXiv preprint
arXiv:1206.6392, 2012.
[62] J. Boigne, B. Liyanage, and T.Ostrem, “Recognizing more emotions ̈
with less data using self-supervised transfer learning,”arXiv preprint
arXiv:2011.05585, 2020.
[63] Y. Xia, L.-W. Chen, A. Rudnicky, and R. M. Stern, “Temporal context
in speech emotion recognition,” inProc. Interspeech, vol. 2021, 2021,
pp. 3370–3374.
[64] L. Pepino, P. Riera, and L. Ferrer, “Emotion recognition from speech
using wav2vec 2.0 embeddings,”arXiv preprint arXiv:2104.03502,
2021.
[65] L.-W. Chen and A. Rudnicky, “Exploring wav2vec 2.0 fine-
tuning for improved speech emotion recognition,” arXiv preprint
arXiv:2110.06309, 2021.
[66] T. Roy, T. Marwala, and S. Chakraverty, “A survey of classification
techniques in speech emotion recognition,”Mathematical Methods in
Interdisciplinary Sciences, pp. 33–48, 2020.
[67] M. El Ayadi, M. S. Kamel, and F. Karray, “Survey on speech emotion
recognition: Features, classification schemes, and databases,”Pattern
recognition, vol. 44, no. 3, pp. 572–587, 2011.
[68] N. Zaheer, O. U. Ahmad, A. Ahmed, M. S. Khan, and M. Shabbir, “Se-
mour: A scripted emotional speech repository for urdu,” inProceedings
of the 2021 CHI Conference on Human Factors in Computing Systems,
2021, pp. 1–12.
[69] M. Lugger, M.-E. Janoir, and B. Yang, “Combining classifiers with
diverse feature sets for robust speaker independent emotion recognition,”
in2009 17th European Signal Processing Conference. IEEE, 2009, pp.
1225–1229.
[70] B. Schuller, M. Lang, and G. Rigoll, “Robust acoustic speech emotion
recognition by ensembles of classifiers,” inTagungsband Fortschritte
der Akustik-DAGA# 05, M ̈unchen, 2005.
[71] J. Rong, G. Li, and Y.-P. P. Chen, “Acoustic feature selection for
automatic emotion recognition from speech,”Information processing &
management, vol. 45, no. 3, pp. 315–328, 2009.
[72] B. Schuller, S. Reiter, R. Muller, M. Al-Hames, M. Lang, and G. Rigoll,
“Speaker independent speech emotion recognition by ensemble classi-
fication,” in2005 IEEE International Conference on Multimedia and
Expo, 2005, pp. 864–867.
[73] T. Iliou and C.-N. Anagnostopoulos, “Comparison of different classifiers
for emotion recognition,” in2009 13th Panhellenic Conference on
Informatics, 2009, pp. 102–106.
[74] W.-H. Cao, J.-P. Xu, and Z.-T. Liu, “Speaker-independent speech emo-
tion recognition based on random forest feature selection algorithm,”

```
in2017 36th Chinese Control Conference (CCC). IEEE, 2017, pp.
10 995–10 998.
[75] P. Shen, Z. Changjun, and X. Chen, “Automatic speech emotion recogni-
tion using support vector machine,” inProceedings of 2011 International
Conference on Electronic & Mechanical Engineering and Information
Technology, vol. 2. IEEE, 2011, pp. 621–625.
[76] K. Wang, N. An, B. N. Li, Y. Zhang, and L. Li, “Speech emotion
recognition using fourier parameters,”IEEE Transactions on affective
computing, vol. 6, no. 1, pp. 69–75, 2015.
[77] S. Yildirim, M. Bulut, C. M. Lee, A. Kazemzadeh, Z. Deng, S. Lee,
S. Narayanan, and C. Busso, “An acoustic study of emotions expressed
in speech,” inEighth International Conference on Spoken Language
Processing, 2004.
[78] O.-W. Kwon, K. Chan, J. Hao, and T.-W. Lee, “Emotion recognition by
speech signals,” inEighth European Conference on Speech Communi-
cation and Technology, 2003.
[79] S. Mirsamadi, E. Barsoum, and C. Zhang, “Automatic speech emotion
recognition using recurrent neural networks with local attention,” in
2017 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP). IEEE, 2017, pp. 2227–2231.
[80] M. Borchert and A. Dusterhoft, “Emotions in speech-experiments with
prosody and quality features in speech for use in categorical and
dimensional emotion recognition environments,” in2005 International
Conference on Natural Language Processing and Knowledge Engineer-
ing. IEEE, 2005, pp. 147–151.
[81] B. Yang and M. Lugger, “Emotion recognition from speech signals using
new harmony features,”Signal processing, vol. 90, no. 5, pp. 1415–1423,
2010.
[82] A. Nogueiras, A. Moreno, A. Bonafonte, and J. B. Marino, “Speech ̃
emotion recognition using hidden markov models,” inSeventh European
Conference on Speech Communication and Technology, 2001.
[83] B. Schuller, G. Rigoll, and M. Lang, “Hidden markov model-based
speech emotion recognition,” in 2003 IEEE International Confer-
ence on Acoustics, Speech, and Signal Processing, 2003. Proceed-
ings.(ICASSP’03)., vol. 2. IEEE, 2003, pp. II–1.
[84] Y.-L. Lin and G. Wei, “Speech emotion recognition based on hmm
and svm,” in2005 international conference on machine learning and
cybernetics, vol. 8. IEEE, 2005, pp. 4898–4901.
[85] E. M. Albornoz, D. H. Milone, and H. L. Rufiner, “Spoken emotion
recognition using hierarchical classifiers,”Computer Speech & Lan-
guage, vol. 25, no. 3, pp. 556–570, 2011.
[86] R. Nakatsu, J. Nicholson, and N. Tosa, “Emotion recognition and its
application to computer agents with spontaneous interactive capabili-
ties,” inProceedings of the seventh ACM international conference on
Multimedia (Part 1), 1999, pp. 343–351.
[87] B. Schuller, R. Muller, M. Lang, and G. Rigoll, “Speaker independent ̈
emotion recognition by early fusion of acoustic and linguistic features
within ensembles,” inNinth European Conference on Speech Commu-
nication and Technology, 2005.
[88] B. Schuller, G. Rigoll, and M. Lang, “Speech emotion recognition
combining acoustic features and linguistic information in a hybrid
support vector machine-belief network architecture,” in2004 IEEE
International Conference on Acoustics, Speech, and Signal Processing,
vol. 1. IEEE, 2004, pp. I–577.
[89] Q. Mao, M. Dong, Z. Huang, and Y. Zhan, “Learning salient features
for speech emotion recognition using convolutional neural networks,”
IEEE transactions on multimedia, vol. 16, no. 8, pp. 2203–2213, 2014.
[90] V. Nandur, “Performance of” gale” using semantically neutral sen-
tences,” Ph.D. dissertation, University of Florida, 2003.
[91] S. L. Castro and C. F. Lima, “Recognizing emotions in spoken language:
A validated set of portuguese sentences and pseudosentences for research
on emotional prosody,”Behavior Research Methods, vol. 42, no. 1, pp.
74–81, 2010.
[92] K. B. Sheehan, “Crowdsourcing research: data collection with amazon’s
mechanical turk,”Communication Monographs, vol. 85, no. 1, pp. 140–
156, 2018.
[93] A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and M. Auli, “Un-
supervised cross-lingual representation learning for speech recognition,”
arXiv preprint arXiv:2006.13979, 2020.
[94] Z. Fan, M. Li, S. Zhou, and B. Xu, “Exploring wav2vec 2.
on speaker verification and language identification,”arXiv preprint
arXiv:2012.06185, 2020.
[95] L. Peng, K. Fu, B. Lin, D. Ke, and J. Zhan, “A Study on Fine-Tuning
wav2vec2.0 Model for the Task of Mispronunciation Detection and
Diagnosis,” inProc. Interspeech 2021, 2021, pp. 4448–4452.
[96] K. R. Scherer, “Vocal communication of emotion: A review of research
paradigms,”Speech communication, vol. 40, no. 1-2, pp. 227–256, 2003.
```

```
Fabio Cataniareceived his Bachelor and Master of
Science degrees in Computer Science and Engineer-
ing and his Ph.D. in Information Technology from
Politecnico di Milano, Italy. His research interests
include human-computer interaction, conversational
technology, and voice signal processing. He is cur-
rently a postdoctoral researcher at MIT on children’s
voice perception and analysis.
```
Franca Garzottois an associate professor of Com-
puter Engineering in the Department of Electron-
ics, Information, and Bioengineering and leads the
Innovative Interactive Interfaces Lab (https://i3lab.
polimi.it/) at Politecnico di Milano. She received a
master’s degree in mathematics from the University
of Padova and a Ph.D. in Information and Systems
Engineering from Politecnico di Milano. Her current
research focuses on advanced interaction technolo-
gies for people with special needs. In this field,
she coordinates several national and international
research projects and received two IBM Faculty Awards.


