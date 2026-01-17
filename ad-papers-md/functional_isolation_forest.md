arXiv:1904.04573v3 [stat.ML] 9 Oct 2019
Proceedings of Machine learning Research 101:1-33, 2019

# Functional Isolation Forest

**Guillaume Staerman**
GUILLAUME.STAERMAN@TELECOM-PARIS.FR

**Pavlo Mozharovskyi**
PAVLO.MOZHAROVSKYI@TELECOM-PARIS.FR

**Stephan Clémençon**
STEPHAN.CLEMENCON@TELECOM-PARIS.FR

**Florence d’Alché-Buc**
FLORENCE.DALCHE@TELECOM-PARIS.FR

LTCI, Télécom Paris, Institut Polytechnique de Paris

## Abstract

For the purpose of monitoring the behavior of complex infrastructures (e.g. aircrafts, transport or energy networks), high-rate sensors are deployed to capture multivariate data, generally unlabeled, in quasi continuous-time to detect quickly the occurrence of anomalies that may jeopardize the smooth operation of the system of interest. The statistical analysis of such massive data of functional nature raises many challenging methodological questions. The primary goal of this paper is to extend the popular ISOLATION FOREST (IF) approach to Anomaly Detection, originally dedicated to finite dimensional observations, to functional data. The major difficulty lies in the wide variety of topological structures that may equip a space of functions and the great variety of patterns that may characterize abnormal curves. We address the issue of (randomly) splitting the functional space in a flexible manner in order to isolate progressively any trajectory from the others, a key ingredient to the efficiency of the algorithm. Beyond a detailed description of the algorithm, computational complexity and stability issues are investigated at length. From the scoring function measuring the degree of abnormality of an observation provided by the proposed variant of the IF algorithm, a *Functional Statistical Depth* function is defined and discussed, as well as a multivariate functional extension. Numerical experiments provide strong empirical evidence of the accuracy of the extension proposed.

**Keywords:** Anomaly detection, functional data analysis, isolation forest, unsupervised learning

## 1. Introduction

The digital information boom, that goes hand in hand with the recent technological advances in data collection and management (e.g. IoT, distributed platforms), offers new perspectives in many areas of human activity (e.g. transportation, energy, health, commerce, insurance), and confronts these domains with major scientific challenges for exploiting these observations. The ever growing availability of massive data, often collected in quasi-real time, engendered high expectations, in particular the need of increased automation and computational efficiency, with the goal to design more and more ‘intelligent’ systems. In particular, modern high-rate sensors enabling the continuous observation of the behavior of complex systems pave the way for the design of efficient unsupervised machine-learning approaches to anomaly detection, that may find applications in various domains ranging from fraud surveillance to distributed fleet monitoring through predictive maintenance or health monitoring of complex systems. However, although many unsupervised learning pro-

© 2019 G. Staerman, P. Mozharovskyi, S. Clémençon & F. d’Alché-Buc.

cedures for anomaly detection (AD in abbreviated form) have been proposed, analyzed and applied in a variety of practical situations (see, e.g., (Chandola et al., 2009)), the case of functional data, though of crucial importance in practice (refer to (Ramsay and Silverman, 2005; Ferraty and Vieu, 2006) for an account of Functional Data Analysis) has received much less attention in the literature, the vast majority of methods that are documented in the literature being generally model-based. The main barrier to the design of nonparametric anomaly detection techniques tailored to the functional framework lies in the huge diversity of patterns that may carry the information that is relevant to discriminate between abnormal and normal observations, see (Hubert et al., 2015).

It seems indeed far from straightforward to extend machine-learning methods for anomaly detection in the finite-dimensional case such as (Scott and Nowak, 2006; Schölkopf et al., 2001; Steinwart et al., 2005; Vert and Vert, 2006; Park et al., 2010), unless preliminary filtering techniques are used. The filtering approach consists in projecting the functional data onto an adequate finite dimensional function subspace and using then the coefficients describing the latter to "feed" next some AD algorithm for multivariate data (Ramsay and Silverman, 2005). The basis functions are either selected through Principal Component Analysis (they correspond in this case to elements of the Karhunen-Loève basis related to the process under study, supposedly of second order), or else are chosen among a dictionary of "time-frequency atoms" according to their capacity to represent efficiently the data. The representation a priori chosen, which can either enhance artificially certain accessory patterns or else make totally disappear some crucial features, critically determines performance of such an approach, the type of anomalies that can be recovered being essentially shaped by this choice.

The angle embraced in the present article is very different, the goal pursued being to extend the popular ISOLATION FOREST methodology (Liu et al., 2008, 2012) to the functional setup. This ensemble learning algorithm builds a collection of isolation trees based on a recursive and randomized tree-structured partitioning procedure. An isolation tree is a binary tree, representing a nested collection of partitions of the finite dimensional feature space, grown iteratively in a top-down fashion, where the cuts are axis perpendicular and random (uniformly, w.r.t. the splitting direction and the splitting value both at the same time). Incidentally, a variant referred to as EXTENDED ISOLATION FOREST (Hariri et al., 2018), has recently been proposed in the purpose of bias reduction: rather than randomly selecting a perpendicular split, a splitting direction is randomly chosen in the unit ball. An anomaly score is assigned to any observation, depending on the length of the path necessary to isolate it from the rest of the data points, the rationale behind this approach being that anomalies should be easier to isolate in a random manner than normal (in the sense of ‘non-abnormal’) data. Beyond obvious advantages regarding computational cost, scalability (e.g. isolation trees can be built from subsamples) and interpretability, the great flexibility offered by Isolation Forest regarding the splitting procedure called recursively makes it appealing when it comes to isolate (multivariate) functions/curves, possibly exhibiting a wide variety of geometrical shapes. It is precisely the goal of this paper to introduce a new generic algorithm, FUNCTIONAL ISOLATION FOREST (FIF) that generalizes (Extended) Isolation Forest to the infinite dimensional context. Avoiding dimensionality reduction steps, this extension is shown to preserve the assets of the original algorithm concerning computational cost and interpretability. Its efficiency is supported by strong empirical evidence through a variety of numerical results.

The paper is organized as follows. Section 2 recalls the principles under the Isolation Forest algorithm for AD in the multivariate case and introduces the framework we consider for AD based on functional data. In Section 3, the extension to the functional case is presented and its properties are discussed at length. In Section 4, we study the behavior of the new algorithm and compare its performance to alternative methods standing as natural competitors in the functional setup through experiments. In Section 5, extension to multivariate functional data is considered, as well as relation to the data depth function and an application to the supervised classification setting. Eventually, several concluding remarks are collected in Section 6.

## 2. Background and Preliminaries

Here we briefly recall the Isolation Forest algorithm and its advantages (Section 2.1) and next introduce the framework for functional anomaly detection we consider throughout the paper (Section 2.2).

### 2.1. Isolation Forest

As a first go, we describe the Isolation Forest algorithm for AD in the multivariate context in a formalized manner for clarity's sake, as well as the Extended Isolation Forest version, see (Liu et al., 2008, 2012) and (Hariri et al., 2018) respectively. These two unsupervised algorithms can be viewed as Ensemble Learning methods insofar as they build a collection of binary trees and an anomaly scoring function based on the aggregation of the latter. Let S_n = {x_1, ..., x_n} be a training sample composed of n independent realizations of a generic random variable, X, that takes its value in a finite dimensional Euclidian space, R^d say, X = (X^(1), ..., X^(d)).

An isolation tree (itree in abbreviated form) T of depth J ≥ 1 is a proper binary tree that represents a nested sequence of partitions of the feature space R^d. The root node corresponds to the whole space C_{0,0} = R^d, while any node of the tree, indexed by the pair (j, k) where j denotes the depth of the node with 0 ≤ j ≤ J and k, the node index with 0 ≤ k ≤ 2^j − 1, is associated to a subset C_{j,k} ⊂ R^d. A non terminal node (j, k) has two children, corresponding to disjoint subsets C_{j+1,2k} and C_{j+1,2k+1} such that C_{j,k} = C_{j+1,2k} ∪ C_{j+1,2k+1}. A node (j, k) is said to be terminal if it has no children.

Each itree is obtained by recursively filtering a subsample of training data of size ψ in a top-down fashion, by means of the following procedure. The dataset composed of the training observations present at a node (j, k) is denoted by S_{j,k}. At iteration k + 2^j of the itree growing stage, a direction m in {1, ..., d}, or equivalently a split variable X^(m), is selected uniformly at random (and independently from the previous draws) as well as a split value κ in the interval [min_{x∈S_{j,k}} x^(m), max_{x∈S_{j,k}} x^(m)] corresponding to the range of the projections of the points in S_{j,k} onto the m-th axis. The children subsets are then defined by C_{j+1,2k} = C_{j,k} ∩ {x ∈ R^d : x^(m) ≤ κ} and C_{j+1,2k+1} = C_{j,k} ∩ {x ∈ R^d : x^(m) > κ}, the children training datasets being defined as S_{j+1,2k} = S_{j,k} ∩ C_{j+1,2k} and S_{j+1,2k+1} = S_{j,k} ∩ C_{j+1,2k+1}. An itree T is thus built by iterating this procedure until all training data points are isolated (or the depth limit J set by the user is attained). A preliminary subsampling stage can be performed in order to avoid swamping and masking effects, when the size of the dataset is too large. When it isolates any training data point, the itree contains exactly ψ - 1 internal nodes and ψ terminal nodes. An itree constructed accordingly to a training subsample allows to assign to each training datapoint x_i a path length h_T(x_i), namely the depth at which it is isolated from the others, i.e. the number of edges x_i traverses from the root node to the terminal node that contains the sole training data x_i. More generally, it can be used to define an anomaly score for any point x ∈ R^d.

**Anomaly Score prediction.** As the terminal nodes of the itree T form a partition of the feature space, one may then define the piecewise constant function h_T : R^d → N by: ∀x ∈ R^d,
h_T(x) = j if and only if x ∈ C_{j,k} and (j, k) is a terminal node.

This random path length is viewed as an indication for its degree of abnormality in a natural manner: ideally, the more abnormal the point x, the higher the probability that the quantity h_T(x) is small. Hence, the algorithm above can be repeated N ≥ 1 times in order to produce a collection of itrees T_1, ..., T_N, referred to as an iforest, that defines the scoring function

s_n(x) = 2^[- (1 / N c(ψ)) Σ_{i=1}^N h_{T_i}(x)], (1)

where c(ψ) is the average path length of unsuccessful searches in a binary search tree, see (Liu et al., 2008) for further details.

**Extended Isolation Forest.** Observing that the geometry of the abnormal regions of the feature space is not necessarily well-described by perpendicular splits (i.e. by unions of hypercubes of the cartesian product R^d), a more flexible variant of the procedure recalled above has been proposed in (Hariri et al., 2018), in the purpose of bias reduction. Rather than selecting a direction in {1, ..., d}, one may choose a direction u ∈ S_{d−1}, denoting by S_{d-1} the unit sphere of the euclidian space R^d. A node is then cut by choosing randomly and uniformly a threshold value in the range of the projections onto this direction of the training data points lying in the corresponding region. In the case where X's distribution has a density f(x) w.r.t. a σ-finite measure λ of reference, the goal of anomaly detection can be formulated as the recovery of sublevel sets {x ∈ R^d : f(x) ≤ q}, q ≥ 0, (under mild assumptions, they are minimum volume sets or quantile regions, see (Polonik, 1997; Scott and Nowak, 2006), when measuring the volume by λ), which may be not accurately approximated by unions of hyperrectangles (in the Gaussian situation for instance, such regions are the complementary sets of ellipsoïds, λ being Lebesgue measure on R^d).

### 2.2. Functional Data Analysis and Anomaly Detection

A functional random variable X is a r.v. that takes its values in a space of functions, see, e.g., (Ferraty and Vieu, 2006). To be more specific, let I ⊂ R_+ be a time interval and consider a r.v. taking its values in the Hilbert space L_2(I) of real valued and square integrable (w.r.t. Lebesgue measure) functions x : I → R:

X : Ω → L_2(I)
ω → X(ω) = (X_t(ω))_{t∈I}.

Without any loss of generality, we restrict ourselves with functions defined on [0, 1] throughout the paper. In practice, only a finite dimensional marginal (X_{t_1}, ..., X_{t_p}), t_1 < ... < t_p, p ≥ 1 and (t_1, ..., t_p) ∈ [0, 1]^p can be observed. However, considering (X_{t_1}, ..., X_{t_p}) as a discretized curve rather than a simple random vector of dimension p permits to take into account the dependence structure between the measurements over time, especially when the time points t_i are not equispaced. To come back to a function from discrete values, interpolation procedures or approximation schemes based on appropriate dictionaries can be used, combined with a preliminary smoothing step when the observations are noisy. From a statistical perspective, the analysis is based on a functional dataset S_n = {x_1, ..., x_n} composed of n ≥ 1 independent realizations of finite-dimensional marginals of the stochastic process X, that may be very heterogeneous in the sense that these marginals may correspond to different time points and be of different dimensionality. One may refer to Ramsay and Silverman (2005)'s book for a deep view on Functional Data Analysis (FDA in short). For simplicity, the functional data considered throughout the paper correspond to the observations of independent realizations of X at the same points.

In this particular context, *functional anomaly detection* aims at detecting the curves that significantly differ from the others among the dataset available. Given the richness of spaces of functions, the major difficulty lies in the huge diversity in the nature of the observed differences, which may not only depend on the locations of the curves. Following in the footsteps of Hubert et al. (2015), one may distinguish between three types of anomalies: *shift* (the observed curve has the same shape as the majority of the sample except that it is shifted away), *amplitude* or *shape* anomalies. All these three types of anomalies can be isolated/transient or persistent, depending on their duration with respect to that of the observations. One may easily admit that certain types of anomalies are harder to detect than others: for instance, an isolated anomaly in shape compared to an isolated anomaly in amplitude (i.e. change point). Although FDA has been the subject of much attention in recent years, very few generic and flexible methods tailored to functional anomaly detection are documented in the machine-learning literature to the best of our knowledge, except for specific types of anomalies (e.g. change-points).

In Statistics, although its applications are by no means restricted to AD, the concept of *functional depth* that allows to define a notion of centrality in the path space and a center-outward ordering of the curves of the functional dataset, see, e.g., (Cuevas et al., 2007; Claeskens et al., 2014; Hubert et al., 2015), has been used for this purpose. However, since the vast majority of functional depth functions introduced only describe the relative location properties of the sample curves, they generally fail to detect other types of anomalies. Another popular approach, usually referred to as *filtering*, consists in bringing the AD problem to the multivariate case by means of an adequate projection using Functional Principal Component Analysis (FPCA) (Ramsay and Silverman, 2005) or a preliminary selected basis of the function space considered (e.g. Fourier, wavelets) and apply next an AD algorithm designed for the finite-dimensional setup to the resulting representation. Such methods have obvious drawbacks. In FPCA, estimation of the Kahrunen-Loève basis can be very challenging and lead to loose approximations, jeopardizing next the AD stage, while the *a priori* representation offered by the 'atoms' of a predefined basis or frame may unsuccessfully capture the patterns carrying the relevant information to distinguish abnormal curves from the others. Another approach is based on the notion of *Minimum Volume sets* (MV-sets in shortened version), originally introduced in (Einmahl and Mason, 1992) and that generalizes the concept of quantile for multivariate distributions and offers a nice nonparametric framework for anomaly detection in finite dimension, see Scott and Nowak (2006)'s work. Given the fact that no analogue of Lebesgue measure on an infinite-dimensional Banach space exists and since, considering a law λ of reference (e.g. the Wiener or a Poisson measure) on the function space H of interest, the volume λ(C) of a measurable subset C ⊂ H can be hardly computed in general, it is far from straightforward to extend MV-set estimation to the functional setup.

The angle embraced in this paper is quite different. The direct approach we promote here is free from any preliminary representation stage and can be straightforwardly applied to a functional dataset. Precisely, in the subsequent section, we propose to extend the IF algorithm to the functional data framework, in a very flexible way, so as to deal with a wide variety of anomaly shapes.

## 3. Functional Isolation Forest

We consider the problem of learning a score function s : H → R that reflects the degree of anomaly of elements in an infinite dimensional space H w.r.t. P. By H, we denote a functional Hilbert space equipped with a scalar product (·,·)_H such that any x ∈ H is a real function defined on [0, 1]. In the following, we describe in detail the proposed FUNCTIONAL ISOLATION FOREST (FIF) algorithm and discuss its properties.

### 3.1. The FIF algorithm

A Functional Isolation Forest is a collection of Functional Isolation Trees (F-itrees) built from S = {x_1, ..., x_n}, a training sample composed of independent realizations of a functional random variable, X, that takes its values in H. Given a functional observation x, the score returned by FIF is a monotone transformation of the empirical mean of the path lengths h_{T_i}(x) computed by the F-itrees T_i, for i = 1, . . . , N as defined in Eq. 1 in the multivariate case. While the general construction principle depicted in Section 2.1 remains the same for a F-itree, dealing with functional values raises the issue of finding an adequate feature space to represent various properties of a function. A function may be considered as abnormal according to various criteria of location and shape, and the features should permit to measure such properties. Therefore four ingredients have been introduced to handle functional data in a general and flexible way: (i) a set of candidate *Split variables* and (ii) a scalar product both devoted to function representation, (iii) a probability distribution to sample from this set and select a single *Split variable*, (iv) a probability distribution to select a *Split value*. The entire construction procedure of a F-itree is described in Figure 1.

**Function representation** To define the set of candidate *Split variables*, a direct extension of the original IF algorithm (Liu et al., 2008) would be to randomly draw an argument value (e.g. time), and use functional evaluations at this point to split a node, but this boils down to only rely on instantaneous observations of functional data to capture anomalies, which in practice will be usually interpolated. Drawing a direction on a unit sphere as in (Hariri et al., 2018) is no longer possible due to the potentially excessive richness of H.

***
**Figure 1: Construction procedure of a F-itree.**

**Description:** The figure shows the pseudocode for constructing a Functional Isolation Tree (F-itree).

**Input:** A subsample {x₁, ..., xψ}, a dictionary D, a probability measure ν and a scalar product (·, ·)_H.

**(a) Initialization:** The root node indexed by (0,0) is associated with the whole input space C₀,₀ = H. The construction starts with the training dataset S₀,₀ = {x₁, ..., xn} composed of n i.i.d. realizations of the random variable X. Go to (b) with (j = 0, k = 0).

**(b) Stopping criterion:** Test if the node (j, k) is terminal: a node (j, k) is declared as terminal if the intersection between the current set C_{j,k} and the current training set S_{j,k} is reduced to a single data point or to a set of predefined cardinal. If the node is terminal, then stop the construction for this node, otherwise go to (c).

**(c) Children node construction:** A non-terminal node (j, k) is split in three steps as follows:
1. Choose a *Split variable* d according to the probability distribution ν on D.
2. Choose randomly and uniformly a *Split value* κ in the interval
[min_{x∈S_{j,k}} (x, d)_H, max_{x∈S_{j,k}} (x, d)_H]
3. Form the children subsets
C_{j+1,2k} = C_{j,k} ∩ {x ∈ H : (x, d)_H ≤ κ},
C_{j+1,2k+1} = C_{j,k} ∩ {x ∈ H : (x, d)_H > κ}.
as well as the children training datasets
S_{j+1,2k} = S_{j,k} ∩ C_{j+1,2k} and S_{j+1,2k+1} = S_{j,k} ∩ C_{j+1,2k+1}.

**(d) Recursion:** Apply the building procedure starting from (a) to nodes (j + 1, 2k) and (j + 1, 2k + 1)

**Output:** (C(0,0), C(1,1), . . .)
***

To circumvent these difficulties, we propose to project the observations on elements of a dictionary D ⊂ H that is chosen to be rich enough to explore different properties of data and well appropriate to be sampled in a representative manner. More explicitly, given a function d ∈ D, the projection of a function x ∈ H on D, (x, d)_H defines a feature that partially describes x. When considering all the functions of dictionary D, one gets a set of candidate *Split variables* that provides a rich representation of function X, depending on the nature of the dictionary. Dictionaries have been throughly studied in the signal processing community to achieve *sparse coding* of signals, see e.g. Mallat and Zhang (1993). They also provide a way to incorporate a *priori* information about the nature of the data, a property very useful in an industrial context in which functional data often come from the observation of a well known device and thus can benefit from expert knowledge.

**Sampling a Split variable** Once a dictionary is chosen, a probability distribution ν on D is defined to draw a *Split variable* d. Note that the choice of the sampling distribution ν gives an additional flexibility to orientate the algorithm towards the search for specific properties of the functions.

**Sampling a Split value** Given a chosen *Split variable* d and a current training dataset S_{j,k}, a *Split value* is uniformly drawn in the real interval defined by the smallest and largest values of the projections on d when considering the observations present in the node.

**Discussion on the dictionary** The choice of a suited dictionary plays a key role in construction of the FIF anomaly score. The dictionary can consist of deterministic functions, incorporate stochastic elements, contain the observations from S, or be a mixture of several mentioned options. In *Computational Harmonic Analysis*, a wide variety of bases or frames, such as wavelets, ridgelets, cosine packets, brushlets and so on, have been developed in the last decades in order to represent efficiently/parsimoniously functions, signals or images exhibiting specific form of singularities (e.g. located at isolated points, along hyperplanes) and may provide massive dictionaries. The following ones will be used throughout the article: *mexican hat wavelet dictionary* (MHW), *Brownian motion dictionary* (B), *Brownian bridge dictionary* (BB), *cosine dictionary* (Cos), *uniform indicator dictionary* (UI), *dyadic indicator dictionary* (DI), and the *self-data dictionary* (Self) containing the dataset itself. See Section B and C of the Supplementary Materials for detailed definitions of these dictionaries and further discussion on them, respectively.

**Discussion on the scalar product** Besides the dictionary, the scalar product defined on H brings some additional flexibility to measure different type of anomaly. While L_2 scalar product allows for detection of *location anomalies*, L_2 scalar product of derivatives (or slopes) would allow to detect anomalies regarding *shape*. This last type of anomalies can be challenging; e.g. Hubert et al. (2015) mention that shape anomalies are more difficult to detect, and Mosler and Mozharovskyi (2017) argue that one should consider both location and slope simultaneously for distinguishing complex curves. Beyond these two, a wide diversity of scalar products can be used, involving a variety of L_2-scalar products related to derivatives of certain orders, like in the definition of Banach spaces such as weighted Sobolev spaces, see Maz’ya (2011).

### 3.2. Ability of FIF to detect a variety of anomalies

As discussed in Section 2.2, most of state-of-the-art methods have a focus on a certain type of anomalies and are unable to detect various deviations from the normal behavior. The flexibility of the FIF algorithm allows for choosing the scope of the detection by selecting both the scalar product and the dictionary. Nevertheless, by choosing appropriate scalar product and dictionary, FIF is able to detect a great diversity of deviations from normal data. First, to account for both location and shape anomalies, we suggest the following scalar product that provides a compromise between the both

(f, g) := α × (f,g)_{L_2} / (||f|| ||g||) + (1 − α) × (f', g')_{L_2} / (||f'|| ||g'||), α ∈ [0, 1],

and illustrate its use right below. Thus, setting α = 1 yields the classical L_2 scalar product, α = 0 corresponds to the L_2 scalar product of derivative, and α = 0.5 is the Sobolev W_{1,2} scalar product. To illustrate the FIF's ability to detect a wide variety of anomalies at a time, we calculate the FIF anomaly scores with the Sobolev scalar product and the *gaussian wavelets dictionary* for a sample consisting of 105 curves defined as follows (inspired by (Cuevas et al., 2007), see Fig. 2):

- 100 curves defined by x(t) = 30(1 − t)^q t^q with q equispaced in [1, 1.4],
- 5 abnormal curves composed by one isolated anomaly x₀(t) = 30(1 − t)¹·²t¹·² with a jump in t = 0.7, one magnitude anomaly x₁(t) = 30(1 − t)¹·⁶t¹·⁶ and three kind of shape anomalies x₂(t) = 30(1 − t)¹·²t¹·² + sin(2πt), x₃(t) = 30(1 − t)¹·²t¹·² noised by ε ~ N(0, 0.3²) on the interval [0.2, 0.8] and x₄(t) = 30(1 − t)¹·²t¹·² + 1/2 sin(10πt).

***
**Figure 2: The simulated dataset with the five introduced anomalies (left). The scored dataset (middle), the darker the color, the more the curves are considered anomalies. The sorted anomaly score of the dataset (right).**

**Description:** The figure contains three plots. The leftmost plot shows a set of curves, with five specific anomalous curves (x₀ to x₄) highlighted in different colors and plotted over a gray shaded area representing the normal curves. The x-axis is "Time" from 0.0 to 1.0, and the y-axis is "x(t)". The middle plot shows the same set of curves, but this time they are color-coded based on their anomaly score, with darker colors indicating higher anomaly. The rightmost plot is a scatter plot showing the sorted anomaly scores. The x-axis is "Index of sorted curves" from 0 to 100, and the y-axis is "Score" from 0.4 to 0.8. The five anomalous points (x₀ to x₄) are highlighted and have the highest scores.
***

One can see that the five anomalies, although very different, are all detected by FIF with a significantly different score.

## 4. Numerical Results

In this section, we provide an empirical study of the proposed algorithm. First, in Section 4.1 we explore the stability and consistency of the score function w.r.t. the probability distribution of a r.v. X and the sample size. Furthermore, we examine the influence of proposed dictionaries on the score function and bring performance comparisons with benchmark methods. Second, in Section 4.2, we benchmark the performance of FIF on several real labeled datasets by measuring its ability to recover an "abnormal" class on the test set. In all experiments, N the number of F-itrees is fixed to 100 and the height limit is fixed to [log₂(ψ)].

### 4.1. Impact of the Hyperparameters on Stability

Since functional data are more complex than multivariate data, and the dictionary constitutes an additional source of variance, a question of stability of the FIF anomaly score estimates is of high interest. This issue is even more important because of the absence of theoretical developments due to their challenging nature. The empirical study is conducted on two simulated functional datasets presented in Fig. 3: Dataset (a) is the standard Brownian motion being a classical stochastic process widely used in the literature. Dataset (b) has been used by Claeskens et al. (2014) and has smooth paths. For each dataset, we choose/add four observations for which the FIF anomaly score is computed after training: a normal observation x₀, two anomalies x₁ and x₂, and a more extreme anomaly x₃. We therefore expect the following ranking of the scores: s_n(x₀) < s_n(x₁) ≤ s_n(x₂) < s_n(x₃), for both datasets.

Further, we provide an illustration of the empirical convergence of the score. All other parameters being fixed, we increase the number of observations n when calculating the scores of the four selected observations; the empirical median and the boxplots of the scores computed over 100 random draws of the dataset are shown in Fig. 4. First, one observes score convergence and variance decrease in n. Further, let us take a closer look at the score tendencies on the example of x₀ and x₃. The score of x₃ first increases (for dataset (a)) and slightly decreases (for dataset (b)) with growing n until n reaches ψ = 64, which happens because this abnormal observation is isolated quite fast (and thus has short path length) but the c(ψ) in the denominator of the exponent of (1) increases in ψ. For n > 64, the score of x₃ decreases in n since hᵢ(x₃) overestimates the real path length of x₃ for subsamples in which it is absent; frequency of such subsamples grows in n and equals, e.g., 0.872 for n = 500. On the other hand, this phenomenon allows to unmask grouped anomalies as mentioned in (Liu et al., 2008). The behavior is reciprocal for the typical observation x₀. Its FIF anomaly score starts by decreasing in n since x₀ tends to belong to the deepest branches of the trees and is always selected while ψ < n. For larger n, the path length of x₀ is underestimated for subsamples where it is absent when growing the tree, which explains slight increase in the score before it stabilizes. A second experiment illustrated in Fig. 5 is conducted to measure the impact of various dictionaries shortly cited in Section 3 and more thoroughly

***
**Figure 3: Datasets (a) (left) and (b) (right) containing, respectively, 500 and 200 functional paths with 4 selected observations.**

**Description:** The figure shows two plots. The left plot displays dataset (a), which consists of multiple paths resembling Brownian motion. A gray shaded area represents the bulk of the data, and four specific paths (x₀, x₁, x₂, x₃) are highlighted in different colors. The right plot displays dataset (b), showing smoother, sinusoidal-like paths. Again, a gray shaded area represents the bulk of the data, and four specific paths are highlighted. Both plots have a horizontal axis from 0.0 to 1.0.
***

***
**Figure 4: Boxplot (over 100 repetitions) of the FIF score for the observations x₀, x₁, x₂, x₃ for different sample sizes. The orange boxplots represent the dataset (a) while the purple boxplots represent the dataset (b).**

**Description:** The figure consists of four panels, one for each observation (X₀, X₁, X₂, X₃). Each panel contains a series of horizontal boxplots. The y-axis, labeled "Sample size", shows different sample sizes (10, 20, 50, 100, 200, 500, 1000, 2000). The x-axis represents the "Score". For each sample size, there are two boxplots: an orange one for dataset (a) and a purple one for dataset (b). The plots illustrate how the distribution of the anomaly score changes with the sample size for normal (X₀) and anomalous (X₁, X₂, X₃) observations.
***

described in Section B of the Supplementary Materials; L₂ scalar product is used. One observes that the variance of the score seems to be mostly stable across dictionaries, for both datasets. Thus, random dictionaries like uniform indicator (UI) or Brownian motion (B) do not introduce additional variance into the FIF score. Since we know the expected ranking of the scores, we can observe that FIF relying on the Self, UI, and dyadic indicator (DI) dictionaries fail to make a strong difference between x₀ and x₁. Since x₁ differs only slightly in the amplitude from the general pattern, these dictionaries seem insufficient to capture this fine dissimilarity: while Self and DI dictionaries simply do not contain enough elements, UI dictionary is to simple to capture this difference (it shares this feature with DI dictionary). For the scalar product L₂ on derivatives (see Fig. 18 in the Supplementary Materials), distinguishing anomalies for the Brownian motion becomes difficult since they differ mainly in location, while for a sine function the scores resemble those with the usual L₂ scalar product. Thus, even though – as seen in Section 3.2 – capturing different types of anomalies is one of the general strengths of the FIF algorithm, the dictionary may still have an impact on detection of functional anomalies in particular cases.

More experiments were run regarding the stability of the algorithm, but for sake of space, we describe them in Section C of the Supplementary Materials.

***
**Figure 5: Boxplot (over 100 repetitions) of the FIF score for the observations x₀, x₁, x₂, x₃ for different dictionaries using the L₂ scalar product. The orange boxplots represent the dataset (a) while the purple boxplots represent the dataset (b).**

**Description:** This figure also has four panels for observations X₀, X₁, X₂, and X₃. Each panel shows horizontal boxplots comparing the anomaly scores obtained with different dictionaries (Self, BB, UI, DI, Cos, B, MHW) on the y-axis. The x-axis is the "Score". For each dictionary, there are two boxplots: orange for dataset (a) and purple for dataset (b). This allows for a visual comparison of the performance and stability of different dictionaries.
***

### 4.2. Real Data Benchmarking

To explore the performance of the proposed FIF algorithm, we conduct a comparative study using 13 classification datasets from the UCR repository (Chen et al., 2015). We consider the larger class as normal and some of others as anomalies (see Table 1 for details). When classes are balanced, i.e for 9 datasets out of 13, we keep only part of the anomaly class to reduce its size, always taking the same observations (at the beginning of the table) for a fair comparison. Since the datasets are already split into train/test sets, we use the train part (without labels) to build the FIF and compute the score on the test set. We assess the performance of the algorithm by measuring an Area Under the Receiver Operation Characteristic curve (AUC) on the test set. Both train and test sets are rarely used during learning in unsupervised setting since labels are unavailable when fitting the model. Thus, when fitting the models on unlabeled training data, good performances on the test set show a good generalization power.

**Competitors** FIF is considered with two finite size dictionaries *dyadic indicator*, the *self-data* and the infinite size dictionary *cosines* (with α = 1 and α = 0); its parameters are set N = 100, ψ = min(256, n) and the height limit to = [log₂(ψ)]). We contrast the FIF method with three most used multivariate anomaly detection techniques and two functional depths, with default settings. The multivariate methods—isolation forest (IF) (Liu et al., 2008), local outlier factor (LOF) (Breunig et al., 2000), and one-class support vector machine (OCSVM) (Schölkopf et al., 2001) — are employed after dimension reduction by Functional PCA keeping 20 principal components with largest eigenvalues after a preliminary step of filtering using Haar basis. The depths are the random projection halfspace depth (Cuevas et al., 2007) and the functional Stahel-Donoho outlyingness (Hubert et al., 2015).

**Analysis of the results** Taking into account the complexity of the functional data, as expected there is no method performing generally best. Nevertheless, FIF performs well in most of the cases, giving best results for 10 datasets and second best for 6 datasets. It is worth to mention that the dictionary plays an important role in identifying anomalies, while FIF seems to be rather robust w.r.t. other parameters: The “CinECGTorso” dataset contains anomalies differing in location shift which are captured by the cosine dictionary. Dyadic indicator dictionary allows to detect local anomalies in "TwoLeadECG" and "Yoga" datasets. Self-data dictionary seems suited for Datasets “SonyRobotAI2" and "Starlight Curves" whose challenge is to cope with many different types of anomalies.

**Table 1: Datasets considered in performance comparison: n is the number of instances, na is the number of anomalies. p is the number of discretization points.**

| | p | training: n_a / n | testing: n_a / n | normal lab | anomaly lab |
| :--- | :-: | :--- | :--- | :--- | :--- |
| Chinatown | 24 | 4 / 14 (29%) | 95 / 345 | 2 | 1 |
| Coffee | 286 | 5 / 19 (26%) | 6 / 19 | 1 | 0 |
| ECGFiveDays | 136 | 2 / 16 (12%) | 53 / 481 | 1 | 2 |
| ECG200 | 96 | 31 / 100 (31%) | 36 / 100 | 1 | -1 |
| Handoutlines | 2709 | 362 / 1000 (36%) | 133 / 370 | 1 | 0 |
| SonyRobotAI1 | 70 | 6 / 20 (30%) | 343 / 601 | 2 | 1 |
| SonyRobotAI2 | 65 | 4 / 20 (20%) | 365 / 953 | 2 | 1 |
| StarLightCurves | 1024 | 100 / 673 (15%) | 3482 / 8236 | 3 | 1 and 2 |
| TwoLeadECG | 82 | 2 / 14 (14%) | 570 / 1139 | 1 | 2 |
| Yoga | 426 | 10 / 173 (06%) | 1393 / 3000 | 2 | 1 |
| EOGHorizontal | 1250 | 10 / 40 (25%) | 30 / 61 | 5 | 6 |
| CinECGTorso | 1639 | 4 / 16 (25%) | 345 / 688 | 3 | 4 |
| ECG5000 | 140 | 31 / 323 (10%) | 283 / 2910 | 1 | 3,4 and 5 |

**Table 2: AUC of different anomaly detection methods calculated on the test set. Bold numbers correspond to the best result while italics to the second best.**

| Methods: | DI_L2 | Cos_Sob | Cos_L2 | Self_L2 | IF | LOF | OCSVM | fHD_RP | fS_DO |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Chinatown | *0.93* | 0.82 | 0.74 | 0.77 | 0.69 | 0.68 | 0.70 | 0.76 | **0.98** |
| Coffee | 0.76 | **0.87** | 0.73 | *0.77* | 0.60 | 0.51 | 0.59 | 0.74 | 0.67 |
| ECGFiveDays | 0.78 | 0.75 | 0.81 | 0.56 | 0.81 | *0.89* | **0.90** | 0.60 | 0.81 |
| ECG200 | *0.86* | **0.88** | **0.88** | *0.87* | 0.80 | 0.80 | 0.79 | 0.85 | *0.86* |
| Handoutlines | 0.73 | **0.76** | 0.73 | 0.72 | 0.68 | 0.61 | 0.71 | 0.73 | **0.76** |
| SonyRobotAI1 | **0.89** | 0.80 | 0.85 | 0.83 | 0.79 | 0.69 | 0.74 | 0.83 | *0.94* |
| SonyRobotAI2 | 0.77 | 0.75 | 0.79 | **0.92** | *0.86* | 0.78 | 0.80 | *0.86* | 0.81 |
| StarLightCurves | 0.82 | 0.81 | 0.86 | **0.86** | 0.76 | 0.72 | 0.77 | 0.77 | *0.85* |
| TwoLeadECG | **0.71** | 0.61 | 0.56 | 0.71 | 0.63 | 0.61 | 0.71 | 0.65 | *0.69* |
| Yoga | **0.62** | 0.54 | 0.60 | 0.58 | 0.57 | 0.52 | *0.59* | 0.55 | 0.55 |
| EOGHorizontal | 0.72 | *0.76* | **0.81** | 0.74 | 0.70 | 0.69 | 0.74 | 0.73 | 0.75 |
| CinECGTorso | 0.70 | **0.92** | *0.86* | 0.43 | 0.51 | 0.46 | 0.41 | 0.64 | 0.80 |
| ECG5000 | *0.93* | **0.98** | **0.98** | 0.95 | 0.96 | 0.93 | 0.95 | 0.91 | *0.93* |

## 5. Extensions of FIF

**Extension to multivariate functions** FIF can be easily extended to the multivariate functional data, i.e. when the quantity of interest lies in R^d for each moment of time. For this, the coordinate-wise sum of the d corresponding scalar products is used to project the data onto a chosen dictionary element: (f, g)_{H^{⊗d}} := ∑_{i=1}^d ⟨f⁽ⁱ⁾, g⁽ⁱ⁾⟩_H. The dictionary is then defined in (H([0, 1]))^{⊗d}, e.g., by componentwise application of one or several univariate dictionaries, see Section 3. In the Supplementary Materials we give an illustration of multivariate functional anomaly detection on the MNIST (Lecun et al., 1998) dataset, each digit being seen as a 2D-curve.

**Connection to data depth** Regarding FIF score as an anomaly ranking yields a connection to the notion of the statistical depth function (see (Mosler, 2013) for an overview), which has been successfully applied in outlier detection (see, e.g., (Hubert et al., 2015)). Statistical data depth has been introduced as a measure of centrality (or depth) of an arbitrary observation x ∈ (H([0, 1]))^{⊗d} with respect to the data at hand S. A data depth measure based on FIF score can be defined for (multivariate) functional data as: D_{FIF}(x; S) = 1 − s_n(x; S). Data depth proves to be a useful tool for a low-dimensional data representation called *depth-based map*. Using this property, Li et al. (2012) and Mosler and Mozharovskyi (2017) define a DD-plot classifier which consists in applying a multivariate classifier to the depth-based map. Low-dimensional representation is of particular interest for functional data and a DD-plot classifier can be defined using the FIF-based data depth. Let S^{trn} = S¹ ∪ ... ∪ S^q be a training set for supervised classification containing q classes, each subset S^j standing for class j. The depth map is defined as follows:

x ↦ φ(x) = (D_{FIF}(x; S¹), ..., D_{FIF}(x; S^q)) ∈ [0, 1]^q.

As an illustration, we apply the depth map to 3 digits (1, 5 and 7, 100 observations per digit for training and 100 testing) of the MNIST dataset after their transformation to two-variate functions using **skimage** python library (see Figure 6). One observes appealing geometrical interpretation (observe, e.g., the location of the abnormally distant – from their corresponding classes – observations) and a clear separation of the classes. To illustrate separability, we apply linear multiclass (one-against-all) SVM in the depth space, which delivers the accuracy of 99% on the test data.

## 6. Conclusion

The Functional Isolation Forest algorithm has been proposed, which is an extension of Isolation Forest to functional data. The combined choice of the dictionary itself, the probability distribution used to pick a *Split variable* and the scalar product used for the projection enables FIF to exhibit a great flexibility in detecting anomalies for a variety of tasks. FIF is extendable to multivariate functional data. When transformed in a data depth definition, FIF can be used for supervised classification via a low-dimensional representation – the depth space. The open-source implementation of the method, along with all reproducing scripts, can be accessed at https://github.com/Gstaerman/FIF.

***
**Figure 6: Depth space embedding of the three digits (1, 5 and 7) of the MNIST dataset.**

**Description:** The figure shows a 3D scatter plot. The three axes are "Depth w.r.t. S¹", "Depth w.r.t. S⁵", and "Depth w.r.t. S⁷". There are three distinct clusters of points, colored red, blue, and green, corresponding to the digits 1, 5, and 7. The clusters are well-separated in this 3D depth space, indicating that the depth map provides a good representation for classification.
***

## References

L. Breiman. Random forests. *Machine Learning*, 45(1):5–32, 2001.

M.M. Breunig, H.-P. Kriegel, R.T. Ng, and J. Sander. LOF: Identifying density-based local outliers. In *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, volume 29, pages 93–104. ACM, 2000.

V. Chandola, A. Banerjee, and V. Kumar. Anomaly detection: A survey. *ACM Computing Surveys (CSUR)*, 41(3):15:1–15:58, 2009.

Y. Chen, E. Keogh, B. Hu, N. Begum, A. Bagnall, A. Mueen, and G. Batista. The UCR time series classification archive, July 2015. URL www.cs.ucr.edu/~eamonn/time_series_data/.

G. Claeskens, M. Hubert, L. Slaets, and K. Vakili. Multivariate functional halfspace depth. *Journal of American Statistical Association*, 109(505):411–423, 2014.

A. Cuevas, M. Febrero, and R. Fraiman. Robust estimation and classification for functional data via projection-based depth notions. *Computational Statistics*, 22(3):481–496, 2007.

J.H.J. Einmahl and D.M. Mason. Generalized quantile processes. *The Annals of Statistics*, 20(2):1062–1078, 1992.

F. Ferraty and P. Vieu. *Nonparametric Functional Data Analysis*. Springer-Verlag, New York, 2006.

R Fraiman and G Muniz. Trimmed means for functional data. *Test*, 10(2):419–440, Dec 2001.

P Geurts, D Ernst, and L Wehenkel. Extremely randomized trees. *Machine Learning*, 63 (1):3–42, Apr 2006.

S. Hariri, M. Carrasco Kind, and R. J. Brunner. Extended isolation forest. *ArXiv e-prints*, 2018. URL https://arxiv.org/abs/1811.02141.

M. Hubert, P.J. Rousseeuw, and P. Segaert. Multivariate functional outlier detection. *Statistical Methods & Applications*, 24(2):177–202, 2015.

Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

J. Li, J.A. Cuesta-Albertos, and R.Y. Liu. DD-classifier: Nonparametric classification procedure based on DD-plot. *Journal of the American Statistical Association*, 107(498): 737–753, 2012.

F. T. Liu, K. M. Ting, and Z. Zhou. Isolation forest. In *Proceedings of the 2008 Eighth IEEE International Conference on Data Mining*, pages 413–422. IEEE Computer Society, 2008.

F. T. Liu, K. M. Ting, and Z. Zhou. Isolation-based anomaly detection. In *ACM Transactions on Knowledge Discovery from Data (TKDD)*, volume 6, pages 1–39, 2012.

S.G. Mallat and Z. Zhang. Matching pursuits with time-frequency dictionaries. *IEEE Transactions on signal processing*, 41(12):3397–3415, 1993.

V. Maz’ya. *Sobolev Spaces: with Applications to Elliptic Partial Differential Equations*. Springer-Verlag, Berlin Heidelberg, 2011.

K. Mosler. Depth statistics. In *Robustness and Complex Data Structures: Festschrift in Honour of Ursula Gather*, pages 17–34. Springer, Berlin Heidelberg, 2013.

K. Mosler and P. Mozharovskyi. Fast DD-classification of functional data. *Statistical Papers*, 58(4):1055–1089, 2017.

C. Park, J.Z. Huang, and Y. Ding. A computable plug-in estimator of minimum volume sets for novelty detection. *Operations Research*, 58(5):1469–1480, 2010.

W. Polonik. Minimum volume sets and generalized quantile processes. *Stochastic Processes and their Applications*, 69(1):1–24, 1997.

J.O. Ramsay and B.W. Silverman. *Functional Data Analysis*. Springer-Verlag, New-York, 2005.

B. Schölkopf, J.C. Platt, J. Shawe-Taylor, A. Smola, and R. Williamson. Estimating the support of a high-dimensional distribution. *Neural Computation*, 13(7):1443–1471, 2001.

C. Scott and R. Nowak. Learning minimum volume sets. *Journal of Machine Learning Research*, 7:665–704, 2006.

I. Steinwart, D. Hush, and C. Scovel. A classification framework for anomaly detection. *Journal of Machine Learning Research*, 6:211–232, 2005.

S van der Walt, J.L. Schänberger, J Nunez-Iglesias, F Boulogne, J.D. Warner, N Yager, E Gouillart, and T Yu. scikit-image: image processing in python. *PeerJ*, 2, 2014.

R. Vert and J.-P. Vert. Consistency and convergence rates of one-class SVMs and related algorithms. *Journal of Machine Learning Research*, 7:817–854, 2006.

---

# Supplementary material

## A. Illustrative figures

An example of a functional dataset—the "Chinatown" dataset form the UCR Chen et al. (2015) repository—and evaluation points is plotted in Figure 7. Figure 8 depicts a sample isolation tree with a selected terminal node.

***
**Figure 7: Example of a functional dataset. Red stars indicate the observed values (at the same equispaced time points), while the blue curves are obtained by linearly interpolating them.**

**Description:** The plot shows multiple time series curves (in blue and purple) representing the "Chinatown" dataset. The curves are plotted over a time axis. Red stars on the curves indicate the discrete points at which data was observed. The curves appear to have a bimodal shape, with two peaks over the time interval.
***

***
**Figure 8: An example of a functional isolation tree structure denoted by T. Here, C₂,₁ is a cell associated with a terminal node.**

**Description:** The figure shows a binary tree structure. The root node is C₀,₀. It splits into C₁,₀ and C₁,₁. C₁,₀ splits into C₂,₀ and C₂,₁. C₁,₁ splits into two nodes both labeled C₂,₃ (this may be a typo in the original figure). C₂,₀ splits into C₃,₀ and C₃,₁. The node C₂,₁ is highlighted with a red circle and is a terminal node. A text box points to it, stating "x ∈ C₂,₁ ⇒ h_T(x) = 2". Other nodes C₃,₅, C₃,₆, C₃,₇, C₃,₈ are also shown as children of the nodes at level 2. This diagram illustrates how the feature space is recursively partitioned.
***

## B. Presentation of used dictionaries

In this part, we define properly every dictionaries used in the paper.

*Self-data dictionary* (Self) consisting of the training dataset itself.

*Brownian motion dictionary* (B) is a combination of the space of continuous function D = C([0, 1]) and the Wiener measure W on D.

*Brownian bridge dictionary* (BB) is a combination of the space of continuous function D = C([0, 1]) and the Brownian bridge measure G on D.

*Cosine dictionary* (Cos) consisting of curves with the following forms:
x_{a,ω}(t) = a cos(2πωt)
with a ∈ [0, 1] and ω ∈ [0, 10].

*Mexican hat wavelet dictionary* (MHW) consists of the negative second derivatives of the normal density, shifted and scaled in a appropriate fashion:
x_{θ,σ}(t) = (2 / (√3σπ^(1/4))) * (1 - ((t-θ)/σ)²) * exp(-((t-θ)² / (2σ²)))
with θ ∈ [−0.8, 0.8] and σ ∈ ([0.04, 0.2]).

*Dyadic indicator dictionary* (DI) consisting of a set of indicator functions on the elements of binary partitioning, for a given J (chosen according to the granularity to be captured or from the discretisation considerations) having as elements {(x_{k,j})_{0≤k<2^j}}_{1≤j≤J}:
x_{k,j}(t) = 1(t ∈ [k/2^j, (k+1)/2^j)).

*Uniform indicator dictionary* (UI) consists of indicator function on [a, b] where a and b are choosen uniformly on [0, 1] such that a < b.

*Dyadic indicator derivative* (DId) consisting of a set of indicator functions on the elements of binary partitioning, for a given J (chosen according to the granularity to be captured or from the discretisation considerations) having as elements {(x_{k,j})_{0≤k<2^j}}_{1≤j≤J}:
x_{k,j}(t) = t * 1(t ∈ [k/2^j, (k+1)/2^j)).

*Uniform indicator derivative* (UId) consists of functions t ↦ t on [a, b] where a and b are choosen uniformly on [0, 1] such that a < b.

## C. Further discussion on the choice of dictionary

To illustrate the dicussion on dictionaries, especially the incorporation of stochastic elements and external informations, we bring an example of the use of the *Brownian motion* dictionary. Let W be the Wiener measure defined on C([0, 1]) the space of continuous function on [0,1] and H be the L₂ space. We define by *Brownian motion dictionary* (B) the *Split variables* space induced by ν = W and D = C([0, 1]). Although seeming universal, this dictionary explores almost the entire argument space equivalently, and in practice can be unable to detect *isolated anomalies*. On Fig. 9 we plot the following synthetic dataset:

- 30 curves defined by x(t) = 30(1 - t)^q t^q on t ∈ [0, 0.2] and x(t) = 30(0.8)^q 0.2^q + N(0, 0.3²) on t ∈ [0.2, 0.7] with q equispaced in [0.5, 0.55].
- 1 abnormal curve with the same shape but that is shifted at the beginning and whose continuation is deep in the 30 preceding curves.

One can see that the anomaly is not detected, that indicated as anomaly curve (the one with highest anomaly score) is on the fringe of the dataset though. Illustrative incorporation of the prior knowledge, in its simplified version, can consist, e.g., in adding to the measure W, a Dirac of the indicator function on the interval of interest x(t) = 1(t ≤ 0.25) with weights: W' := 0.2W + 0.8δ_x; this assigns the highest anomaly score to the desired observation. In the sequel, ν follows a uniform distribution if is not explicitly mentioned.

When having not enough prior knowledge, e.g. just knowing to stick to local features of functional data but not the precise interval, one would like to use a dictionary exploring different localities. To illustrate possible advantage of this approach, we use *Mexican hat wavelet* and *Dyadic indicator* dictionaries. Regard the "Chinatown" dataset Chen et al. (2015), which represents pedestrian count in Chinatown-Swanston St North for 12 months during year 2017. With 14 functions (working days) representing normal observations and taking 4 functions (weekends) as anomalies (Figure 10). One observes that while the Mexican hat wavelet dictionary correctly detects part of the anomalies, due to its smooth nature it is distracted by two normal curves with high deviation on the second half of the domain.

***
**Figure 9: Synthetic data containing an isolated anomaly and the observation having the highest anomaly score, with dictionary being pure Brownian motion (left) and Brownian motion mixed with an indicator function in the area of interest in proportions 4 to 1 (right).**

**Description:** The figure contains two plots. Both plots show a set of red curves representing normal data and a single blue curve representing an anomaly. The left plot shows the result with a pure Brownian motion dictionary; the anomalous blue curve is not the one identified as most anomalous. The right plot shows the result with a mixed dictionary; here, the anomalous blue curve is correctly identified as having the highest anomaly score. The x-axis ranges from 0.0 to 1.0.
***

Having straight fronts and begin non-zero only in a small part of the domain, the dyadic indicator dictionary detects all four abnormal observations. Nevertheless, it is not adapted to scalar product that involves derivative. To adapt the dyadic indicator dictionary and the uniform indicator dictionary for the scalar product involving derivatives, we define their *slope* versions being x_{k,j}(t) = t * 1(t ∈ [k/2^j, (k+1)/2^j)) and x(t) = t * 1(t ∈ [t₁, t₂]), respectively, with the notation defined above (i.e. their derivatives become indicator functions). Clearly, this list can be extended with further task-specific dictionaries.

***
**Figure 10: The "Chinatown" dataset, with normal observations in red and anomalies in blue: The data and the true anomalies (left), anomalies detected using the Mexican hat wavelet dictionary (middle), and anomalies detected using the dyadic indicator dictionary (right).**

**Description:** This figure has three plots, all showing the "Chinatown" dataset. Normal curves are in red, and anomalous curves are in blue (dotted lines). The left plot shows the ground truth, clearly separating normal and anomalous curves. The middle plot shows which curves were detected as anomalous by a Mexican hat wavelet dictionary; it misclassifies some normal curves. The right plot shows the detections by a dyadic indicator dictionary, which appears to correctly identify all anomalous curves. The x-axis is "Time" from 0.0 to 1.0.
***

Before, we were considering dictionaries that are independent of data. Nevertheless one can use observations or their certain transform as a dictionary itself: projections on both normal and abnormal observations shall differ for normal ones and for anomalies; this suggests the *self-data dictionary* (Self). This can be extended to the *local self-data dictionary* which consists of the product of the self-data dictionary with the uniform indicator dictionary. As an example, we apply this to the "ECG5000" dataset plotted in Figure 11, where, different to the cosine dicitonary, it allows to detect all abnormal observations.

***
**Figure 11: The "ECG5000" dataset, with normal observations in red and anomalies in blue: The data and the true anomalies (left), anomalies detected using the cosine dictionary (middle), and anomalies detected using the self data dictionary (right).**

**Description:** This figure is similar to the previous one but for the "ECG5000" dataset. It has three plots. The left plot shows the ground truth with normal (red) and anomalous (blue, dotted) ECG signals. The middle plot shows the detections using a cosine dictionary, which misses some anomalies. The right plot shows detections using a self-data dictionary, which successfully identifies all anomalous signals. The x-axis is "Time" from 0.0 to 1.0.
***

***
**Figure 12: FIF anomaly scores for a sample of 100 curves with α = 1 (left) and α = 0 (right). Anomaly score increases from magenta to yellow in the left plot and decreases in the right plot.**

**Description:** The figure displays two plots of functional data. In both plots, the curves are colored based on their anomaly score. The left plot uses a scalar product with α = 1 (L₂ norm), where the few noisy, spiky curves are colored magenta (low anomaly score) and are embedded within the yellow (high anomaly score) normal curves. The right plot uses a scalar product with α = 0 (L₂ norm of derivatives), where the noisy curves are now colored yellow (high anomaly score) and stand out clearly from the magenta (low score) normal curves. This illustrates the effect of the choice of scalar product on detecting different types of anomalies.
***

To conclude this, we provide a last example (see Fig. 12) where we highlight the impact of the scalar product choice. To illustrate the score change caused by different values of α, we calculate the FIF anomaly scores with α = 1 and α = 0 for a sample consisting of 100 curves as follows (inspired by Cuevas et al. (2007), see Fig. 12):

- 90 curves defined by x(t) = 30(1 − t)^q t^q with q equispaced in [1, 1.4],
- 10 abnormal curves defined by x(t) = 30(1 − t)¹·²t¹·² noised by ε ~ N(0, 0.3²) on the interval [0.2, 0.8].

One can see that even though the 10 noisy curves are abnormal for the majority of the data, they are considered as normal ones when only location is taken into account. On the other hand, they are easily distinguished with the high anomaly score when derivatives are examined.

### C.1. Direction importance of finite size Dictionaries

Although feature importance have been tackled in supervised random trees (see e.g. Breiman (2001), Geurts et al. (2006)), this has not been adressed in the *Isolation Forest* literature (see Liu et al. (2008), Liu et al. (2012) and Hariri et al. (2018)). As a very randomized procedure, there is no incrementally way to define feature importance from the supervised setting. Nevertheless, it is a matter of interest in many anomaly detection applications to get interpretability of models, especially when dealing with functional data where many information are contained in curves. Thus, it is rewarding to get an *a posteriori* sparse representation of the dictionary D which corresponds to the discriminating directions that have great importance in the construction of the model. Furthermore, it could bring some information on the distribution of normal data by studying the dispersion of the projection coefficients on a direction d (e.g multi-modality). To extend this notion to the *Functional Isolation Forest* algorithm, we propose two ways to evaluate the importance of the elements of D to discriminate anomaly curves. The general idea is to give importance to elements of D which allows to discriminate between the sample. The naive idea is to add "+1" to the elements of D where an instance of the node sample is isolated (except for the cells with only two instances) such that good directions are those with a high score (after the forest construction). A clever one, more adaptive, would be to get weighted gain since curves isolated at nodes closer to the root should be more rewarding. To do this, we choose to give a reward depending on the size of the sample node where a curve is isolated. Precisely, the given reward is equal to the size of the node sample divided by the (sub)-sample used to build the tree. An example of the latter is given in Figure 13. The experiment is conducted on the real-world CinECGTorso dataset (more details in Section 4.2). We use FIF with the *Dyadic indicator dictionary* and the L₂ scalar product. As we can see, the two most important elements of the dictionary are indicator functions which localize the peak around t = 0.4 where anomalies are really different from the normal ones. These leads to some interpretability of a "black box" procedure.

***
**Figure 13: CinECGTorso training dataset (a), red curves correspond to anomalies while blue curves to normal data. The direction importances given by the "adaptative" way are represented by (b) and the two most important functions (from the dyadic dictionary) used by FIF to build the model are plotted in (c).**

**Description:** The figure contains three plots. Plot (a) shows the CinECGTorso training dataset, with normal data in blue and anomalies in red. Plot (b) is a bar chart showing the importance of different directions (dictionary elements), with a few bars being significantly taller than the rest. Plot (c) shows two indicator functions (step functions), which correspond to the two most important directions identified in plot (b). These functions are non-zero around t = 0.4, highlighting the region where normal and anomalous curves differ the most.
***

### D. Study of the parameters of FIF

In this section, we present results of a simulation study of the variance of the FIF algorithm. The experiments were conducted on the datasets (a) and (b) from Section 4 (see also Figure 3 of Section 4), for each of the four specified observations x₀, x₁, x₂, x₃ using the following settings (except varying parameter):

*Dictionary:* Gaussian wavelets (negative second derivative of the standard Gaussian density) with random variance selected in an uniform way in [0.2, 1] and a translation parameter selected randomly in [−4, 4]. We fixed the size of the dictionary to 1000.

*Scalar product:* L² dot product.

*Size of the dataset:* n = 500.

*Subsampling size:* ψ = 64.

*The number of trees:* N = 100.

*The height limit:* fixed to l = [log₂(ψ)].

The figures below indicate boxplots of the FIF anomaly score, over 100 runs. Empirical study of the FIF anomaly score and its variance when increasing the number of F-itrees is depicted in Figure 14.

***
**Figure 14: Boxplot (over 100 repetitions) of the FIF score for the observations x₀, x₁, x₂, x₃ for different sample sizes. The orange boxplots represent the dataset (a) while the purple boxplots represent the dataset (b).**

**Description:** The figure has four panels, one for each observation (X₀, X₁, X₂, X₃). Each panel contains horizontal boxplots showing the distribution of the FIF score. The y-axis (labeled "N") represents the number of trees in the forest (10, 20, 50, 100, 200, 500). The x-axis is the "Score". For each value of N, there are two boxplots: orange for dataset (a) and purple for dataset (b). The plots show that as the number of trees increases, the variance of the score decreases, leading to more stable estimates.
***

Empirical study of the FIF anomaly score and its variance when increasing the subsample size is depicted in Figure 15.

***
**Figure 15: Boxplot (over 100 repetitions) of the FIF score for the observations x₀, x₁, x₂, x₃ for different sample sizes. The orange boxplots represent the dataset (a) while the purple boxplots represent the dataset (b).**

**Description:** This figure is similar to the previous one, with four panels for X₀, X₁, X₂, and X₃. Here, the y-axis is "Subsample size" (16, 32, 64, 128, 256, 350, 500). The x-axis is "Score". The boxplots (orange for dataset (a), purple for dataset (b)) illustrate how the score distribution changes as the subsample size used to build the trees varies.
***

Empirical study of the FIF anomaly score and its variance when increasing the height limit of the F-itree is depicted in Figure 16.

***
**Figure 16: Boxplot (over 100 repetitions) of the FIF score for the observations x₀, x₁, x₂, x₃ for different sample sizes. The orange boxplots represent the dataset (a) while the purple boxplots represent the dataset (b).**

**Description:** The figure has four panels for X₀, X₁, X₂, and X₃. The y-axis is "Height limit" (from 3 to 8). The x-axis is "Score". The horizontal boxplots (orange for dataset (a), purple for dataset (b)) show the effect of changing the maximum allowed depth of the F-itrees on the resulting anomaly score distribution.
***

Taking finite size versions of the infinite *gaussian wavelets* dictionary, an empirical study of the FIF anomaly score and its variance when increasing the size of the dictionary is depicted in Figure 17.

***
**Figure 17: Boxplot (over 100 repetitions) of the FIF score for the observations x₀, x₁, x₂, x₃ for different sample sizes. The orange boxplots represent the dataset (a) while the purple boxplots represent the dataset (b).**

**Description:** This figure has four panels for X₀, X₁, X₂, and X₃. The y-axis is "Dictionary size" (50, 100, 200, 500, 750, 1000, 1500, 2000, 5000, 10000). The x-axis is "Score". The horizontal boxplots (orange for dataset (a), purple for dataset (b)) show that as the dictionary size increases, the variance of the score tends to stabilize.
***

Empirical study of the FIF anomaly score for a variety of dictionaries with the L₂ scalar product of the derivatives is depicted in Figure 18.

***
**Figure 18: Boxplot (over 100 repetitions) of the FIF score for the observations x₀, x₁, x₂, x₃ for different dictionaries using the L₂ scalar product of the derivatives. The orange boxplots represent the dataset (a) while the purple boxplots represent the dataset (b).**

**Description:** This figure is similar to Figure 5, with four panels for X₀, X₁, X₂, and X₃. It compares the anomaly scores obtained with different dictionaries (Self, BB, UId, DId, Cos, B, MHW) on the y-axis, but this time using the L₂ scalar product on the derivatives. The x-axis is "Score". The boxplots (orange for dataset (a), purple for dataset (b)) show the performance of different dictionaries for detecting shape anomalies.
***

**Analysis of the results of Section C** In a first experiment, we show the boxplots of the score estimated by FIF when increasing the number of F-itrees and observe that, as expected, the variance diminishes when N grows (see Figure 14). We also see in Figure 15 that with an increasing subsample size ψ the FIF anomaly score increases for anomalies since these are more often present in the subsample and thus isolated faster (with shorter path length) when calculating the score than when they were absent in the subsample; this effect is reciprocal for normal observations. A similar behavior is observed with increasing height limit l in Figure 16. The variance of the score tends to slightly increase with ψ and l because of more observations/branching possibilities. If the dictionary is sufficiently rich, its size does not influence the FIF anomaly score and its variance stabilizes relatively fast while growing the size of the dictionary (see Figure 17) which encourages the use of massive (and infinite size) dictionaries.

## E. Complementary results on the performance comparison

### E.1. benchmark datasets

Here, we plot the thirteen benchmark train datasets used in the experiment. Anomalies are represented by blue color while normal data are drawn in red.

***
**Description:** The figure contains 13 small plots, each showing a different benchmark dataset used in the experiments. The datasets are: Chinatown, Coffee, ECGFiveDays, ECG200, Handoutlines, SonyRobotAI1, SonyRobotAI2, StarLightCurves, TwoLeadECG, Yoga, EOGHorizontal, CinECGTorso, and ECG5000. In each plot, normal time series are shown in red and anomalous time series are shown in blue. This provides a visual reference for the data used in the benchmarking study.
***

### E.2. Functional depth

In this part, we present further 8 functional depths which are outperformed (on an average) by the two depth functions presented in Section 4 on the 13 real-world datasets and we display their AUC performance. SFD corresponds to *simplicial integrated depth*, HFD to *Halfspace integrated depth*, RP-SD to the *random projection method with simplicial depth*, RP-RHD to the *random projection method with random halfspace depth*, fAO to the *functional adjusted outlyingness*, fDO to the *functional directional outlyingness* to and fbd to *functional bagdistance*. The reader is referred to Cuevas et al. (2007); Fraiman and Muniz (2001); Hubert et al. (2015) for the bibliography on employed functional data depth notions.

**Table of AUC results for various functional depth methods.**

| Datasets | S-FD | H-FD | Modal | RP-SD | RP-RHD | fAO | fDO | fbd |
|---|---|---|---|---|---|---|---|---|
| Chinatown | 0.74 | 0.77 | 0.75 | 0.77 | 0.73 | 0.70 | 0.83 | 0.83 |
| Coffee | 0.60 | 0.59 | 0.69 | 0.70 | 0.51 | 0.53 | 0.59 | 0.60 |
| ECGFiveDays | 0.65 | 0.64 | 0.60 | 0.64 | 0.56 | 0.72 | 0.76 | 0.80 |
| ECG200 | 0.82 | 0.82 | 0.84 | 0.85 | 0.74 | 0.78 | 0.82 | 0.82 |
| Handoutlines | 0.70 | 0.70 | 0.75 | 0.72 | 0.63 | 0.60 | 0.71 | 0.73 |
| SonyRobotAI1 | 0.89 | 0.89 | 0.94 | 0.83 | 0.62 | 0.90 | 0.93 | 0.93 |
| SonyRobotAI2 | 0.82 | 0.82 | 0.92 | 0.86 | 0.71 | 0.80 | 0.82 | 0.80 |
| StarLightCurves | 0.80 | 0.80 | 0.85 | 0.78 | 0.68 | 0.80 | 0.82 | 0.83 |
| TwoLeadECG | 0.68 | 0.67 | 0.68 | 0.66 | 0.60 | 0.67 | 0.69 | 0.69 |
| Yoga | 0.55 | 0.53 | 0.57 | 0.57 | 0.54 | 0.54 | 0.55 | 0.56 |
| EOGHorizontal | 0.59 | 0.52 | 0.84 | 0.74 | 0.64 | 0.53 | 0.59 | 0.66 |
| CinECGTorso | 0.69 | 0.69 | 0.73 | 0.62 | 0.66 | 0.85 | 0.83 | 0.79 |
| ECG5000 | 0.90 | 0.90 | 0.92 | 0.92 | 0.84 | 0.87 | 0.92 | 0.92 |

### E.3. Isolation forest after dimension reduction by filtering methods on the benchmark datasets

Here, we show the results of the filtering approach using 106 bases from the **PyWavelets** python library and the Fourier basis. Afterwards, we apply (multivariate) Isolation Forest on the coefficients of the projections and display the AUC performance.

*(Due to the extreme size and density of the following tables, they are omitted in this summary. They present a comprehensive benchmark of using standard Isolation Forest after projecting the functional data onto various wavelet bases (bior, coif, db, etc.) and Fourier basis, reporting the AUC for each combination of dataset and basis.)*

### E.4. IF/FPCA with different filtering step on the benchmark datasets

Here, we show the results of the FPCA approach using the Fourier basis and 106 further bases from the **PyWavelets** python library as preliminary filtering stage. Afterwards, we apply (multivariate) Isolation Forest on the coefficients of the projections and display the AUC performance.

*(Similar to the previous section, the tables for IF/FPCA results are omitted due to their large size. They detail the AUC performance of Isolation Forest combined with Functional Principal Component Analysis, using a wide variety of basis functions for the initial FPCA step.)*

## F. Multivariate Functional Isolation Forest and depth mapping

***
**Figure 19: FIF anomaly scores for a sample of 110 digits (100 seven and 10 two). Left plot corresponds to the sorted score of these curves. Right plot represents the digits in three dimensions, green ones correspond to normal data, anomalies score increases from orange to dark red. Bottom plot shows the fifteen detected anomalies.**

**Description:** The figure has three parts. The top-left is a scatter plot of sorted anomaly scores, showing a clear "elbow" separating the high-score anomalies from the low-score normal data. A red line indicates a threshold. The top-right is a 3D plot showing the functional data (digit contours) embedded in space. Normal data ('7's) are green, while anomalies ('2's and abnormal '7's) are colored on a scale from orange to red based on their score. The bottom part shows images of the 15 curves with the highest anomaly scores. These include all ten '2's and five unusually shaped '7's.
***

FIF can be easily extended to the multivariate functional data, i.e. when the quantity of interest lies in R^d for each moment of time:

X : Ω → (H([0, 1]))^⊗d
ω ↦ ((X₁(ω))_{t∈[0,1]}, ..., (X_d(ω))_{t∈[0,1]})

For this, the coordinate-wise sum of the d corresponding scalar products is used to project the data onto a chosen dictionary element:

(f, g)_{H^⊗d} := ∑_{i=1}^d (f⁽ⁱ⁾, g⁽ⁱ⁾)_H.

Further, a dictionary should be defined in (H([0,1]))^⊗d. This can be done, e.g., by either componentwise application of one or several univariate dictionaries from Sections 3 and B, or by constructing of special d-variate ones. For illustration purposes, regard the following example constructed based on the MNIST Lecun et al. (1998) dataset. First, we extract the digits’ contours (skeletons) using skimage python library van der Walt et al. (2014). Then each observation is transformed into a curve in (L₂([0, 1]) × L₂([0, 1])) (one vertical and one horizontal coordinates) using length parametrization on [0,1]. We construct the problem by taking 100 curves from class 7 and adding 10 observations from class 2. We apply FIF with two-dimensional *sinuscosine* dictionary and the following scalar product : (f, g)_{(L₂)^⊗d}. *sinuscosine* is constructed as a direct extension of *cosine* dictionary introduced for FIF by selecting randomly cosine or sinus function on each coordinates. Figure 19 shows anomaly detection using the visual elbow rule to define the threshold. Among those detected, five digits are indeed 7s, but do not resemble them and thus are identified as anomalies.