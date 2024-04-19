# Multi Modal Sentiment Analysis Using MOSI dataset

CMU-Multimodal SDK provides tools to easily load well-known multimodal datasets and rapidly build neural multimodal deep models. Hence the SDK comprises of two modules: 1) mmdatasdk: module for downloading and procesing multimodal datasets using computational sequences. 2) mmmodelsdk: tools to utilize complex neural models as well as layers for building new models. The fusion models in prior papers will be released here. 

All the datasets here are processed using the SDK (even the old_processed_data folder which uses SDK V0). You can acquire the citations for the computational sequences used in your project by calling the below functions on your dataset:	

```python
>>> mydataset.bib_citations(open('mydataset.bib','w'))
>>> mycompseq.bib_citations(open('mycompseq.bib','w'))
```

If you use the data and models created under the CMU Multimodal SDK, please consider citing the research paper that sparked the creation of the SDK and unified multiple multimodal datasets:

Data for MOSI and MOSEI: https://github.com/pliang279/MultiBench


## CMU Multimodal Data SDK (mmdatasdk)

CMU-Multimodal Data SDK simplifies downloading and loading multimodal datasets. The module mmdatasdk treats each multimodal dataset as a combination of **computational sequences**. Each computational sequence contains information from one modality in a heirarchical format, defined in the continuation of this section. Computational sequences are self-contained and independent; they can be used to train models in isolation. They can be downloaded, shared and registered with our trust servers. This allows the community to share data and recreate results in a more elegant way using computational sequence intrgrity checks. Furthermore, this integrity check allows users to download the correct computational sequences. 

Each computational sequence is a heirarchical data strcuture which contains two key elements 1) "data" is a heirarchy of features in the computational sequence categorized based on unique multimodal source identifier (for example video id). Each multimodal source has two matrices associated with it: features and intervals. Features denote the computational descriptors and intervals denote their associated timestamp. Both features and intervals are numpy 2d arrays. 2) "metadata": contains information about the computational sequence including integrity and version information. The computational sequences are stored as hdf5 objects on hard disk with ".csd" extension (computational sequential data). Both the data and metadata are stored under "root name" (root of the heirarchy)

A dataset is defined as a dictionary of multiple computational sequences. Entire datasets can be shared using recipes as opposed to old-fashioned dropbox links or ftp servers. Computational sequences are downloaded one by one and their individual integrity is checked to make sure they are the ones users wanted to share. Users can register their extracted features with our trust server to use this feature. They can also request storage of their features on our servers 


## 🚀 Installation

The first step is to download the SDK:

```bash
git clone git@github.com:A2Zadeh/CMU-MultimodalSDK.git
```

Next, you need to install the SDK on your python enviroment.

```bash
cd CMU-MultimodalSDK
pip install .
```
This will install the mmsdk module and all the required dependencies on your python enviroment and you can start using it.

You can also install the SDK in development mode:

```bash
cd CMU-MultimodalSDK
pip install -e .
```


## Usage

The first step in most machine learning tasks is to acquire the data. We will work with CMU-MOSI for this readme. 

```python
>>> from mmsdk import mmdatasdk
```

Now that mmdatasdk is loaded you can proceed to fetch a dataset. The datasets are a set of computational sequences, where each computational sequence hosts the information from a modality or a view of a modality. For example a computational sequence could be the word vectors and another computational sequence could be phoneme 1-hot vectors. 

If you are using a standard dataset, you can find the list of them in the mmdatasdk/dataset/standard_datasets. We use CMU-MOSI for now. We will work with highlevel features (glove embeddings, facet facial expressions, covarep acoustic features, etc)

```python
>>> from mmsdk import mmdatasdk
>>> cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')
```

This will download the data using the links provided in *mmdatasdk.cmu_mosi.highlevel* dictionary (mappings between computational sequence keys and their respective download link) and put them in the *cmumosi/* folder. 

The data that gets downloaded comes in different frequencies, however, they computational sequence keys will always be the same. For example if video v0 exists in glove embeddings, then v0 should exist in other computational sequences as well. The data with different frequency is applicable for machine learning tasks, however, sometimes the data needs to be aligned. The next stage is to align the data according to a modality. For example we would like to align all computational sequences according to the labels of a dataset. First, we fetch the opinion segment labels computational sequence for CMU-MOSI. 

```python
>>> cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'cmumosi/')
```

Next we align everything to the opinion segment labels. 

```python
>>> cmumosi_highlevel.align('Opinion Segment Labels')
```

*Opinion Segment Labels* is the key for the labels we just fetched. Since every video has multiple segments according to annotations and timing in opinion segment labels, each video will also be accompanied by a [x] where x denotes which opinion segment the computational sequence information belongs to; for example v0[2] denotes third segment of v0 (starting from [0]). 


# Supported Datasets
**CMU-MOSEI**: CMU-MOSEI contains a large number of in-the-wild videos annotated for sentiment and emotions. The annotations follow consensus-based online perceptions. Alongside the value of this dataset for sentiment and emotion recognition, it is also suitable for representation learning due to large size of the dataset. 

**CMU-MOSI**: CMU-MOSI is a standard benchmark for multimodal sentiment analysis. It is specially suited to train and test multimodal models, since most of the newest works in multimodal temporal data use this dataset in their papers. 
