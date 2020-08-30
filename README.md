# TextualSpeakerChangeDetection

*************************************
Prerequisities:
*************************************
1. Python 3.5 (or higher)
2. Python Dependencies: PyTorch (torch), pandas, numpy, gensim, sklearn

*************************************
Running:
*************************************
1. Storage (off-line mode) : Run the partitioned_face_container_creator.py. 
This script reads a set of images from a pre-defined folder, then converting the images to an embedding vector of size 512,
and store the images according to their sub-folders belonging. (i.e. the image labels). In this case, the labels are the person
identities who gets partition over their name in the created collection. Finally, the partitioned person images names are stored
in a pickle file (since up to this time, Milvus doesn't supports the partition name in a query's result).

2. Inference (online mode) : Run the partitioned_face_cosine_searcher.py.
This script gets a set of images from a pre-defined folder (which has no sub-directories since there are no labels). As asme as
in the first script, the images are then converted into a 512-dimensional vector rperesentation, and inferred via the MTCNN and
ResnNet Inception Neural networks that detects where are the face(s) in the image.
Next, are extracted from the collection of milvus database the top K (pre-defined) similar vectors, and for any such one, if it's lower than
a pre defined threshold then its partition name is returned, i.e. the person label. Finally, a majority decision is selected.  

