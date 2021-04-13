# From http://www.katrinerk.com/courses/computational-semantics-undergraduate/demo-the-building-blocks-of-a-distributional-model

###############
# Demo: The building block of a distributional model
# Katrin Erk, September 2016
# This code is meant to demonstrate a simple distributional model.
# It is *not* optimized th deal with large amounts of data!
#
# Suggested use:
# Load file into the Idle shell, choose "Run module",
# then run the test...() predicate you are interested in.
# But make sure to change the name of the demo directory first. 

# the corpus:
# a collection of more-or-less gothic tales from Project Gutenberg
demo_dir = "/Users/katrinerk/Desktop/demo_corpora_mini/"


##########
# Preprocessing:
# Design decision:
# We filter away stopwords,
# add part-of-speech tags,
# and reduce words to stems

import nltk
import string

# input: a string that may contain multiple words.
# output: a list of strings, preprocessed
def preprocess(s):
    # split up into words, lowercase, remove punctuation at beginning and end of word
    return [ w.lower().strip(string.punctuation) for w in s.split() ]

# or like this:
# def preprocess(s):
#     words =  [ ]
#     for w in s.split():
#         word = w.lower()
#         word = word.strip(string.punctuation)
#         words.append(word)
#     return words


##
# Run this:
def test_preprocess():
    print("Preprocessing demo:\n", preprocess("This is a test sentence, which contains some punctuation."))

#####################
# Counting words:
# We want to make a list of the N most frequent words in our corpus

import os

def do_word_count(demo_dir, numdims):
    # we store the counts in word_count
    # using NLTK's FreqDist
    word_count = nltk.FreqDist()
    
    # We iterate over the corpus files
    for filename in os.listdir(demo_dir):
        if filename.endswith("txt"):
            print("reading file", filename)
            text = open(os.path.join(demo_dir, filename)).read()
            word_count.update(preprocess(text))
            
    # keep_wordfreq is a list of (word, frequency) pairs
    keep_wordfreq = word_count.most_common(numdims)
    keep_these_words = [ w for w, freq in keep_wordfreq ]
    # print("Target words:\n", keep_these_words, "\n")
    
    return keep_these_words

# or like this, without FreqDist:
# def do_word_count(demo_dir, numdims):
#     word_count = { }

#     for filename in os.listdir(demo_dir):
#         if filename.endswith("txt"):
#             print("reading file", filename)
#         text = open(os.path.join(demo_dir, filename)).read()
#         for taggedword in preprocess(text):
#             if taggedword not in word_count:
#                 word_count[ taggedword ] = 0
#             word_count[ taggedword ] += 1
#
#     def map_word_to_count(word): return word_count[ word ]
#     keep_these_words = sorted(word_count.keys(), key = map_word_to_count)[:numdims]
#     
#     # print("Target words (and also dimensions):\n", keep_these_words, "\n")
#
#     return keep_these_words


##
# run this:
def test_wordcount():
    print("Doing a frequency-based cutoff: keeping only the N most frequent context words.")
    
    # with 10 dimensions
    keepwords = do_word_count(demo_dir, 10)
    print("Keeping only 10 dimensions, then I get:", keepwords, "\n")

    # with 100 dimensions
    keepwords = do_word_count(demo_dir, 100)
    print("Keeping 100 dimensions, then I get:", keepwords, "\n")

##
# We will need the function make_word_index below.
# It maps each word that we want to keep around as a context item
# to an index, which will be its place in the table of counts,
# that is, its dimension in the space
def make_word_index(keep_these_words):
    # make an index that maps words from 'keep_these_words' to their index
    word_index = { }
    for index, word in enumerate(keep_these_words):
        word_index[ word ] = index

    return word_index
   
####################
# Identifying context items for a word

###
# identifying context words for a narrow context window of 2 words on either side
# of the target:
# takes as input a sequence of words for counting. 
# For each word in the sequence, make 4 pairs:
# (word, left neighbor of word), (word, left neighbor of left neighbor of word),
# (word, right neighbor of word), (word, right neighbor of right neighbor of word),
# so pair each word with all its context items in the context window.
# Return a list of these pairs. 
def co_occurrences(wordsequence):
    target_context_pairs = [ ]

    # for a sequence of length N, count from 0 to N-1 
    for index in range(len(wordsequence) - 1):
        # count that word[index] as a target co-occurred with the next word as a context item,
        # and vice versa
        target_context_pairs.append( (wordsequence[index], wordsequence[index+1]) )
        target_context_pairs.append( (wordsequence[index+1], wordsequence[index]) )

        if index + 2 < len(wordsequence):
            # there is a word 2 words away
            # count that word[index] as a target co-occurred with the but-next word as a context item,
            # and vice versa
            target_context_pairs.append( (wordsequence[index], wordsequence[index+2]) )
            target_context_pairs.append( (wordsequence[index+2], wordsequence[index]) )

    return target_context_pairs

###
# run this to test co-occurrences
def test_cooccurrences():
    text = """You will not find Dr. Jekyll; he is from home," replied Mr. Hyde"""
    print("Testing the function that pairs up each target word with its context words.")
    print("Original text:", text, "\n")

    words = preprocess(text)
    cooc = co_occurrences(words)
    print("These are the target/context pairs:", cooc, "\n")

###################
# doing the actual counting:
# We keep the counts for each word in a numpy array.
# The mapping from word to counts is done through a Python dictionary

import numpy

# read all files in demo_dir, and compute a counts vector
# of length numdims for each relevant word.
# The function takes as input also a mapping word_index from relevant words
# to their dimension, from which we derive a set relevant_words.
# This function reads the texts one sentence at a time.
# In each sentence, it identifies context words in the window
# defined by co_occurrences(), and stores them if both the target
# and its context words are relevant_words
def make_space(demo_dir, word_index, numdims):

    # relevant words: those that have an entry in word_index
    relevant_words = set(word_index.keys())

    # space: a mapping from relevant_words to an array of integers (raw counts)
    space = { }
    # fill the space with all zeros.
    for word in relevant_words:
        space[ word ] = numpy.zeros(numdims, dtype = numpy.int)

    ##
    # Design decision: We want to take sentence boundaries into account
    # when computing distributional representations.
    # So we need to detect sentence boundaries first.
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # We iterate over the corpus files
    # and count word co-occurrences in a window of 2
    for filename in os.listdir(demo_dir):
        if filename.endswith("txt"):
            print("reading file", filename)
            # read the text
            text = open(os.path.join(demo_dir, filename)).read()
            # split the text into sentences
            sentences = sent_detector.tokenize(text)
            # process one sentence at a time
            for sentence in sentences:
                words = preprocess(sentence)

                # determine pairs of co-occurrences to count,
                # and store them in the matrix
                for target, cxitem in co_occurrences(words):
                    # are these two words relevant?
                    if target in relevant_words and cxitem in relevant_words:
                        # what is the row for this context item?
                        cxitem_index = word_index[ cxitem]
                        # now count
                        space[ target ][cxitem_index] += 1


    return space

###
# run this
def test_space():
    numdims = 50
    # which words to use as targets and context words?
    ktw = do_word_count(demo_dir, numdims)
    # mapping words to an index, which will be their column
    # in the table of counts
    wi = make_word_index(ktw)
    words_in_order = sorted(wi.keys(), key=lambda w:wi[w])
    
    print("word index:")
    for word in words_in_order:
        print(word, wi[word], end= " ")
    print("\n")

    space = make_space(demo_dir, wi, numdims)
    
    print("some words from the space")
    for w in words_in_order[:10]:
        print(w,  space[w], "\n")
    

#########
# transform the space using positive pointwise mutual information

# target t, dimension value c, then
# PMI(t, c) = log ( P(t, c) / (P(t) P(c)) )
# where
# P(t, c) = #(t, c) / #(_, _)
# P(t) = #(t, _) / #(_, _)
# P(c) = #(_, c) / #(_, _)
#
# PPMI(t, c) =   PMI(t, c) if PMI(t, c) > 0
#                0 else
def ppmi_transform(space, word_index):
    # #(t, _): for each target word, sum up all its counts.
    # row_sums is a dictionary mapping from target words to row sums
    row_sums = { }
    for word in space.keys():
        row_sums[word] = space[word].sum()

    # #(_, c): for each context word, sum up all its counts
    # This should be the same as #(t, _) because the set of targets
    # is the same as the set of contexts.
    # col_sums is a dictionary mapping from context word indices to column sums
    col_sums = { }
    for index in word_index.values():
        col_sums[ index ] = sum( [ vector[ index ] for vector in space.values() ])

    # sanity check: row sums same as column sums?
    for word in space.keys():
        if row_sums[word] != col_sums[ word_index[word]]:
            print("whoops, failed sanity check for", word, row_sums[word], col_sums[word_index[word]])
    
    # #(_, _): overall count of occurrences. sum of all row_sums
    all_sums = sum(row_sums.values())

    # if all_sums is zero, there's nothing we can do
    # because we then cannot divide by #(_, _)
    if all_sums == 0:
        print("completely empty space, returning it unchanged")
        return space

    # P(t) = #(t, _) / #(_, _)
    p_t = { }
    for word in space.keys():
        p_t[ word ] = row_sums[ word ] / all_sums

    # P(c) = #(_, c) / #(_, _)
    p_c = { }
    for index in col_sums.keys():
        p_c[ index ] = col_sums[ index ] / all_sums

    # ppmi_space: a mapping from words to vectors of values 
    ppmi_space = { }
    # first we map from words to values P(t, c)
    for word in space.keys():
        ppmi_space[ word ] = space[ word ] / all_sums
    # divide each entry by P(t)
    for word in space.keys():
        if p_t[ word ] == 0:
            # I haven't seen this word ever, so I cannot
            # divide by P(t). But the whole entry for this word
            # should be 0's, so leave as is.
            pass
        else:
            ppmi_space[ word ] = ppmi_space[ word ] / p_t[ word ]
    # divide each entry by P(c)
    for index in p_c.keys():
        if p_c[ index ] == 0:
            # I haven't seen this context item ever,
            # so I cannot divide by P(c).
            # But every target word will have an entry of 0.0
            # on this column, so nothing more to do.
            pass
        else:
            for word in space.keys():
                ppmi_space[ word ][index] = ppmi_space[ word][index] / p_c[ index ]
                
    # take the logarithm, ignore entries that are zero
    for word in space.keys():
        with numpy.errstate(divide="ignore",invalid="ignore"):
            ppmi_space[ word ] = numpy.log(ppmi_space[ word ])
            

    # turn negative numbers to zero
    for word in space.keys():
        ppmi_space[word] = numpy.maximum(ppmi_space[word], 0.0)

    return ppmi_space

###
# run this:
def test_ppmispace():
    numdims = 50
    # which words to use as targets and context words?
    ktw = do_word_count(demo_dir, numdims)
    # mapping words to an index, which will be their column
    # in the table of counts
    wi = make_word_index(ktw)
    words_in_order = sorted(wi.keys(), key=lambda w:wi[w])
    
    print("word index:")
    for word in words_in_order:
        print(word, wi[word], end=" ")
    print("\n")

    space = make_space(demo_dir, wi, numdims)
    ppmispace = ppmi_transform(space, wi)
    
    print("some raw counts vectors and some ppmi vectors")
    for w in words_in_order[:10]:
        print("---------", "\n", w)
        print("raw", space[w])
        # for the PPMI space, we're rounding to 2 digits after the floating point
        print("ppmi", numpy.round(ppmispace[w], 2), "\n")
        

#################
# transforming the space using singular value decomposition.
# 
def svd_transform(space, originalnumdimensions,keepnumdimensions):
    # space is a dictionary mapping words to vectors.
    # combine those into a big matrix.
    spacematrix = numpy.empty((len(space.keys()), originalnumdimensions))

    rowlabels = sorted(space.keys())

    for index, word in enumerate(rowlabels):
        spacematrix[index] = space[word]

    # now do SVD
    umatrix, sigmavector, vmatrix = numpy.linalg.svd(spacematrix)

    # remove the last few dimensions of u and sigma
    utrunc = umatrix[:, :keepnumdimensions]
    sigmatrunc = sigmavector[ :keepnumdimensions]

    # new space: U %matrixproduct% Sigma_as_diagonal_matrix   
    newspacematrix = numpy.dot(utrunc, numpy.diag(sigmatrunc))

    # transform back to a dictionary mapping words to vectors
    newspace = { }
    for index, word in enumerate(rowlabels):
        newspace[ word ] = newspacematrix[index]
        
    return newspace


####
# run this:
def test_svdspace():
    numdims = 50
    # which words to use as targets and context words?
    ktw = do_word_count(demo_dir, numdims)
    # mapping words to an index, which will be their column
    # in the table of counts
    wi = make_word_index(ktw)
    words_in_order = sorted(wi.keys(), key=lambda w:wi[w])
    
    print("word index:")
    for word in words_in_order:
        print(word, wi[word], end=" ")
    print("\n")

    space = make_space(demo_dir, wi, numdims)
    ppmispace = ppmi_transform(space, wi)
    svdspace = svd_transform(ppmispace, numdims, 5)
    
    print("some vectors")
    for w in words_in_order[:10]:
        print("--------------", "\n", w)
        print("raw", space[w])
        # for the PPMI and SVD spaces, we're rounding to 2 digits after the floating point
        print("ppmi", numpy.round(ppmispace[w], 2), "\n")
        print("svd", numpy.round(svdspace[w], 2), "\n")
        

###
# similarity measure: cosine
#                           sum_i vec1_i * vec2_i
# cosine(vec1, vec2) = ------------------------------
#                        veclen(vec1) * veclen(vec2)
# where
#
# veclen(vec) = squareroot( sum_i vec_i*vec_i )
#

import math

def veclen(vector):
    return math.sqrt(numpy.sum(numpy.square(vector)))

def cosine(word1, word2, space):
    vec1 = space[ word1 ]
    vec2 = space[word2]

    veclen1 = veclen(vec1)
    veclen2 = veclen(vec2)

    if veclen1 == 0.0 or veclen2 == 0.0:
        # one of the vectors is empty. make the cosine zero.
        return 0.0

    else:
        # we could also simply do:
        # dotproduct = numpy.dot(vec1, vec2)
        dotproduct = numpy.sum(vec1 * vec2)

        return dotproduct / (veclen1 * veclen2)


#######
# run this:
def test_cosine():
    # this time we're not removing any words
    numdims = 100
    # which words to use as targets and context words?
    ktw = do_word_count(demo_dir, numdims)
    # mapping words to an index, which will be their column
    # in the table of counts
    wi = make_word_index(ktw)
    words_in_order = sorted(wi.keys(), key=lambda w:wi[w])
    
    space = make_space(demo_dir, wi, numdims)
    ppmispace = ppmi_transform(space, wi)
    svdspace = svd_transform(ppmispace, numdims, 5)
    
    print("some cosines")
    print("'lawyer' and 'lean':")
    print("raw", cosine("lawyer", "lean", space))
    print("ppmi", cosine("lawyer", "lean", ppmispace))
    print("svd", cosine("lawyer", "lean", svdspace))
    
    print("'a' and 'and':")
    print("raw", cosine("a", "and", space))
    print("ppmi", cosine("a", "and", ppmispace))
    print("svd", cosine("a", "and", svdspace))

    print("'friendly' and 'cold':")
    print("raw", cosine("friendly", "cold", space))
    print("ppmi", cosine("friendly", "cold", ppmispace))
    print("svd", cosine("friendly", "cold", svdspace))
    

####################
# finding the word most similar to a given target
def most_similar_to(word1, space):

    sims = [ (word2, cosine(word1, word2, space)) for word2 in space.keys() if word2 != word1 ]

    return sorted(sims, key = lambda p:p[1], reverse=True)


#############################
# run this:
def test_mostsimilar():
    # this time we're not removing any words
    numdims = 100
    # which words to use as targets and context words?
    ktw = do_word_count(demo_dir, numdims)
    # mapping words to an index, which will be their column
    # in the table of counts
    wi = make_word_index(ktw)
    words_in_order = sorted(wi.keys(), key=lambda w:wi[w])
    
    space = make_space(demo_dir, wi, numdims)
    ppmispace = ppmi_transform(space, wi)
    svdspace = svd_transform(ppmispace, numdims, 5)
    
    print("ten most similar to 'friendly':")
    print("raw", most_similar_to("friendly", space)[:10])
    print("ppmi", most_similar_to("friendly", ppmispace)[:10])
    print("svd", most_similar_to("friendly", svdspace)[:10])
    
