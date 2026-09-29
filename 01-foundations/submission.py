import collections

############################################################
# Problem 3a

def computeMaxWordLength(text):
    """
    Given a string |text|, return the longest word in |text|.  If there are
    ties, choose the word that comes latest in the alphabet.
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    
    words=text.split()

    return max(words, key=lambda w:(len(w), w))

    # END_YOUR_CODE

############################################################
# Problem 3b

def manhattanDistance(loc1, loc2):
    """
    Return the Manhattan distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)

    return abs(loc1[0]-loc2[0])+abs(loc1[1]-loc2[1])

    # END_YOUR_CODE

############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    High-level idea: generate sentences similar to a given sentence.
    Given a sentence (sequence of words), return a list of all possible
    alternative sentences of the same length, where each pair of adjacent words
    also occurs in the original sentence. (The words within each pair should appear 
    in the same order in the output sentence as they did in the orignal sentence.)
    Notes:
    - The order of the sentences you output doesn't matter.
    - You must not output duplicates.
    - Your generated sentence can use a word in the original sentence more than
      once.
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)

    word=sentence.split()
    pair={}
    result=set()

    for i in range(0, len(word)-1):
        if word[i] not in pair:
            pair[word[i]]=[]
        pair[word[i]].append(word[i+1])

    def backTrack(current, last, length):
        if len(current)==length:
            result.add(' '.join(current))
            return
    
        if last in pair:
            for i in pair[last]:
                current.append(i)
                backTrack(current, i, length)
                current.pop()

    for i in word:
        backTrack([i], i, len(word))

    return list(result)

    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    
    for key in v1:
        if key in v2:
            dot=v1[key]*v2[key]
            
    return dot
    
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    
    for key, value in v2.items():
        v1[key]+=value*scale

    return v1
    
    # END_YOUR_CODE

############################################################
# Problem 3f

from collections import Counter

def computeMostFrequentWord(text):
    """
    Splits the string |text| by whitespace and returns two things as a pair: 
        the set of words that occur the maximum number of times, and
    their count, i.e.
    (set of words that occur the most number of times, that maximum number/count)
    You might find it useful to use collections.defaultdict(float).
    """
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    
    word=text.split()
    count=Counter(word) 

    Max=max(count.values())
    freq={word for word, count in count.items() if count==Max}
    
    return freq, Max
    
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindrome(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    
    if len(text)==0:
        return 0
    else:
        t=text

    dp=[[0]*len(t) for _ in range(len(t))]

    for i in range(0, len(t)):
        dp[i][i]=1

    for length in range(2, len(t)+1):
        for i in range(0, len(t)-length+1):
            j=i+length-1
            if t[i]==t[j]:
                dp[i][j]=dp[i+1][j-1]+2
            else:
                dp[i][j]=max(dp[i+1][j], dp[i][j-1])

    return dp[0][len(t)-1]

    # END_YOUR_CODE