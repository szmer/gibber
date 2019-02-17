from wsd_config import vecs_path

first_line = True
word_n = 0
word_to_idx = {} # ie. indices of vectors
idx_to_word = {}

# we'll read those from the data file
vecs_count = 0
vecs_dim = 0

print('Loading word vectors from {}.'.format(vecs_path))
# Get the vector word labels (we'll get vectors themselves in a moment).
with open(vecs_path) as vecs_file:
    for line in vecs_file:
        if first_line:
            # Read metadata.
            vecs_count = int(line.split(' ')[0])
            vecs_dim = int(line.split(' ')[1])
            first_line = False
            continue
        # Read lemma base forms.
        word = line.split(' ')[0].lower()
        word_to_idx[word] = word_n
        idx_to_word[word_n] = word
        word_n += 1

word_vecs = np.loadtxt(vecs_path, encoding="utf-8",
                       dtype=np.float32, # tensorflow's requirement
                       skiprows=1, usecols=tuple(range(1, vecs_dim+1)))

# Add the dummy boundary/unknown marker.
word_vecs = np.vstack([word_vecs, np.zeros((1,vecs_dim), dtype=np.float32)])
vecs_count += 1

# Get the word's vector, or the dummy marker.
def word_id(word):
    return word_to_idx[word] if word in word_to_idx else vecs_count-1

# We need a special token for cases when the target word is near the start or end of sentence.
bound_token_id = vecs_count - 1 # the zero additional vector
