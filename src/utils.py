import stanza

def initialize(use_gpu=True):
    tr_nlp = stanza.Pipeline("tr", processors="tokenize,mwt,lemma", use_gpu=use_gpu)
    it_nlp = stanza.Pipeline("it", processors="tokenize,mwt,lemma", use_gpu=use_gpu)
    return tr_nlp, it_nlp

def get_idioms(dataset, tagger_dict):
    idioms = []

    for tokens, tags, langs in dataset:
        idiom = ""

        # Group tokens by language
        lang_tokens = {}
        for idx, lng in enumerate(langs):
            lang_tokens.setdefault(lng, []).append((idx, tokens[idx]))

        # Process each language once
        lemmas = [None] * len(tokens)
        for lng, idx_token_pairs in lang_tokens.items():
            sentence = " ".join([tok for _, tok in idx_token_pairs])
            doc = tagger_dict[lng](sentence)
            extracted_lemmas = [word.lemma for sent in doc.sentences for word in sent.words]

            # Match lemmas to original token positions
            for (idx, _), lemma in zip(idx_token_pairs, extracted_lemmas):
                lemmas[idx] = lemma

        # Now build the idiom
        for lemma, tag in zip(lemmas, tags):
            if tag in {1, 2} and lemma:
                idiom += lemma + " "

        if idiom.strip():
            idioms.append(idiom.strip())

    return idioms


def overlap_percentage_l1_in_l2(list1, list2):
    count_in = 0
    for elem in list1:
        if elem in list2:
            count_in += 1
    
    return count_in/len(list1)