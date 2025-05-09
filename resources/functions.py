import pandas as pd
import ast
import os

def itu_to_tsv(input_csv, output_tsv):
    df = pd.read_csv(input_csv)
    
    # check if the output directory exists, if not create it
    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)

    with open(output_tsv, 'w', encoding='utf-8') as fout:

        # iterate through each row in the DataFrame
        for _, row in df.iterrows():
            switch = False
            try:
                # Extract the relevant columns
                tokens = ast.literal_eval(row['tokenized_sentence'])
                indices = ast.literal_eval(row['indices'])
                language = row['language']

                # assign O tags to all tokens
                tags = ['O'] * len(tokens)
                # this sentence is not idiomatic
                if indices != [-1]:
                    # assign B-IDIOM tag to the first token then I-IDIOM to the rest
                    for i in indices:
                        if not switch:
                            tags[i] = 'B-IDIOM'
                            switch = True
                        else:
                            tags[i] = 'I-IDIOM'

                # add a space between each sentence
                for token, tag in zip(tokens, tags):
                    fout.write(f"{token}\t{tag}\t{language}\n")
                fout.write("\n")

            # catch errors
            except Exception as e:
                print(f"Error in row: {e}")

def itu_to_tsv_test(input_csv, output_tsv):
    df = pd.read_csv(input_csv)
    
    # check if the output directory exists, if not create it
    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)

    with open(output_tsv, 'w', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            # Extract the relevant columns
            tokens = ast.literal_eval(row['tokenized_sentence'])
            indices = ["" for _ in range(len(tokens))]
            language = row['language']

            # add a space between each sentence
            for token, index in zip(tokens, indices):
                fout.write(f"{token}\t{index}\t{language}\n")
            fout.write("\n")
            

def parse_cupt(file_path):
    """
    Parse a CUpt file, splitting on '_' **only** when both sides are non-empty.
    Returns List of sentences, each a list of (word, is_idiom) tuples.
    """
    sentences = []
    sentence = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                if not line and sentence:
                    sentences.append(sentence)
                    sentence = []
                continue

            cols = line.split("\t")
            form = cols[1]
            is_idiom = form.endswith("*")
            if is_idiom:
                form = form[:-1]

            # only split when parts are both non-empty
            if "_" in form:
                parts = form.split("_")
                if len(parts) > 1 and all(len(p) > 0 for p in parts):
                    for part in parts:
                        sentence.append((part, is_idiom))
                else:
                    continue
            else:
                sentence.append((form, is_idiom))

    if sentence:
        sentences.append(sentence)
    return sentences


def transform_cupt_to_tsv(input_path, output_path, lang):
    data = parse_cupt(input_path)

    with open(output_path, "w", encoding="utf-8") as out:
        for sentence in data:
            in_idiom = False
            for word, is_idiom in sentence:
                # now 'word' is a string, and 'is_idiom' is the bool flag
                if not is_idiom:
                    tag = "O"
                else:
                    if not in_idiom:
                        tag = "B-IDIOM"
                        in_idiom = True
                    else:
                        tag = "I-IDIOM"

                # write only strings
                out.write(word + "\t" + tag + "\t" + lang + "\n")
            out.write("\n")


def combine_tsv_files(tr_path, it_path, output_path):
    # read the two TSV files and combine them
    with open(tr_path, "r", encoding="utf-8") as f_tr, \
         open(it_path, "r", encoding="utf-8") as f_it, \
         open(output_path, "w", encoding="utf-8") as f_out:

        tr_lines = f_tr.readlines()
        it_lines = f_it.readlines()
        f_out.writelines(tr_lines + it_lines)

def combine_all_tsv_files(tsv1, tsv2, tsv3, output_path):
    # read the two TSV files and combine them
    with open(tsv1, "r", encoding="utf-8") as itu, \
         open(tsv2, "r", encoding="utf-8") as parsame, \
         open(tsv3, "r", encoding="utf-8") as id10m, \
         open(output_path, "w", encoding="utf-8") as f_out:

        itu_lines = itu.readlines()
        parsame_lines = parsame.readlines()
        id10m_lines = id10m.readlines()

        f_out.writelines(itu_lines + parsame_lines + id10m_lines)

def add_language(path, outpath, lang):
    # add language to ID10M data
    with open(path, "r", encoding="utf-8") as f, open(outpath, "w", encoding="utf-8") as out_f:
        for line in f:
            if line.strip() == "":
                out_f.write("\n")
            else:
                out_f.write(line.strip() + "\t" + lang + "\n")