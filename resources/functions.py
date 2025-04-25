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

def parse_cupt(file_path):
    sentences = []
    sentence = []
    
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith("#") or line == "":
                # if end of sentence and sentence is not empty, add it to sentences and reset
                if line == "" and sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            # Split the line into columns and append to the current sentence
            columns = line.split('\t')
            sentence.append(columns)
    
    if sentence:
        sentences.append(sentence)
    
    return sentences


def tranform_cupt_to_tsv(input_path, output_path, lang):
    
    cupt_data = parse_cupt(input_path)

    with open(output_path, "w", encoding="utf-8") as file:
        for sentence in cupt_data:
            switch = False
            for token in sentence:
                # Skip empty tokens
                if token[1] == "_":
                    continue
                # add word to the file
                file.write(token[1] + "\t")
                # if token ends with * then its not idiom
                if token[-1] =="*":
                    file.write("O\t")
                # first token will get B-IDIOM and the rest I-IDIOM
                elif token[-1] != "O" and switch == False:
                    file.write("B-IDIOM\t")
                    switch = True
                else:
                    file.write("I-IDIOM\t")
                file.write(lang + "\n")
            # add a space between each sentence
            file.write("\n")

def combine_tsv_files(tr_path, it_path, output_path):
    # read the two TSV files and combine them
    with open(tr_path, "r", encoding="utf-8") as f_tr, \
         open(it_path, "r", encoding="utf-8") as f_it, \
         open(output_path, "w", encoding="utf-8") as f_out:

        tr_lines = f_tr.readlines()
        it_lines = f_it.readlines()
        f_out.writelines(tr_lines + it_lines)

def add_language(path, outpath, lang):
    # add language to ID10M data
    with open(path, "r", encoding="utf-8") as f, open(outpath, "w", encoding="utf-8") as out_f:
        for line in f:
            if line.strip() == "":
                out_f.write("\n")
            else:
                out_f.write(line.strip() + "\t" + lang + "\n")