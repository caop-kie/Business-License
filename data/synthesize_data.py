import os
import random

chinese_name_file_loc = '/Users/panfengcao/Chinese-Names-Corpus/Chinese_Names_Corpus/Chinese_Names_Corpus（120W）.txt'
chinese_name_file_dst = '/data/corpus/legal_person.txt'
company_name_file_loc = '/Users/panfengcao/Company-Names-Corpus/Company-Names-Corpus（480W）.txt'
company_name_file_dst = '/data/corpus/name.txt'


def synthesize(entity_type, from_loc, to_loc):
    if entity_type == 'company_name' or entity_type == 'people_name':
        with open(from_loc, encoding='utf-8') as f:
            data = f.read()
            lines = data.split('\n')
            random.shuffle(lines)
        with open(to_loc, 'w', encoding='utf-8') as f:
            f.writelines(lines[0:1000])


if __name__ == '__main__':
    synthesize('company_name', company_name_file_loc, company_name_file_dst)