import gzip
# fa_in = open("apex_rip_cDNA_screened.fa", "r")
# fa_Info = []
# fa_Seq = []
# fa_Num = -1
# fa_name=[]

# for line in fa_in.readlines():
#     line = line.rstrip()
#     if line[0] == ">":
#         fa_Info.append(line[2:17])
#         fa_Num = fa_Num + 1
#         fa_Seq.append("")
#     else:
#         fa_Seq[fa_Num] = fa_Seq[fa_Num] + line



# for i in range(len(apex_info)) :
#     apex_name = apex_info[i][0][0:15]
#     apex_info[i].insert(1,"null")
#     for j in range(len(fa_Info)):
#         if apex_name == fa_Info[j]:
#             apex_info[i][1] = fa_Seq[j]
#             break
# count = 0
# for l in apex_info:
#     if l[1]=="null":
#         count +=1
# print(count)


CHROMOSOMES = [str(i + 1) for i in range(22)] + ['X', 'Y']

CHR_link="./homo_dna/Homo_sapiens.GRCh38.dna.chromosome.chr_N.fa.gz"

apex = open("news.txt","r")

apex_info = []
apex_seq=[]
for line in apex.readlines():
    line = line.rstrip()
    apex_info.append(line.split(" "))

lost=[]
count = 1
f = open('apex_seq.txt','w')
for ap in apex_info:
    if ap[1] == "null":
        lost.append(ap[0])
        continue
    chr_N = ap[1]
    start = int(ap[2])
    end = int(ap[3])
    now_link=CHR_link.replace('chr_N',chr_N)
    temp =[ap[0]]

    #f.write(">"+temp[0])
    opener = gzip.open if now_link.endswith(".gz") else open
    with opener(now_link) as source:
        for line in source.readlines():
            if line[0] == ">":
                continue
            temp.append(line[start-1:end])
            print(str(line))
    apex_seq.append(temp)
    print("cpmplete"+str(count))
    count+=1

# f = open('apex_seq.txt','w')
# for i in apex_seq :
#     s=""
#     for j in range(len(i)):
#         s=s+i[j]+" "
#     f.write(s+"\n")
# l = open('lost.txt','w')
# for i in apex_seq :
#     s=""
#     for j in range(len(i)):
#         s=s+i[j]+" "
#     f.write(s+"\n")
