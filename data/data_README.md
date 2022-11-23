### Data Collection

#### Data Sources
We collected data from four different data sets, compared and pre-processed these data to obtain the data we need and available.

<img width="647" alt="截圖 2022-11-23 下午8 58 50" src="https://user-images.githubusercontent.com/29274119/203554136-43b7f7a0-f05c-43ae-b47e-6807fc523e24.png">



<br>

#### Data Processing

Using NCBI API and Ensembl Biomart can convert labels and obtain the data we need, such as protein sequences, and can also divide gene sequences into CDS, 3'UTR and 5'UTR.Then compare these data and do some sequence similarity comparisons to find the sequence we need.In this project we will compare with the Apex dataset and Cefra dataset in RNATracker.


<img width="408" alt="截圖 2022-11-23 下午9 25 38" src="https://user-images.githubusercontent.com/29274119/203558208-dcb9d2be-513a-42b0-b9fa-03a1316c958c.png">

<img width="519" alt="截圖 2022-11-23 下午9 25 44" src="https://user-images.githubusercontent.com/29274119/203558319-d9486c75-c498-4c0a-83c9-2d47727a3158.png">

<br>


#### Reference

[1]RNALocate: a resource for RNA subcellular localizations. Nucleic Acids Res. 2017 Jan 4,Zhang T, Tan P, Wang L, Jin N, Li Y, Zhang L, Yang H, Hu Z, Zhang L, Hu C, Li C, Qian K, Zhang C, Huang Y, Li K, Lin H, Wang D. 
[2]Live-cell mapping of organelle-associated RNAs via proximity biotinylation combined with protein-RNA crosslinking, eLife 2017;6, Pornchai Kaewsapsak, David Michael Shechner, William Mallard, John L Rinn, Alice Y Ting 
[3]CeFra-seq reveals broad asymmetric mRNA and noncoding RNA distribution profiles in Drosophila and human cells. RNA. 2018 Jan ,Benoit Bouvrette LP, Cody NAL, Bergalet J, Lefebvre FA, Diot C, Wang X, Blanchette M, Lécuyer E. 
[4]Atlas of Subcellular RNA Localization Revealed by APEX-Seq,Cell,Volume 178, Issue 2,2019 ,Furqan M. Fazal, Shuo Han, Kevin R. Parker, Pornchai Kaewsapsak, Jin Xu, Alistair N. Boettiger, Howard Y. Chang, Alice Y. Ting
