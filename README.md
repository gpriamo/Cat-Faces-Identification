# BiometricProject
Biometric Systems Project 2020


### Dependencies ###
- opencv-contrib-python       
- opencv-python
- PIL 

## TODO ##
### Phase 1 ###
- [x] Download immagini e creazione gallery
- [ ] Lavoro
### Phase 2 ### 
- [ ] Performance evaluation (all vs all matrix)
1) K-Fold cross validation (see Lesson2-bis)
2) All against all distance matrix for identification open set (see Lesson2-bis)


3 datset
	- Raw
	- Best
	- Best aligned

Test per iperparametri
	Scegliere J configurazioni* per ogni modello
		performance evaluation per mostrare le differenze


Eigenfaces:
	10, 80, (1000), DEFAULT
Fisher:
	10, 80, (1000), DEFAULT
LBP:
	for grid in [(8,8), (12,12)]:
		radius, neighbors = DEFAULT (1,8) - (1,4) - (2,8)  - (2,12)  - (2,16)

Prendere la configurazione migliore per ognuno dei 3 modelli
	plottare insieme le curve 


Rate da plottare:
	FRR
	FAR
	GRR
	ROC
	
### Phase 3 ### 
- [ ] Report