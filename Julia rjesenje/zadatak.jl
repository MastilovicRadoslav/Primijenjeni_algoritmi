#PR106/2020 Mastilović Radoslav

using StatsModels
using GLM
using DataFrames
using CSV
using Lathe.preprocess: TrainTestSplit
using Plots
using Statistics
using StatsBase
using MLBase
using ROC

# 1. PRIPREMA I PROVJERA PODATAKA!

# UČitavanje podataka!
data = DataFrame(CSV.File("pacijenti.csv"))
println("Podaci:")
display(describe(data))
# Podjela na skup za obuku i testiranja:
dataTrain, dataTest = TrainTestSplit(data,.80)

# 2. LOGISTIČKA REGRESIJA!
println()
println("LOGISTICKA REGRESIJA!")
println()
# Formiranje formule za logistički regresor:
f1 = @formula(bolesni ~ pol+visina+tezina+dbp+sbp) #Bolesni
# Poziv logistički regresora
logisticRegressorBolesni = glm(f1, dataTrain, Binomial(), ProbitLink()) 
# Testranje podataka logistickom regresijom
dataPredictedTestBolesni = predict(logisticRegressorBolesni, dataTest)
#Ispis podataka
println("Bolesni: $(dataTest.bolesni))")
println("Predvidjeni podaci za bolesne: $(round.(dataPredictedTestBolesni; digits = 2))")
println()



# Formiranje formule za logistički regresor
f2 = @formula(zdravi ~ pol+visina+tezina+dbp+sbp) #Zdravi
# Poziv logistički regresora
logisticRegressorZdravi = glm(f2, dataTrain, Binomial(), ProbitLink()) 
# Testranje podataka logistickom regresijom
dataPredictedTestZdravi = predict(logisticRegressorZdravi, dataTest)
#Ispis podataka
println("Zdravi: $(dataTest.zdravi))")
println("Predvidjeni podaci za zdrave: $(round.(dataPredictedTestZdravi; digits = 2))")
println()


# Formiranje formule za logistički regresor
f3 = @formula(nijePoznato ~ pol+visina+tezina+dbp+sbp) #nijePoznato
# Poziv logistički regresora
logisticRegressorNijePoznato = glm(f3, dataTrain, Binomial(), ProbitLink()) 
# Testranje podataka logističkom regresijom
dataPredictedTestNijePoznato = predict(logisticRegressorNijePoznato, dataTest)
#Ipis podataka
println("Nije poznato: $(dataTest.nijePoznato))")
println("Predvidjeni podaci za nijePoznato: $(round.(dataPredictedTestNijePoznato; digits = 2))")
println()

# 3. ONE-VERSUS-ONE ALGORITAM!
println("ONE-VERSUS-ONE ALGORITAM!\n")
#Provjere koji podaci pripadaju kojoj klasi:

#BOLESNI

dataPredictedTestClassBolesni = repeat(0:0, length(dataPredictedTestBolesni)) 
#Formiranje matrice
for i in 1:length(dataPredictedTestBolesni)
    if (dataPredictedTestBolesni[i] > 0.5)
        dataPredictedTestClassBolesni[i] = 1
    end
end
#Pomoćni niz za indexe podataka koji pripadaju  klasi bolesni
indexBolesni = [] 
#Punjenje pomoćnog niza sa indexima
for i in 1 : length(dataPredictedTestClassBolesni)
    if dataPredictedTestClassBolesni[i] == 1
        append!(indexBolesni, i)
    end
end
println("Indeksi podataka koji pripadaju klasi bolesni:")
#Ispis indexa podataka koji pripadaju klasi bolesni
println("$indexBolesni") 


#ZDRAVI
#Matrica za indekse
dataPredictedTestClassZdravi = repeat(0:0, length(dataPredictedTestZdravi))
#Formiranje matrice
for i in 1:length(dataPredictedTestZdravi)
    if (dataPredictedTestZdravi[i] > 0.5)
        dataPredictedTestClassZdravi[i] = 1
    end
end
#Pomoćni niz za indexe podataka koji pripadaju klasi zdravi
indexZdravi = [] 
#Punjenje pomoćnog niza sa indexima
for i in 1 : length(dataPredictedTestClassZdravi)
    if dataPredictedTestClassZdravi[i] == 1
        append!(indexZdravi, i)
    end
end
println("Indeksi podataka koji pripadaju klasi zdravi:")
#Ispis indexa podataka koji pripadaju klasi zdravi
println("$indexZdravi") 


#NIJE POZNATO
#Matrica za indekse
dataPredictedTestClassNijePoznato = repeat(0:0, length(dataPredictedTestNijePoznato)) 
#Formiranje matrice
for i in 1:length(dataPredictedTestNijePoznato)
    if (dataPredictedTestNijePoznato[i] > 0.5)
        dataPredictedTestClassNijePoznato[i] = 1
    end
end
#Pomoćni niz za indexe podataka koji pripadaju klasi nijePoznato
indexNijePoznato = [] 
# punjenje pomoćnog niza sa indexima
for i in 1 : length(dataPredictedTestClassNijePoznato)
    if dataPredictedTestClassNijePoznato[i] == 1
        append!(indexNijePoznato, i)
    end
end
println("Indeksi podataka koji pripadaju klasi nijePoznato:")
#Ispis indexa podataka koji pripadaju klasi nije poznato
println("$indexNijePoznato\n") 

#Promjenljive
brojPodatakaKojiPripadajuBolesnim = 0
brojPodatakaKojiPripadajuZdravim = 0
brojPodatakaKojiPripadajuNijePoznato = 0

#Prebrojavanje koliko ima podataka koji pripadaju klasi Bolesni
for i in 1:length(dataPredictedTestBolesni)
    if (dataPredictedTestBolesni[i] > 0.5)
        global brojPodatakaKojiPripadajuBolesnim += 1
    end 
end
#Ispis broja bolesnih
println("Broj podataka koji pripadaju klasi Bolesni je: $(brojPodatakaKojiPripadajuBolesnim)")


#Prebrojavanje koliko ima podataka koji pripadaju klasi Zdravi

for i in 1:length(dataPredictedTestZdravi)
    if (dataPredictedTestZdravi[i] > 0.5)
        global brojPodatakaKojiPripadajuZdravim += 1
    end 
end
#Ispis broja zdravih:

println("Broj podataka koji pripadaju klasi Zdravi je: $(brojPodatakaKojiPripadajuZdravim)")


#Prebrojavanje koliko ima podataka koji pripadaju klasi Nije poznato

for i in 1:length(dataPredictedTestNijePoznato)
    if (dataPredictedTestNijePoznato[i] > 0.5)
        global brojPodatakaKojiPripadajuNijePoznato += 1
    end 
end

#Ispis broja Nije poznato:

println("Broj podataka koji pripadaju klasi nijePoznato je: $(brojPodatakaKojiPripadajuNijePoznato)")


# Provera kojoj klasi pripada najviše podataka
dataPredictedTest = [] # pomoćni niz, u koji smeštamo one podatke koje izaberemo
podaciZaTest = [] # koju kolonu uzimamo iz dataTest

if  brojPodatakaKojiPripadajuBolesnim > brojPodatakaKojiPripadajuZdravim
    if brojPodatakaKojiPripadajuBolesnim > brojPodatakaKojiPripadajuNijePoznato  
        global dataPredictedTest = dataPredictedTestBolesni #Ubacujemo niz predvidjenih podataka za klasu Bolesni u pomocni niz
        global podaciZaTest = dataTest.bolesni               #Podatke iz kolone bolesni smjestamo u nazivKolone niz
        println("Najviše podataka pripada klasi bolesni, pa nju uzimamo!")
    end
elseif  brojPodatakaKojiPripadajuZdravim > brojPodatakaKojiPripadajuBolesnim
    if brojPodatakaKojiPripadajuZdravim > brojPodatakaKojiPripadajuNijePoznato  
        global dataPredictedTest = dataPredictedTestZdravi #Ubacujemo niz predvidjenih podataka za klasu Bolesni u pomocni niz
        global podaciZaTest = dataTest.zdravi                #Podatke iz kolone zdravi smjestamo u nazivKolone niz
        println("Najviše podataka pripada klasi zdravi, pa nju uzimamo!")
    end
elseif  brojPodatakaKojiPripadajuNijePoznato > brojPodatakaKojiPripadajuBolesnim
     if brojPodatakaKojiPripadajuNijePoznato > brojPodatakaKojiPripadajuZdravim  
        global dataPredictedTest = dataPredictedTestNijePoznato #Ubacujemo niz predvidjenih podataka za klasu Bolesni u pomocni niz
        global podaciZaTest = dataTest.nijePoznato           #Podatke iz kolone zdravi smjestamo u nazivKolone niz
        println("Najviše podataka pripada klasi nijePoznato, pa nju uzimamo!")
    end
end
println()

println("Ispis klase koja ima najvise pozitivnih klasifikacija:$(podaciZaTest)")
println()


c =  cor(data.dbp, data.sbp)
println("Koeficijent korelacije je: $c")
if c>0.9
   println("Postoji veoma jaka veza izmedju podataka")
elseif c>0.7
   println("Postoji jaka veza izmedju podataka ")
elseif c>0.5
   println("Postoji umerena veza izmedju podataka ")
else
   println("Veza izmedju podataka nije dovoljno jaka")
end

covDbpSbp = cov(data.dbp, data.sbp)
println("Cov za gornji i donji pritisak je: $covDbpSbp")
println()

# 4. Analiza kvaliteta modela
println("Ocjenjivanje kvaliteta modela:")

# kreiranje matrice
dataPredictedTestClass = repeat(0:0, length(dataPredictedTest))

for i in 1:length(dataPredictedTest)
    if (dataPredictedTest[i] < 0.5)
        dataPredictedTestClass[i] = 0
    elseif dataPredictedTest[i] >= 0.5
        dataPredictedTestClass[i] = 1
    end
end

FPTest = 0 # false positives
FNTest = 0 # false negatives
TPTest = 0 # true positives
TNTest = 0 # true negatives

# dodela vrednosti za FPTest, FNTest, TPTest i TNTest
for i in 1:length(dataPredictedTestClass)
    if podaciZaTest[i] == 0 && dataPredictedTestClass[i] == 0
        global TNTest +=1
    elseif podaciZaTest[i] == 0 && dataPredictedTestClass[i] == 1
        global FPTest +=1
    elseif podaciZaTest[i] == 1 && dataPredictedTestClass[i] == 0
        global FNTest +=1
    elseif podaciZaTest[i] == 1 && dataPredictedTestClass[i] == 1
        global TPTest +=1
    end
end

# accuracy (preciznost) = (TP+TN)/(TP+TN+FP+FN) = (TP+TN)/(P+N)
accuracyTest = (TPTest + TNTest) / (TPTest + TNTest + FPTest + FNTest)

# sensitivity (osetljivost, True positive rates) = TP/(TP+FN) = TP/P
sensitivityTest = TPTest / (TPTest + FNTest)

# specificity (specifičnost, True negative rates) = TN/(TN+FP) = TN/N
specificityTest = TNTest / (TNTest + FPTest)


println("TP = $TPTest, FP = $FPTest, TN = $TNTest, FN = $FNTest")

println("Preciznost za test skup je $accuracyTest")

println("Osetljivost za test skup je $sensitivityTest")

println("Specificnost za test skup je $specificityTest")

# Roc kriva
rocTest = ROC.roc(dataPredictedTest, podaciZaTest, true)

aucTest = AUC(rocTest) # predstavlja objektivnu meru kvaliteta klasifikatora
println("Povrsina ispod krive u procentima je: $aucTest\n")

if (aucTest > 0.9)
    println("Klasifikator je jako dobar")
elseif (aucTest > 0.8)
    println("Klasifikator je veoma dobar")
elseif (aucTest > 0.7)
    println("Klasifikator je dosta dobar")
elseif (aucTest > 0.5)
    println("Klasifikator je relativno dobar")
else
    println("Klasifikator je los")
end

