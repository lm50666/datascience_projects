from math import *
import sys
import random
class Node:
    def __init__(self,ime,tree,dubina,najcesci):
        self.ime=ime
        self.tree=tree
        self.dubina=dubina
        self.leaf="no"
        self.najcesci=najcesci

    def getName(self):
        return self.ime
    def getTree(self):
        return self.tree
    def getDubina(self):
        return self.dubina

class Leaf:
    def __init__(self,vrijednost):
        self.vrijednost=vrijednost
        self.leaf="yes"
    def getName(self):
        return self.vrijednost


def entropija(po_kategorijama,ukupno):
    entropija=0

    for k in po_kategorijama.values():
        entropija-=(k/ukupno)*log((k/ukupno),2)
    return entropija
def najcesca_oznaka(D):
    rjecnik={}
    lista_maxi=[]
    for i in D:
        if i[-1] not in rjecnik:
            rjecnik[i[-1]]=1
        else:
            rjecnik[i[-1]]+=1
    maxi=max(rjecnik.values())
    for i in rjecnik:
        if rjecnik[i]==maxi:
            lista_maxi.append(i)
    return min(lista_maxi)
def svi_isti(D):
    lista=[]
    for i in D:
        if i[-1] not in lista:
            lista.append(i[-1])
    return len(lista)==1

def najveca_ig(Dd,X,entropija1):

    rjecnik_entropija = {}
    rjecnik_entropija2={}
    povratni_rjecnik={}
    for x in X:
        indeks=X.index(x)
        lista_vrijednosti=[]

        entr={}

        brojac=len(Dd)

        ig=entropija1

        for red in Dd:
            if red[indeks] not in lista_vrijednosti:
                lista_vrijednosti.append(red[indeks])
        rjecnik_entropija2={}

        for i in lista_vrijednosti:
            ukupno=0
            rjecnik_targeti={}
            brok=0

            for red in Dd:

                if red[indeks]==i and red[-1] not in rjecnik_targeti.keys():
                    rjecnik_targeti[red[-1]]=1
                    ukupno+=1
                elif red[indeks]==i and (red[-1] in rjecnik_targeti.keys()):
                    rjecnik_targeti[red[-1]]+=1
                    ukupno+=1

            entr=entropija(rjecnik_targeti,ukupno)

            rjecnik_entropija2[i]=entr
            ig-=(ukupno/brojac)*entr
        povratni_rjecnik[x]=rjecnik_entropija2.copy()
        rjecnik_entropija[x]=ig
    maxl=max(rjecnik_entropija.values())
    listaa=[]
    for l in rjecnik_entropija:
        if rjecnik_entropija[l]==maxl:
            listaa.append(l)
    listaa.sort()
    povratna=[]
    index=X.index(listaa[0])
    for d in Dd:
        if d[index] not in povratna:
            povratna.append(d[index])

    return listaa[0],povratni_rjecnik,povratna

def printer(stablo):
    lista=[]
    if isinstance(stablo,Leaf):
        print()
        return
    stringic=str(stablo.getDubina())+":"+str(stablo.getName())

    for i in stablo.getTree():
        lista.append(i[1])
    while(len(lista)>0):
        cvor=lista.pop(0)
        if cvor.leaf=="yes":continue
        stringic+=", "+str(cvor.getDubina())+":"+str(cvor.getName())
        for i in cvor.getTree():
            lista.append(i[1])
    print(stringic)
def izbaci(D, X, atribut, x):

    kopija = []
    lista_indexa = []
    konacna = []
    index1 = X.index(x)

    for red in D.copy():
        kopija.append(red)
    brojac=0
    for red in D.copy():
        if atribut != red[index1]:

            lista_indexa.append(brojac)
        brojac+=1
    lista_indexa.reverse()
    for i in lista_indexa:
        kopija.pop(i)
    for k in kopija:
        lista = k.copy()
        lista.pop(index1)
        konacna.append(lista)
    return konacna
def predic(red,stablo,atributi,najcesci):

    cvor=stablo
    while cvor.leaf!="yes":
        flag=0
        index=atributi.index(cvor.getName())
        for i in cvor.getTree():

            if i[0]==red[index]:
                cvor=i[1]
                flag=1
                break
        if flag==0:
            return cvor.najcesci
    return cvor.getName()

def id3(D,Dd,X,y,entropy,dubina,max_depth,vp=''):
    if len(Dd)==0:
        return Leaf(vp)
    v=najcesca_oznaka(Dd)
    if len(X)==0 or svi_isti(Dd):
        return Leaf(v)

    if max_depth!='undefined':
        if dubina>=max_depth:

            return Leaf(v)
    x,entropije,vx=najveca_ig(Dd,X,entropy)
    subtrees=[]
    for vq in vx:

        a=izbaci(Dd,X,vq,x)
        atri=X[:]
        atri.remove(x)
        entroo=entropije[x][vq]
        t=id3(D,a,atri,y,entroo,dubina+1,max_depth,v)
        subtrees.append((vq,t))

    return Node(x,subtrees,dubina,v)
class ID3:
    def __init__(self,mode,model,max_depth):
        self.mode=mode
        self.model=model
        self.max_depth=max_depth


    ######Parsiranje ulaznog csv-fajla
    def fit(self,train_dataset):
        dat=open(train_dataset,'r')
        lines=dat.readlines()
        atributi=lines[0].strip().split(',')
        linije=[]
        for i in lines[1::]:
            line=i.strip().split(',')
            linije.append(line)

        rjecnik_target={}
        for i in linije:
            d=i[-1]
            if d in rjecnik_target:
                rjecnik_target[d]+=1
            else:
                rjecnik_target[d]=1
        ukupno=0
        for i in rjecnik_target.values():
            ukupno+=i
        entropijaa=entropija(rjecnik_target,ukupno)
        self.stablo=id3(linije,linije,atributi[0:-1],rjecnik_target,entropijaa,0,self.max_depth)
        printer(self.stablo)
    def predict(self,test_dataset):
        predictions=[]
        dat=open(test_dataset,'r')
        lin=dat.readlines()
        atributi = lin[0].strip().split(',')
        line=[]
        targeti=[]
        for i in lin[1::]:
            linee=i.strip().split(',')
            line.append(linee)
        najcesci = najcesca_oznaka(line)
        for redak in line:
            predictions.append(predic(redak,self.stablo,atributi,najcesci))

        for kraj in line:
            targeti.append(kraj[-1])
        uspjesno=0
        counter=0
        tar=list(set(targeti))
        tar.sort()

        matrica=[[0 for i in range(len(set(targeti)))]for j in range(len(set(targeti)))]
        for s in predictions:
            print(s,end=' ')
        print()
        for brojac in range(len(targeti)):
            if targeti[brojac]==predictions[brojac]:
                uspjesno+=1
            counter+=1
        print(uspjesno/counter)
        for brojac in range(len(targeti)):
            i=tar.index(targeti[brojac])
            j=tar.index(predictions[brojac])
            matrica[i][j]+=1
        for i in matrica:
            for j in i:
                print(j,end=' ')
            print()

class RandomForest:
    def __init__(self,max_depth,number_of_trees,feature_ratio,example_ratio):
        self.max_depth=max_depth
        self.number_of_trees=number_of_trees
        self.example_ratio = example_ratio
        self.feature_ratio = feature_ratio
    def fit(self,train_dataset):
        dat = open(train_dataset, 'r')
        lines = dat.readlines()
        atributi = lines[0].strip().split(',')
        linije = []
        for i in lines[1::]:
            line = i.strip().split(',')
            linije.append(line)
        instance_sub=round(self.example_ratio*len(linije))
        feature_sub=round(self.feature_ratio*len(atributi[0:-1]))
        self.stabla=[]
        for i in range(self.number_of_trees):
            atri_za_print=[]
            ulazni=[]

            atri=(random.sample(range(0,len(atributi)-1),feature_sub))
            for k in atri:
                atri_za_print.append(atributi[k])
            redovi=[]
            for kk in range(instance_sub):
                redovi.append(random.choice(range(0,len(linije))))
            rjecnik_target={}
            print(*atri_za_print,sep=" ")
            print(*redovi,sep=" ")
            for broj in redovi:
                lina=linije[broj]
                indeksi=atri
                nova=[]
                for ind in indeksi:
                    nova.append(lina[ind])
                nova.append(lina[-1])
                ulazni.append(nova)
            for ui in ulazni:
                d = ui[-1]
                if d in rjecnik_target:
                    rjecnik_target[d] += 1
                else:
                    rjecnik_target[d] = 1
            ukupno = 0

            for g in rjecnik_target.values():
                ukupno += g

            entopija=entropija(rjecnik_target,ukupno)

            stabloo=id3(ulazni,ulazni,atri_za_print,rjecnik_target,entopija,0,self.max_depth)
            self.stabla.append(stabloo)
    def predict(self,test_dataset):
        dat=open(test_dataset,'r')
        lin=dat.readlines()
        atributi = lin[0].strip().split(',')
        line=[]
        for i in lin[1::]:
            linee=i.strip().split(',')
            line.append(linee)
        najcesci = najcesca_oznaka(line)
        uspjesno = 0
        counter = 0
        targeti=[]
        for kraj in line:
            targeti.append(kraj[-1])
        tar = list(set(targeti))
        tar.sort()
        matrica = [[0 for i in range(len(set(targeti)))] for j in range(len(set(targeti)))]
        rezultati=[]
        for redak in line:
            rjecnik={}
            for drvo in self.stabla:
                a=predic(redak,drvo,atributi,najcesci)

                if a not in rjecnik:
                    rjecnik[a]=1
                else:
                    rjecnik[a]+=1

            maksi=max(rjecnik.values())
            lis=[]
            for rijec in rjecnik:
                if rjecnik[rijec]==maksi:
                    lis.append(rijec)
            lis.sort()
            rezultat=lis[0]

            rezultati.append(rezultat)
        for rez in rezultati:
            print(rez,end=" ")

        print()
        for brojac in range(len(targeti)):
            if targeti[brojac] ==rezultati[brojac]:
                uspjesno += 1
            counter += 1
        print(uspjesno / counter)
        for brojac in range(len(targeti)):
            i = tar.index(targeti[brojac])
            j = tar.index(rezultati[brojac])
            matrica[i][j] += 1
        for i in matrica:
            for j in i:
                print(j, end=' ')
            print()


def main():
    fit_datoteka=sys.argv[1]
    train_datoteka=sys.argv[2]
    config_datoteka=sys.argv[3]
    dat2=open(config_datoteka,'r')
    dat2=dat2.readlines()
    linije=[]
    rjecnik_konfig={}
    for line in dat2:
        redak=line.strip().split("=")
        rjecnik_konfig[redak[0]]=redak[1]
    if rjecnik_konfig['model']=='ID3':
        if 'max_depth' in rjecnik_konfig and rjecnik_konfig['max_depth']=='-1':
            maxdep='undefined'
        elif 'max_depth' not in rjecnik_konfig:
            maxdep='undefined'
        else:
            maxdep=int(rjecnik_konfig['max_depth'])
        a = ID3('test', 'test',maxdep)
        a.fit(fit_datoteka)
        a.predict(train_datoteka)
    if rjecnik_konfig['model']=='RF':
        if 'max_depth' in rjecnik_konfig and rjecnik_konfig['max_depth']=='-1':
            maxdep='undefined'
        elif 'max_depth' not in rjecnik_konfig:
            maxdep='undefined'
        else:
            maxdep=int(rjecnik_konfig['max_depth'])
        feature_ratio=float(rjecnik_konfig['feature_ratio'])
        example_ratio=float(rjecnik_konfig['example_ratio'])
        number_of_trees=int(rjecnik_konfig['num_trees'])
        model=RandomForest(maxdep,number_of_trees,feature_ratio,example_ratio)
        model.fit(fit_datoteka)
        model.predict(train_datoteka)
main()