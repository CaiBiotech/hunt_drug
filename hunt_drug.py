# -*- coding:utf-8 -*-

__cmddoc__ = """

hunt_drug.py - Reads an SDFile and Find similar drug using RDKIT tools
#Contact: haiming_cai@hotmail.com - 2022 - CHINA-VMI

Basic usage: 
python hunt_drug.py -ref ref.sdf -db TEST.sdf
python hunt_drug.py -ref ref.smi -db TEST.smi
python hunt_drug.py -ref ref.smi -db TEST.smi
python hunt_drug.py -ref ref.smi -db TEST.smi -k 0.5 -t 100
python hunt_drug.py -ref ref.sdf -db TEST.sdf -k 0.5 -t 100

Info. Please used the Canonical SMILES download from PubChem, and the SDF file with 2D structure.

""" 
## ������ذ�
import argparse
import pprint
import sys
import os
import subprocess as sp
from io import StringIO
import pandas as pd
import numpy as np
import math
import logging
import datetime
from rdkit import RDLogger
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import rdDepictor
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.ChemicalFeatures import BuildFeatureFactory
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

def sdf2smi(sdflist, key1):
     mylogger = RDLogger.logger()
     mylogger.setLevel(val=RDLogger.ERROR)

     #suppl = []
     #for mol in Chem.SDMolSupplier(sdflist)
     #    if mol:
     #       try:
     #          suppl.append(mol)
     #       except:
     #          pass 

     suppl = [ mol for mol in Chem.SDMolSupplier(sdflist) ]

     ID_list = []
     smiles_list = []
     for mol in suppl:
         if mol:
             name = mol.GetProp(key1)
             ID_list.append(name)
             try:
                smi = Chem.MolToSmiles(mol,isomericSmiles=False)
                smiles_list.append(smi)
             except:
                pass
     mol_datasets = pd.DataFrame({'smiles':smiles_list,'ID':ID_list})
     #mol_datasets.to_csv(os.path.join('./',"TMDB.smi"), sep='\t', index=False, header=False, encoding='utf-8')
     mol_datasets = mol_datasets[['smiles','ID']]
     return mol_datasets

def smi2list(smifile, key="_Name"):
     #mylogger = RDLogger.logger()
     #mylogger.setLevel(val=RDLogger.ERROR)

     #print(smifile)
     suppl = Chem.SmilesMolSupplier(smifile, delimiter='\t', titleLine=False, nameColumn=1)
     ID_list = []
     smiles_list = []
     #print(suppl)
     for mol in suppl:
        if mol:
           try:
              smi = Chem.MolToSmiles(mol,isomericSmiles=False)
              #print(smi)
              smiles_list.append(smi)
              name = mol.GetProp(key)
              #print(name)
              ID_list.append(name) 
           except:
              pass

     mol_datasets = pd.DataFrame({'smiles':smiles_list,'ID':ID_list})
     mol_datasets = mol_datasets[['smiles','ID']]
     return mol_datasets

def getHybridFP(fp1, fp2):
     newfp = DataStructs.ExplicitBitVect(fp1.GetNumBits()+fp2.GetNumBits())
     newfp.SetBitsFromList(fp1.GetOnBits())
     bits2 = [i+nbits for i in fp2.GetOnBits()]
     newfp.SetBitsFromList(bits2)
     return newfp

def smi2fps_dict(smilist, nbits, longbits, output='result_total.fps'):
     #print(smilist)
     cordict = {}
     n = 0
     for index,value in smilist.iterrows():
         name = value['ID']
         #print(name)
         smiles = value['smiles']
         #print(smiles)
         mol = Chem.MolFromSmiles(smiles)
         #print(mol)
         if mol:
            fpdict = {}
            fpdict['Name'] = name
            fpdict['Smiles'] = smiles
            # Morganָ����ԭ�ӶԺ�����Ťתһ����Ĭ�����ϵ��ʹ�ü���������Ҳ���Խ����Ǽ���Ϊλ����
            fpdict['ecfp0'] = AllChem.GetMorganFingerprintAsBitVect(mol, 0, nBits=nbits)
            fpdict['ecfp2'] = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=nbits)
            fpdict['ecfp4'] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
            fpdict['ecfp6'] = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nbits)
            # ͨ����Morgan�㷨Ӧ����һ���û��ṩ��ԭ�Ӳ���ʽ�����Թ�����һϵ�е�ָ�ơ�����Morganָ��ʱ���������ṩָ�Ƶİ뾶
            fpdict['ecfc0'] = AllChem.GetMorganFingerprint(mol, 0)
            fpdict['ecfc2'] = AllChem.GetMorganFingerprint(mol, 1)
            fpdict['ecfc4'] = AllChem.GetMorganFingerprint(mol, 2)
            fpdict['ecfc6'] = AllChem.GetMorganFingerprint(mol, 3)
            fpdict['fcfp2'] = AllChem.GetMorganFingerprintAsBitVect(mol, 1, useFeatures=True, nBits=nbits)
            fpdict['fcfp4'] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=nbits)
            fpdict['fcfp6'] = AllChem.GetMorganFingerprintAsBitVect(mol, 3, useFeatures=True, nBits=nbits)
            fpdict['fcfc2'] = AllChem.GetMorganFingerprint(mol, 1, useFeatures=True)
            fpdict['fcfc4'] = AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)
            fpdict['fcfc6'] = AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)
            fpdict['lecfp4'] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=longbits)
            fpdict['lecfp6'] = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=longbits)
            fpdict['lfcfp4'] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=longbits)
            fpdict['lfcfp6'] = AllChem.GetMorganFingerprintAsBitVect(mol, 3, useFeatures=True, nBits=longbits)
            fpdict['maccs'] = MACCSkeys.GenMACCSKeys(mol)
            fpdict['ap'] = Pairs.GetAtomPairFingerprint(mol)
            fpdict['tt'] = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
            fpdict['hashap'] = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits)
            fpdict['hashap4'] = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits, maxLength=4)
            fpdict['hashtt'] = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nbits)
            fpdict['avalon'] = fpAvalon.GetAvalonFP(mol, nbits)
            fpdict['laval'] = fpAvalon.GetAvalonFP(mol, longbits)
            fpdict['rdk5'] = Chem.RDKFingerprint(mol, maxPath=5, fpSize=nbits, nBitsPerHash=2)
            fpdict['rdk6'] = Chem.RDKFingerprint(mol, maxPath=6, fpSize=nbits, nBitsPerHash=2)
            fpdict['rdk7'] = Chem.RDKFingerprint(mol, maxPath=7, fpSize=nbits, nBitsPerHash=2)
            ##fpdict['m2ap'] = getHybridFP(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits), rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits))
            ##calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            ##fpdict['rdkDes'] = calc.CalcDescriptors(mol)              
            #print(fpdict)
            #print(cordict)
            cordict[str(name)] = fpdict
            #print(cordict)
            n+=1 
     return cordict

def fpssimilar(reffps, dbfps, outpath="./"):
     #storeresult = []
     n = len(dbfps.keys())
     m = (len(list(reffps.values())[0])-2)*3 + 2
     shape1 = (n, 0)
     shape2 = (0, m)
     tmp13 = np.empty(shape1)
     tmp15 = np.empty(shape2)
     for REF in reffps.keys():
         # ��¼��ͷ
         tmp1 = []
         # ��ȡ�ο��ֵ�
         dictset1 = reffps.get(REF)
         ##print(dictset1)
         for key1 in dictset1.keys():
             if key1 == 'Name':
                tmp1.append("Refer_ID")
             elif key1 == 'Smiles':
                tmp1.append("Refer_smile")
             else:
                ##print(key1)
                tmp1.append("%s_ID" % str(key1)) 
                tmp1.append("%s_smile" % str(key1))               
                tmp1.append("%s_similarity" % str(key1))
         #print(storehead)
         # �ݴ�ÿ���ο�����
         tmp2 = [] 
         tmp12 = np.empty(shape1)
         for key2 in dictset1.keys():
             tmp3 = []
             if key2 == 'Name':
                tmp2.append(dictset1[key2])
             elif key2 == 'Smiles':
                tmp2.append(dictset1[key2])
             else:
                #print(tmp2)
                tmp3 = []
                for TARGET in dbfps:
                    # ��ȡ�����ֵ�
                    dictset2 = dbfps.get(TARGET)
                    # �ݴ�ÿ����������
                    tmp4 = []
                    for key3 in dictset2.keys():
                        if key3 == 'Name':
                           tmp4.append(dictset2[key3])
                           continue
                        elif key3 == 'Smiles':
                           tmp4.append(dictset2[key3])
                        # ÿ����������������ָ�Ʊ���
                        elif key3 == key2:
                           tmp5 = []
                           if key3 in ["ecfc0", "ecfc2", "ecfc4", "ecfc6", "fcfc2","fcfc4", "fcfc6", 'ap','tt']:
                               ## ����Dice�����Է���
                               simscore = DataStructs.DiceSimilarity(dictset1[key2],dictset2[key3])
                               #print(simscore)
                               tmp5 = tmp4 + [simscore]
                               tmp3.append(tmp5)
                           elif key3 in ['ecfp0', 'ecfp2', 'ecfp4', 'ecfp6', 'fcfp2', 'fcfp4', 'fcfp6','lecfp4','lecfp6','lfcfp4','lfcfp6','maccs','hashap','hashap4','hashtt','avalon','laval','rdk5','rdk6','rdk7','m2ap','rdkDes']:
                               ## ����Tanimoto�����Է���
                               simscore = DataStructs.FingerprintSimilarity(dictset1[key2],dictset2[key3])
                               #print(simscore)
                               tmp5 = tmp4 + [simscore]
                               tmp3.append(tmp5)
                tmp11 = sorted(tmp3, key=lambda x: -x[2])
                tmp12 = np.hstack((tmp12, np.array(tmp11)))
         tmp13 = np.expand_dims(tmp2,0).repeat(n, axis=0)
         tmp14 = np.hstack((tmp13,tmp12))
         tmp15 = np.vstack((tmp15, tmp14))
         #print(tmp15.tolist())
         tmp16 = np.vstack((tmp1, tmp14))
         #print(tmp16.tolist())
         np.savetxt(os.path.join(outpath,"%s_sim.csv")% REF, tmp16, delimiter=',', fmt='%s')
     tmp17 = np.vstack((tmp1, tmp15))
     np.savetxt(os.path.join(outpath,"total_sim.csv"),tmp17,delimiter=',', fmt='%s')
     storehead = tmp1
     storeresult = tmp15  
     return storehead, storeresult

def filterfps(pdresult,cutoff=0.5, outpath="./"):
    reflist = pdresult['Refer_ID'].unique()
    # ��������ɸѡ�����Target��Ϊ�ֵ䣬����Ϊ�ο�ID
    totalresult = pd.DataFrame()
    tmp = {}    
    for REF in reflist:
        # ��¼��ͷ��Ϣ
        headresult = pdresult.columns
        #print(headresult)
        # ��ȡÿ���ο���ָ����������
        #print(REF)
        fatherpd = pdresult[pdresult['Refer_ID'] == REF]
        #print(fatherpd)
        refid = fatherpd['Refer_ID'].unique()
        smileid = fatherpd['Refer_smile'].unique()
        del fatherpd["Refer_ID"]
        del fatherpd["Refer_smile"]
        filterresult = pd.DataFrame()
        total_num = len(fatherpd.columns)
        subset_num = math.floor((total_num/3))
        tmp2 = []
        for index in range(subset_num):
            #print(index)
            sonpd = fatherpd.iloc[:,3*index:(3*index+3)]
            #sonpd = fatherpd.iloc[:,0:3]
            #print(sonpd)
            # ɸѡ����cutoff�ļ�¼
            filterpd = sonpd[sonpd.iloc[:,2].astype(float) > float(cutoff)]
            #print(filterpd)
            filterresult = pd.concat([filterresult,filterpd],axis=1,ignore_index=False)
            #tmp2.append(filterpd.iloc[:,0].tolist())
            # ������������Target����list
            tmp2 = tmp2 + filterpd.iloc[:,0].tolist()
            #print(list(set(tmp2)))
        tmp[REF] = list(set(tmp2))
        #print(len(filterresult))
        #print(refid)
        #print(smileid) 
        # ���ο���������ͬά��չ--> refresult
        tmp3 = {"Refer_ID":refid.repeat(len(filterresult)),"Refer_smile":smileid.repeat(len(filterresult))}
        #print(refid.repeat(len(filterresult)))
        refresult = pd.DataFrame(tmp3)
        #print(refresult)
        # ��ɸѡ�����ݼ���ο�����
        filterresult = pd.concat([refresult,filterresult.reset_index(drop=True)],axis=1,ignore_index=False)
        #print(filterresult)
        # ��ɸѡ�����һ����
        filterresult.to_csv(os.path.join(outpath,"%s_filter_with_cutoff_%s.csv")% (REF,cutoff), sep=',', index=False,encoding='utf-8')
        # ������ۻ����ݸ�totalresult
        totalresult = pd.concat([totalresult,filterresult.reset_index(drop=True)],axis=0,)
    #print(tmp)
    filtertargetdic = tmp
    return filtertargetdic, totalresult

def topfps(pdresult, topset=10, outpath="./"):
    reflist = pdresult['Refer_ID'].unique()
    # ��������ɸѡ�����Target��Ϊ�ֵ䣬����Ϊ�ο�ID
    totalresult = pd.DataFrame()
    tmp1 = {} 
    for REF in reflist: 
        fatherpd = pdresult[pdresult['Refer_ID'] == REF]
        if int(topset) < len(fatherpd):
           filterpd = fatherpd[0:int(topset)]
        else:
           filterpd = fatherpd
        #print(filterpd)
        tmp2 =[]
        total_num = len(filterpd.columns)-2
        subset_num = math.floor((total_num/3))
        for index in range(subset_num):
            sonpd = filterpd.iloc[:,(3*index+2):(3*index+5)]
            #print(sonpd)
            tmp2 = tmp2 + sonpd.iloc[:,0].tolist()  
        tmp1[REF] = list(set(tmp2))
        filterpd.to_csv(os.path.join(outpath,"%s_filter_with_top_%s.csv")% (REF,topset), sep=',', index=False,encoding='utf-8')
        # ������ۻ����ݸ�totalresult
        totalresult = pd.concat([totalresult,filterpd.reset_index(drop=True)],axis=0)
    filtertargetdic = tmp1
    #print(filtertargetdic)
    return filtertargetdic,totalresult

def savepng(dicresult, refpd, dbpd, n=10, outpath='./'):
     for REF in dicresult.keys():
         #print(REF)          
         refsmi = refpd[refpd['ID'].str.contains(REF)]['smiles'].reset_index(drop=True)[0]
         ref_mol = Chem.MolFromSmiles(refsmi)
         ref_ECFP4_fps = AllChem.GetMorganFingerprintAsBitVect(ref_mol,2)

         dblist = dicresult.get(REF)
         #print(dblist)
         if int(n) < len(dblist):
            TARGET = dblist[:n]
            #print(TARGET)
         else:
            TARGET = dblist
            #print(TARGET)  

         dbsmipd = dbpd[dbpd['ID'].isin(TARGET)].reset_index(drop=True)
         #print(dbsmipd)
         # ��Ŀ��ҩ���dataframe����ṹ��ϢROMOl
         PandasTools.AddMoleculeColumnToFrame(dbsmipd, smilesCol='smiles')
         #print(dbsmipd)
         # ���ɷ���ָ��
         bulk_ECFP4_fps = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in dbsmipd['ROMol']]
         # ����������
         similarity_efcp4 = [DataStructs.FingerprintSimilarity(ref_ECFP4_fps,x) for x in bulk_ECFP4_fps]
         dbsmipd['Tanimoto_Similarity (ECFP4)'] = similarity_efcp4
         # ���ݷ���ָ������
         dbsmipd = dbsmipd.sort_values(['Tanimoto_Similarity (ECFP4)'], ascending=False)
         # ����2D�ṹͼ
         img = PandasTools.FrameToGridImage(dbsmipd.head(n), legendsCol="Tanimoto_Similarity (ECFP4)", subImgSize=(200,200),molsPerRow=4)
         img.save(os.path.join(outpath, "%s_top_%d_molecules.png") % (REF,n))

def pd2smi(dicresult, dbpd, outpath='./'):
     for REF in dicresult.keys():
         dblist = dicresult.get(REF)
         dbsmipd = dbpd[dbpd['ID'].isin(dblist)].reset_index(drop=True)
         dbsmipd.to_csv(os.path.join(outpath,"%s_filter_fps.smi")% (REF), sep='\t', index=False, header=False, encoding='utf-8')

def savepng2(pdresult, n=10, method ='ecfp4',outpath='./'):
     reflist = pdresult['Refer_ID'].unique()
     #print(reflist)
     for REF in reflist:
         fatherpd = pdresult[pdresult['Refer_ID'] == REF].reset_index(drop=True)
         if int(n) < len(fatherpd):
            tompd = fatherpd[:int(n)]
            #tompd = fatherpd.head(int(n))
            #print(tompd)
         else:
            tompd = fatherpd
            #print(tompd) 

         id = '%s_ID' % method
         smile = '%s_smile' % method
         similary = '%s_similarity' % method
         filterlist = [id, similary, smile]
         #print(filterlist)
         sonpd = tompd[filterlist]
         # �ӱ���������column����smile�ŵ��������PandasTools.AddMoleculeColumnToFrame����
         sonpd = sonpd.reindex(columns = filterlist)
         sonpd = sonpd.dropna()
         #print(REF)
         #print(sonpd)
         ## ��Ŀ��ҩ���dataframe����ṹ��ϢROMOl
         if sonpd.empty:
            print("The filter result without suitable for figure build.")
         else:     
            PandasTools.AddMoleculeColumnToFrame(sonpd, smilesCol=smile)
            #print(sonpd)
            ## ����2D�ṹͼ
            img = PandasTools.FrameToGridImage(sonpd.head(n), legendsCol=similary, subImgSize=(200,200),molsPerRow=4)
            img.save(os.path.join(outpath, "%s_top_%d_molecules.png") % (REF, n))

def np2csv(resultnp,resulthead,output='fps.similar.csv', outpath="./"):
    tmp = np.vstack((resulthead,resultnp))
    np.savetxt(os.path.join(outpath,output),tmp,delimiter=',', fmt='%s')    

if __name__ == '__main__':

	parser = argparse.ArgumentParser(prog='easy_novel.py', description='Finding novel compound from CompondDB with fingerprint similar search using RDkit.')
	parser.add_argument('-ref', '--inputreffile', dest='REF', help='input sdf/smi file path for refercence compound')
	parser.add_argument('-db', '--inputdbfile', dest='DB', help='input sdf/smi file path for database compound')
	parser.add_argument('-out', '--outputpath', dest='OUT', default="./", help='file to write results')
	#parser.add_argument('-r', '--radius', dest='RADIUS', default=2, help='option for fps')
	parser.add_argument('-b', '--nBits', dest='NBITS', default=1024, help='option for fps')
	parser.add_argument('-lb', '--longbits', dest='LONGBITS', default=16384, help='option for fps')
	parser.add_argument('-k','--cutoff', dest='CUTOFF', default=0.5, help='set cut off value for filter')
	parser.add_argument('-t','--topset', dest='TOPSET', default=10, help='set top range for filter')
	parser.add_argument('-f','--flag', dest='FLAG', default="ID", help='set id name for each compound')
	parser.add_argument('-v', '--verbose', default=False, action="store_true", help='verbose output')
	args = parser.parse_args()

	if args.REF and args.DB:
		if '.smi' in args.REF:
			dataset1 = smi2list(args.REF)
			#print(dataset1)
		elif '.sdf' in args.REF:
			dataset1 = sdf2smi(args.REF, args.FLAG)
			#print(dataset1)
		else:
			logging.info('The input reference files was set with error filetype. Pleasa correct and resubmit again.')

		if '.smi' in args.DB:
			dataset2 = smi2list(args.DB)
		elif '.sdf' in args.DB:
			dataset2 = sdf2smi(args.DB, args.FLAG)
		else:
			logging.info('The input database files was set with error filetype. Pleasa correct and resubmit again.')

		theTime = datetime.datetime.now()
		print("The job was begining! %s" % theTime) 

		reffptdic = smi2fps_dict(dataset1, nbits=args.NBITS, longbits=args.LONGBITS)
		#print(reffptdic)
		dbfptdic = smi2fps_dict(dataset2, nbits=args.NBITS, longbits=args.LONGBITS)
		#print(dbfptdic)
		headcontent, dataset3 = fpssimilar(reffptdic, dbfptdic)
		#print(headcontent)
		#print(dataset3.tolist())
		#print(dataset3)
		##pd2csv(dblist=dataset3,head=headcontent)
		dataset4 = pd.DataFrame(dataset3,columns=headcontent)
		#print(dataset4)

		theTime = datetime.datetime.now()
		print('The calculation for finger was begining! %s' % theTime)

		if args.CUTOFF: 
			dataset5,dataset6 = filterfps(dataset4, cutoff=args.CUTOFF)
			try:
				os.mkdir("cutoff_filter")
			except OSError as error:
				print("The cutoff_filter path had been built!")
			#savepng(dataset5, dataset1, dataset2, n=10, outpath='./cutoff_filter')
			pd2smi(dataset5, dataset2, outpath='./cutoff_filter')
			savepng2(dataset6, n=10, method ='ecfp0', outpath='./cutoff_filter')
		else:
			logging.info('The input option was lost with cutoff or topset values. Pleasa correct and resubmit again.')

		if args.TOPSET:
			dataset5,dataset6 = topfps(dataset4, topset=args.TOPSET)
			try:
				os.mkdir("top_filter")
			except OSError as error:
				print("The top_filer path had been built!")
			#savepng(dataset5, dataset1, dataset2, n=10, outpath='./top_filer')
			pd2smi(dataset5, dataset2, outpath='./top_filter')
			savepng2(dataset6, n=10, method ='ecfp0', outpath='./top_filter')
		else:
			logging.info('The input option was lost with cutoff or topset values. Pleasa correct and resubmit again.')

		theTime = datetime.datetime.now()
		print('Everything is ok! %s' % theTime)

	else:
		logging.info('The input files including reffile and dbfile was blank. Pleasa correct and resubmit again.')