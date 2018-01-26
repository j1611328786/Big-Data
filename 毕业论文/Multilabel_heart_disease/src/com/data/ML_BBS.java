package com.data;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

public class ML_BBS {
   /*
    *双重均衡采样机制 
    * 	
    */
	public static void dobbs(MultiLabelInstances dataset) {
		FindSmallLabels fsl = new FindSmallLabels(dataset);
		fsl.between_labels();
		double meanIR = fsl.getMEANIR();// 标签间的不均衡度
		double[] IRBL=fsl.getIRLB();//每个标签的不均衡性
		List<Integer> smalllabels = fsl.getsmalllabels();// 小标签集合
		List<Integer> maxlabels=fsl.getmaxlabels();//大标签集合
		int meanInstances=fsl.getMeanInstances();//平均样本数
		int[] labeldices=dataset.getLabelIndices();
		int undersample=0,oversample=0;//欠采样、过采样的数量
		int[] labelsFrequency=fsl.getFrequency();//每个标签出现的次数
		for(int i=0;i<IRBL.length;i++)
			System.out.println(" IRBL : "+IRBL[i]);
		System.out.println("meanIR: "+meanIR);
		System.out.println(smalllabels);
		System.out.println(maxlabels);
		for(int j=0;j<labelsFrequency.length;j++)
			System.out.println(labelsFrequency[j]);
		
		
		
		//取出小类标签的样本集
		int numFeatures=dataset.getFeatureIndices().length;
		Instances[] smallsample=new Instances[smalllabels.size()];
		Map[]  vdmMap=new Map[smalllabels.size()];
		MLSMOTE ms=new MLSMOTE(dataset);
		for(int i=0;i<smallsample.length;i++) {
			smallsample[i]=ms.getLabel(smalllabels.get(i));
			//System.out.println(smallsample[i].size());
		}
		for(int i=0;i<smallsample.length;i++)
			vdmMap[i]=ms.getVdmp(smallsample[i]);
		
		Instances instances=dataset.getDataSet();
		int len=instances.size();
		int i=0;
		
		/*
		 *设置threshold目的是当所有标签的不均衡度接近于meanIR的时候认为此时数据集达到了平衡状态， 也就是说threshold越小越好逐渐趋于稳定，
		 *当threshold从递减状态变为递增时，采样结束
		 */
		double threshold=getThreshold(IRBL,meanIR);
		double old_threshold;
		
		while((!smalllabels.isEmpty()||!maxlabels.isEmpty())&&i<len) {
			Instance instance=instances.get(i);
			boolean flag=false;
			for(int j=0;j<smalllabels.size();j++) {
				if(instance.value(smalllabels.get(j))==1)
				{
					flag=true;
					//MLSMOTE过采样
					Instance[] nndArray=ms.getKNN(smallsample[j], vdmMap[j], instance);
					instances.add(ms.getSyntheticExample(instance, nndArray));
					oversample++;
					
				}
			}
			 if(!flag) {
				 /*
				  * 仅删除所有未出现小类标签的实例,欠采样*  
				  * *考虑到大样本标签集中的不均衡度，最大样本标签集和最小样本标签集（即将接近meanIRR）的差距，
				  *  
				  */
				 for(int j=0;j<labeldices.length;j++) {
						if(!maxlabels.contains(labeldices[j])&&instance.value(labeldices[j])==1)
						{
							flag=true;
							break;
						}
					}
				 if(!flag)
				 {
					 instances.delete(i);//删除该样本
					 i--;
				     undersample++;
				  }
			}
			 i++;
			 if((i+1)%10==0) {
				 FindSmallLabels fs = new FindSmallLabels(dataset);
			     fs.between_labels();
			     IRBL=fs.getIRLB();
			     meanIR=fs.getMEANIR();
			     labelsFrequency=fs.getFrequency();
			     old_threshold=threshold;
			     threshold=getThreshold(IRBL,meanIR);
			     //System.out.println("threshold: "+threshold+" meanIR: "+meanIR);
		    	 
			    
			     for(int j=0;j<maxlabels.size();j++)
			    	 if(labelsFrequency[maxlabels.get(j)-numFeatures]<=meanInstances) {
			    		 //该标签的样本集特点是不均衡度接近meanIR,当大类标签变成小类标签时,对该标签的采样过程结束
			    		 //重新划分大小类标签集（为了防止大类标签进行过采样，将其从大类标签集中删掉）
			    		 //为了加大容错性，为meanIR设置阈值
			    		 maxlabels.remove(j); 
			    	 }
			     
			       //当小样本集标签非空，说明过采样未结束，并将大于meanInstances的标签从小样本集标签中删除
			       for(int j=0;j<smalllabels.size();j++) {
				    	 if(labelsFrequency[smalllabels.get(j)-numFeatures]>=meanInstances)
				    		 smalllabels.remove(j);
				     }
			     
			     
			     for(int j=0;j<maxlabels.size();j++)
			    	 if(IRBL[maxlabels.get(j)-numFeatures]>=meanIR-5) {
			    		 //该标签的样本集特点是不均衡度接近meanIR,当大类标签变成小类标签时,对该标签的采样过程结束
			    		 //重新划分大小类标签集（为了防止大类标签进行过采样，将其从大类标签集中删掉）
			    		 //为了加大容错性，为meanIR设置阈值
			    		 maxlabels.remove(j); 
			    	 }
			     
			     
			    	 //当小样本集标签非空，说明过采样未结束，并将大于meanInstances的标签从小样本集标签中删除
			    	 for(int j=0;j<smalllabels.size();j++) {
				    	 if(IRBL[smalllabels.get(j)-numFeatures]<=meanIR+5)
				    		 smalllabels.remove(j);
				     }
			     
			     
			     
			     //System.out.println("threshold: "+threshold);
			     
		      }
		}
		for(int j=0;j<IRBL.length;j++)
			System.out.println(" IRBL : "+IRBL[j]);
		System.out.println(meanIR);
		System.out.println("oversample: "+oversample+"  undersample: "+undersample);
		System.out.println("----------------labelFrequency--------------------");
		for(int j=0;j<labelsFrequency.length;j++)
			System.out.println(labelsFrequency[j]);
		System.out.println("--------------------smalllabels----------------------");
		for(int v:smalllabels)
			System.out.println(v);
		System.out.println("---------------------maxlabels------------------------");
		for(int v:maxlabels)
			System.out.println(v);
		
	}

	/*
	 * 
	 * 设置采样结束的阈值，为了防止采样过度导致小类标签变成大类标签，大类标签变成小类标签
	 * 
	 */
private static double getThreshold(double[] iRBL, double meanIR) {
	// TODO Auto-generated method stub
	double threshold=0;
	for(int i=0;i<iRBL.length;i++) {
		threshold+=(iRBL[i]-meanIR)*(iRBL[i]-meanIR);
	}
	return Math.sqrt(threshold)/iRBL.length;
}

}
