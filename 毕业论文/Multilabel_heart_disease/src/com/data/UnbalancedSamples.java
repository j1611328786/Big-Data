package com.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;
import weka.core.Instance;
import weka.core.Instances;

public class UnbalancedSamples {

	
	/*简单欠采样数据集*/
	public static void Undersampling(MultiLabelInstances dataset,LabelSet under_labels,double threshold){
		Statistics stat=new Statistics();
		stat.calculateStats(dataset);
		HashMap<LabelSet, Integer> multi_fre=stat.labelCombCount();

		System.out.println(multi_fre.size());
		Integer count=multi_fre.get(under_labels);//
		System.out.println(count);
		
		int sum_dataset=dataset.getNumInstances();
		int[] label_position=dataset.getLabelIndices();//
		
		double proportion=count*1.0/sum_dataset;
	    Instances data=dataset.getDataSet();
	    
	   
       
	    Iterator<Instance> it=data.iterator();
        while(it.hasNext()&&(proportion>threshold))
        {
        		Instance in=it.next();
				double[] labels=new double[label_position.length];
				for(int j=0;j<label_position.length;j++){
					labels[j]=in.value(label_position[j]);
				}
				LabelSet labelset=new LabelSet(labels);
				if(labelset.hammingDifference(under_labels)==0)
				{	
					it.remove();
					--count;
					--sum_dataset;
				    
				}
				proportion=count*1.0/sum_dataset;
        }
        
	}
	
	
	/*
	 * 
	 * ML-RUS
	 * while 采样比例大于 percentage
	 *     if 样本i 中不包含所有小类标签（smalllabels）
	 *        删除该样本
	 *        重新计算 非小类标签集maxlabels中 标签的不均衡度，且将大于 原始meanIR的标签 删除 并归入 小类标签集smalllabels
	 * 
	 * */
	public static void calML_RUS(MultiLabelInstances dataset,double percentage) throws Exception {
		FindSmallLabels fs=new FindSmallLabels(dataset);
		System.out.println("--------------IMR--------------- ");
		fs.between_labels();
		//fs.inner_labels();
		ArrayList<Integer>  smalllabels=fs.getsmalllabels();
		ArrayList<Integer>  maxlabels=fs.getmaxlabels();
		double meanIR=fs.getMEANIR();
		Instances datasetInstances=dataset.getDataSet();
		Iterator<Instance> it=datasetInstances.iterator();
		int remove_number=(int)(dataset.getNumInstances()*(1-percentage));
		int[] labelsFrequency=fs.getFrequency();//每个标签出现的次数
		int numFeatures=dataset.getFeatureIndices().length;
		
		while(remove_number>0 && it.hasNext()) {
			Instance instance=it.next();
			boolean flag=false;
			for(int i=0;i<smalllabels.size();i++) {
				if(instance.value(smalllabels.get(i))==1)
				{
					flag=true;
					break;
				}
			}
			if(!flag) {//仅删除所有未出现小类标签的实例
				
				//对即将要删除的样本中 出现的标签 次数递减	
				 Iterator<Integer> it1=maxlabels.iterator();
				 while(it1.hasNext()) 
				 {
					   int value=it1.next();
					   if(instance.value(value)==1)
						   labelsFrequency[value-numFeatures]--;
				  }
			   it.remove();//删除该样本
			   
			   int max=labelsFrequency[0];
			   for(int i=1;i<labelsFrequency.length;i++)
				   if(max<labelsFrequency[i])
					   max=labelsFrequency[i];
			   
			   Iterator<Integer> it2=maxlabels.iterator();
			   while(it2.hasNext()) 
			   {
				   int value=it2.next();
			       if((max*1.0/labelsFrequency[value-numFeatures])>=meanIR)
			       {
				      it2.remove();
				      System.out.println("--------------------");
				      smalllabels.add(value);
			       }
			   }
			   remove_number--;
			}
		}
		Iterator<Integer> its=smalllabels.iterator();
		while(its.hasNext()) 
		{
		  System.out.println(its.next());	
		}
	}
}
