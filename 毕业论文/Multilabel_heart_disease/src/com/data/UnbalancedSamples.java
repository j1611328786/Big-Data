package com.data;

import java.util.HashMap;
import java.util.Iterator;

import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;
import weka.core.Instance;
import weka.core.Instances;

public class UnbalancedSamples {

	
	/*简单欠采样数据集*/
	public static MultiLabelInstances Undersampling(MultiLabelInstances dataset,LabelSet under_labels,double threshold){
		MultiLabelInstances new_dataset=dataset.clone();
		
		Statistics stat=new Statistics();
		stat.calculateStats(new_dataset);
		HashMap<LabelSet, Integer> multi_fre=stat.labelCombCount();

		System.out.println(multi_fre.size());
		Integer count=multi_fre.get(under_labels);//
		System.out.println(count);
		
		int sum_dataset=new_dataset.getNumInstances();
		int[] label_position=new_dataset.getLabelIndices();//
		
		double proportion=count*1.0/sum_dataset;
	    Instances data=new_dataset.getDataSet();
	    
	   
       
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
        	
        return new_dataset;
        
	}
			

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
	}

}
