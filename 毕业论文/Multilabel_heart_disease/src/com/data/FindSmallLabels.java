package com.data;

import java.util.ArrayList;
import java.util.HashMap;

import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;

/*
 * 寻找小类别标签
 * 评估小类别标签的方法分为两种：
 *  一、标签内的评估
 *    每个标签j 在样本中出现的次数d1与未出现的次数d2的比值-max(d1,d2)/min(d1,d2) 记作 imR(j)
 *     imR(j)求和取平均，记作avgimR, avgimR衡量标签内的数据不均衡性。若imR(j)>avgimR,为小类别标签
 *    
 *  二、标签间的评估
 *     IRLBlj=（出现在样本集次数最多的标签）/（标签j出现的次数）,衡量标签间出现次数的差异性。meanIR 所有标签IRLBlj的平均值
 */
public class FindSmallLabels {
	public static ArrayList<Integer> inner_labels(MultiLabelInstances dataset) {
		int number=dataset.getNumInstances();
		System.out.println("样本数,标签数  "+number+","+dataset.getNumLabels());
		String[] labelnames=dataset.getLabelNames();
		int[] labelindices=dataset.getLabelIndices();
		Statistics stat=new Statistics();
		stat.calculateStats(dataset);
		double[] multi_fre=stat.priors();//样本集中每个单标签占得比例
        int len=labelnames.length;
		
		double[] imR=new double[len];
		double avgimR=0;
		for(int i=0;i<len;i++) {
			int dj=(int)(multi_fre[i]*number);//标签i出现的样本数
			int d_j=number-dj;//标签i未出现的次数
			if(dj>d_j) {
				int tmp=dj;
				dj=d_j;
				d_j=tmp;
			}
			imR[i]=d_j*1.0/dj;
			System.out.println(labelnames[i]+" imr : "+imR[i]);
		}
		for(int i=0;i<len;i++)
			avgimR+=imR[i];
		avgimR=avgimR/len;
		System.out.println(avgimR);
		
		ArrayList<Integer> smalllabels=new ArrayList<Integer>();//小类别标签在样本中的位置，即选出的小类别标签集
		for(int i=0;i<len;i++) {
			if(imR[i]>avgimR)
				smalllabels.add(labelindices[i]);
		}
		return smalllabels;
		
	}

	public static ArrayList<Integer> between_labels(MultiLabelInstances dataset) {
		return null;
	}

}
