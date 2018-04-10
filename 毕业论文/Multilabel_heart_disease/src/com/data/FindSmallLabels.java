package com.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;
import weka.core.Instance;
import weka.core.Instances;

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
	private MultiLabelInstances dataset;//数据集
	private ArrayList<Integer> smalllabels=new ArrayList<Integer>();
	private ArrayList<Integer> maxlabels=new ArrayList<Integer>();
	private Instances smallSample=null;//小类标签对应的样本集
	private Instances maxSample=null;//大类标签对应的样本集
	private int[] labelsFrequency;//标签出现的频度
	private double[] imR;//每个标签在多标签内的不均衡度
	private double[] irlb;//每个标签在多标签间的不均衡度
	private double avgimR;//标签内的多标签不均衡性度量
	private double meanIR;//标签间的多标签不均衡性度量
	
	public FindSmallLabels(MultiLabelInstances dataset) {
		this.dataset=dataset;
	}
	
	public void inner_labels() {
		int number=dataset.getNumInstances();
		System.out.println("样本数,标签数  "+number+","+dataset.getNumLabels());
		String[] labelnames=dataset.getLabelNames();
		int[] labelindices=dataset.getLabelIndices();
		Statistics stat=new Statistics();
		stat.calculateStats(dataset);
		double[] multi_fre=stat.priors();//样本集中每个单标签占得比例
        int len=labelnames.length;
		
		imR=new double[len];
		avgimR=0;
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
		
		//小类别标签在样本中的位置，即选出的小类别标签集
		for(int i=0;i<len;i++) {
			if(imR[i]>avgimR)
				smalllabels.add(labelindices[i]);
			if(imR[i]<avgimR)
				maxlabels.add(labelindices[i]);
		}
		
	}

	public void  between_labels() {
		int number=dataset.getNumInstances();
		String[] labelnames=dataset.getLabelNames();
		int[] labelindices=dataset.getLabelIndices();
		/*System.out.println("------------labelindices--------------");
		for(int i=0;i<labelindices.length;i++)
			System.out.print(labelindices[i]+"  ");
		*/
		Statistics stat=new Statistics();
		stat.calculateStats(dataset);
		double[] multi_fre=stat.priors();//样本集中每个单标签占得比例
        int len=labelnames.length;
		
        labelsFrequency=new int[len];
		irlb=new double[len];
		labelsFrequency[0]=(int)(multi_fre[0]*number);
		int max=labelsFrequency[0];
		for(int i=1;i<len;i++) {
			labelsFrequency[i]=(int)(multi_fre[i]*number);//标签i出现的样本数
			if(max<labelsFrequency[i])
				max=labelsFrequency[i];
		}
		for(int i=0;i<len;i++) 
		{
			irlb[i]=max*1.0/labelsFrequency[i]; //IRLBlj=（出现在样本集次数最多的标签）/（标签j出现的次数）,衡量标签间出现次数的差异性
			//System.out.println(labelnames[i]+" IRBL : "+irlb[i]);
		}
		
		meanIR=0;
		for(int i=0;i<len;i++)
			meanIR+=irlb[i];
		meanIR=meanIR/len;//平均，meanIR
		//System.out.println("meanIR: "+meanIR);
		
		//小类别标签在样本中的位置，即选出的小类别标签集
		for(int i=0;i<len;i++) {
			if(irlb[i]>meanIR) 
				smalllabels.add(labelindices[i]);
			if(irlb[i]<meanIR)
				maxlabels.add(labelindices[i]);
		}
	}
	
	/*
	 * 计算样本集中指定标签的不均衡性值
	 */
	public static double callabelIMR(MultiLabelInstances instances,int index) {
		int number=instances.getNumInstances();
		Statistics stat=new Statistics();
		stat.calculateStats(instances);
		double[] multi_fre=stat.priors();//样本集中每个单标签占得比例
        int len=multi_fre.length;
		
		double imr=0;
		int dj=(int)(multi_fre[index]*number);//标签i出现的样本数
		int d_j=number-dj;//标签i未出现的次数
		if(dj>d_j) 
		{
			int tmp=dj;
			dj=d_j;
			d_j=tmp;
		}
		return d_j*1.0/dj;
	}
	
	public static double callabelIRLB(MultiLabelInstances instances,int index) {
		int number=instances.getNumInstances();
		Statistics stat=new Statistics();
		stat.calculateStats(instances);
		double[] multi_fre=stat.priors();//样本集中每个单标签占得比例
        int len=multi_fre.length;
        int lenf=instances.getFeatureIndices().length;
	
		double max=multi_fre[0]*number;
		for(int i=1;i<len;i++) {
			double irlb=multi_fre[i]*number;//标签i出现的样本数
			if(max<irlb)
				max=irlb;
		}
		return max/(multi_fre[index-lenf]*number); //IRLBlj=（出现在样本集次数最多的标签）/（标签j出现的次数）,衡量标签间出现次数的差异性
	}

    public double[] getIMR() {
    	return imR;
    }

    public double getAVGIMR() {
    	return avgimR;
    }
    
    public double[] getIRLB() {
    	return irlb;
    }
    
    public double getMEANIR() {
    	return meanIR;
    }
    
    public ArrayList<Integer> getsmalllabels(){
    	System.out.print("\n选出的小类标签集： ");
    	for(int i=0;i<smalllabels.size();i++)
    		System.out.print(smalllabels.get(i)+"  ");
    	System.out.println();
    	return smalllabels;
    }

    public ArrayList<Integer> getmaxlabels(){
    	System.out.print("\n选出的大类标签集： ");
    	for(int i=0;i<maxlabels.size();i++)
    		System.out.print(maxlabels.get(i)+"  ");
    	System.out.println();
    	return maxlabels;
    }
    
    public int[] getFrequency() {
    	return labelsFrequency;
    }
    /*
     * 满足IRL(y)=meanIR,对应的样本集大小，
     * 
     * meanIR=max/meanInstances;
     * 
     */
    public int getMeanInstances() {
    	int meanInstances=labelsFrequency[0];
    	for(int i=1;i<labelsFrequency.length;i++)
    		if(meanInstances<labelsFrequency[i])
    			meanInstances=labelsFrequency[i];
        return (int) (meanInstances/meanIR);
    }
    
	// 取出出现小类标签所对应的样本集
	private void splitSample() {
		Instances sample=dataset.clone().getDataSet();
		
		if(maxSample==null) {
			maxSample=new Instances(sample);
			maxSample.clear();
		}
		    
		
		Iterator<Instance> it = sample.iterator();
		while (it.hasNext()) {
			Instance instance = it.next();
			boolean flag=false;
			for(Integer label:smalllabels) {
				int v=(int) instance.value(label);
				if (v== 1) {
					flag=true;
					break;
				}
			}
			if(!flag) {
				maxSample.add(instance);
				it.remove();
			}
		}
		smallSample=sample;
	}
	
	public Instances getSmallSample() {
		if(smallSample==null)
		   splitSample();
		return smallSample;
	}
	
	public Instances getMaxSample() {
		if(maxSample==null)
			splitSample();
		return maxSample;
	}
        
	public static void printDistribution(MultiLabelInstances dataset) {
		System.out.println("----------------------标签集的均衡性分布--------------------------------------");
		FindSmallLabels fs=new FindSmallLabels(dataset);
		fs.between_labels();
		ArrayList<Integer>  smalllabels=fs.getsmalllabels();
		ArrayList<Integer>  maxlabels=fs.getmaxlabels();
		double[] IRBL=fs.getIRLB();
		double meanIR=fs.getMEANIR();
		int avginstances=fs.getMeanInstances();
		
	
		for(int i=0;i<IRBL.length;i++)
			System.out.println("IRBL : "+IRBL[i]);
		System.out.println("meanIR: "+meanIR);
		System.out.println(smalllabels);
		System.out.println(maxlabels);
		System.out.println("平均样本数： "+avginstances);
	}
	
	public void printDistribution(){
		printDistribution(this.dataset);
	}
	
}
