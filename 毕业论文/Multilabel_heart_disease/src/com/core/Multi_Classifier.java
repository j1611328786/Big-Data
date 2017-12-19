package com.core;


import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.classifiers.functions.Logistic;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LibSVM;


import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import com.data.*;

import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelOutput;
import mulan.transformations.BinaryRelevanceTransformation;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import mulan.classifier.transformation.AdaBoostMH;
import mulan.data.Statistics;

public class Multi_Classifier {
	
	private static MultiLabelInstances dataset;
	private int numlabels;
	private Instances test;
	private String xmlFilename;
	private BinaryRelevance br;
	private AdaBoostMH adb;
	
	public Multi_Classifier(String[] arg){
		try{ 
			String arffFilename = Utils.getOption("arff", arg);
			xmlFilename = Utils.getOption("xml", arg);	
			dataset = new MultiLabelInstances(arffFilename, xmlFilename);
			numlabels=dataset.getNumLabels();
			System.out.println(arg.length);
			if(arg.length==6){
				String testFilename = Utils.getOption("unlabeled", arg);
				test=new Instances(new FileReader(testFilename));
			}
			else
				test=null;
			if(test!=null)
				System.out.println(" test num: "+test.numInstances());
			System.out.println("dataset num: "+dataset.getDataSet().numInstances());
		}catch(Exception e){
			System.out.println("");
		}        
		
	}
	
	public void set_dataset(MultiLabelInstances dataset){
		this.dataset=dataset;
		
	}
	
	public void set_test(Instances test){
		this.test=test;
		
	}
	
	
	public void add_sample(List<Instance> samples){
		Instances in=dataset.getDataSet();
		in.addAll(samples);
		try {
			dataset=new MultiLabelInstances(in,xmlFilename);
		} catch (InvalidDataFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();   
		}
		
	}
	
	public Statistics statics() throws InvalidDataFormatException{
		//System.out.println(dataset.getLabelAttributes());
		Statistics stat=new Statistics();
		stat.calculateStats(dataset);
		System.out.println("statisic traning sample: "+stat);
		if(test!=null)
		{
			stat.calculateStats(new MultiLabelInstances(test,xmlFilename));
		    System.out.println("statisic test sample: "+stat);
		}
		return stat;
	}
	
	 
    //单分类效果
	public <Intances> void run_single() throws Exception {
		BinaryRelevanceTransformation brt=new BinaryRelevanceTransformation(dataset);
		BinaryRelevanceTransformation brt1=new BinaryRelevanceTransformation(new MultiLabelInstances(test,xmlFilename));//test
		String[] labelNames=dataset.getLabelNames();
		for(int i=0;i<numlabels;i++) {
			
		    //随机森林模型
			RandomForest classifier=new RandomForest();
			System.out.println("the label is : "+ labelNames[i]);
			Instances instrain=brt.transformInstances(i);
			classifier.buildClassifier(instrain);
			System.out.println("gobalinfo: \n"+classifier.globalInfo());
			System.out.println("classifier : \n"+classifier);
			//测试数据
			Instances instest=brt1.transformInstances(i);
			Evaluation ev=new Evaluation(instest);
		    ev.evaluateModel(classifier, instest);
			System.out.println(ev.toSummaryString());
			System.out.println("F-measure: "+ev.fMeasure(0)+","+ev.fMeasure(1));
			
		}
		brt=null;
		brt1=null;
	
	}
	
	//评估方法
	public List<Measure> setMeasures(int numOfLabels){
		 List<Measure> measures = new ArrayList<Measure>();
		  // add example-based measures
         measures.add(new HammingLoss());
         measures.add(new SubsetAccuracy());
         measures.add(new ExampleBasedPrecision());
         measures.add(new ExampleBasedRecall());
         measures.add(new ExampleBasedFMeasure());
         measures.add(new ExampleBasedAccuracy());
         measures.add(new ExampleBasedSpecificity());
         // add label-based measures
         measures.add(new MicroPrecision(numOfLabels));
         measures.add(new MicroRecall(numOfLabels));
         measures.add(new MicroFMeasure(numOfLabels));
         measures.add(new MicroSpecificity(numOfLabels));
         measures.add(new MacroPrecision(numOfLabels));
         measures.add(new MacroRecall(numOfLabels));
         measures.add(new MacroFMeasure(numOfLabels));
         measures.add(new MacroSpecificity(numOfLabels));
         // add ranking based measures
         measures.add(new AveragePrecision());
         measures.add(new Coverage());
         measures.add(new OneError());
         measures.add(new IsError());
         measures.add(new ErrorSetSize());
         measures.add(new RankingLoss());
    return measures;              
	}
	
	public void run_br(Classifier baseClassifier) throws Exception{
		br = new BinaryRelevance(baseClassifier);
        Evaluator eval = new Evaluator();
        
        br.build(dataset);
        System.out.println(eval.evaluate(br, new MultiLabelInstances(test,xmlFilename),setMeasures(numlabels)));
	}
	
	public void br() throws Exception {
		Classifier baseClassifier;
		//randomforest
		System.out.println("-----------------------------------BR-RandomForest-------------------------------");
		baseClassifier=new RandomForest();
		run_br(baseClassifier);
		//logistic
		System.out.println("-----------------------------------BR-Logistic-------------------------------");
		baseClassifier = new Logistic();
		run_br(baseClassifier);
		//j48
		System.out.println("-----------------------------------BR-j48-------------------------------");
		baseClassifier = new J48();
		run_br(baseClassifier);
	    //svm
		System.out.println("-----------------------------------BR-libsvm-------------------------------");
	    baseClassifier =new LibSVM();
	    run_br(baseClassifier);
		
	}
	
	
	public void run_Ada() throws Exception{
		//J48(c4.5)
		Classifier baseClassifier = new Logistic();
        adb= new AdaBoostMH();
        System.out.println(adb.getBaseClassifier());
        
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;
        /*
         *������֤ 
         *
        int numFolds = 3;
        results = eval.crossValidate(br, dataset, numFolds);
        System.out.println(results);*/
        //System.out.println(dataset.getDataSet().get(0));
        
         
        br.build(dataset);
        System.out.println(eval.evaluate(br, new MultiLabelInstances(test,xmlFilename), dataset));
        
        
	}
	
	
	
	
	public void prediction() throws InvalidDataException, ModelInitializationException, Exception{
		int numInstances = test.numInstances();
		FileWriter fw=new FileWriter(new File("pre.csv"),true);
		PrintWriter pw=new PrintWriter(fw);
		int[] label_position=dataset.getLabelIndices();//��ǩλ��
		
		for (int instanceIndex=0; instanceIndex < numInstances; instanceIndex++) {
		    Instance instance = test.instance(instanceIndex);
		    double[] labels=new double[label_position.length];
		    String str="";
			for(int j=0;j<label_position.length;j++){
				labels[j]=instance.value(label_position[j]);//���ݱ�ǩλ�û����Ӧ��ǩֵ
				str+=(String.valueOf(labels[j])+",");
			}
		    MultiLabelOutput output = br.makePrediction(instance);
		    // do necessary operations with provided prediction output, here just print it out
		    pw.print(output);
		    pw.println("true labels: ["+str+"]");
		}
		pw.flush();
		pw.close();
		fw.close();
	}
	
	/*
	 * 划分测试集、训练集,参数为划分比例
	 */
    public void split_arff(double percentage) throws InvalidDataFormatException{
    	Instances new_dataset=dataset.getDataSet().resample(new Random());//重排样本集
    	int testLen=(int)((1-percentage)*dataset.getNumInstances());
    	if(test==null)
    		test=new Instances(new_dataset,0);
    	Iterator<Instance> it=new_dataset.iterator();
    	while(testLen>0&&it.hasNext()){
    		test.add(it.next());
    		it.remove();
    		testLen--;
    	}
    	dataset=new MultiLabelInstances(new_dataset,xmlFilename);
    	System.out.println("training: "+dataset.getNumInstances()+", testing: "+test.numInstances());
	}
	

    
	/*
	 * 保存训练集，测试集到指定的参数路径中
	 */
	public void save_arff(String output_training,String output_test) throws IOException{
		 ArffSaver saver = new ArffSaver();
		 saver.setInstances(dataset.getDataSet());
		 if(output_training!=null){
			 saver.setFile(new File(output_training));
		 }
		 saver.writeBatch();
		
		 saver.setInstances(test);
		 if(output_test!=null){
			 saver.setFile(new File(output_test));
		 }
		 saver.writeBatch();
	}
	
	public void resample_simple() throws Exception {
		/*
		 *仅仅欠采样大类别的数据量 ，可记作原始数据集
		 * 
		 */
		System.out.println("underSamples dataset statistic: ");
		LabelSet labelset=new LabelSet(new double[]{0,0,0,0,0,0,0,1,0}); //大类别样本欠采样
		UnbalancedSamples.Undersampling(dataset,labelset , 0.1);//训练集大类样本欠采样，采样后的比重为80%
		LabelSet labelset1=new LabelSet(new double[]{0,0,0,0,0,0,0,0,1}); 
		UnbalancedSamples.Undersampling(dataset,labelset1 , 0.1);
		LabelSet labelset2=new LabelSet(new double[]{0,0,0,0,0,0,1,1,1}); 
		UnbalancedSamples.Undersampling(dataset,labelset2 , 0.1);
		statics();
		
	}
	
	public void resample_RUS() throws Exception {
		double p=0.05;
		while(p>=0.05) {
			System.out.println("---------------------p= "+p+"------------------------------");
			UnbalancedSamples.calML_RUS(dataset,p);//0.9 0.85  0.8 0.75 0.7 0.65 0.6 0.55 0.5
			statics();
			//split_arff(0.7); //按照70%比例划分训练集测试集
			//save_arff("training_simple"+p+".arff", "testing_simple"+p+".arff");
			p-=0.1;
			//br();
		}
	}
	
	public void resample_MLSMOTE() {
		
	}
	

	
	//-arff dataset_2.arff  -xml output_2.xml
	//-arff ../mimic2/holdout-1/training_part1.arff  -xml ../mimic2/labels.xml  -unlabeled ../mimic2/holdout-1/testval_part2.arff
	
	public static void main(String[] args) throws InvalidDataException, ModelInitializationException, Exception{
		Multi_Classifier br=new Multi_Classifier(args);
		/*FindSmallLabels fl=new FindSmallLabels(dataset);
		System.out.println(fl.inner_labels(dataset));//测试标签内的不均衡性,返回小类别标签
		System.out.println(fl.between_labels(dataset));//测试标签间的不均衡性，返回小类别标签
		*/
		//br.resample_simple();
		br.resample_RUS();
		//br.split_arff(0.7); //按照70%比例划分训练集测试集
		//br.save_arff("training_simple.arff", "testing_simple.arff");
		//br.statics();
		//br.br();
		//br.run_Ada();
	}

}
