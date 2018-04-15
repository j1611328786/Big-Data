package com.core;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.LibSVM;
import weka.filters.supervised.instance.SMOTE;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.SMO;

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
import mulan.transformations.LabelPowersetTransformation;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.*;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import mulan.classifier.transformation.AdaBoostMH;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.HierarchyBuilder;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.lazy.MLkNN;
import mulan.data.Statistics;

public class Multi_Classifier {

	private static MultiLabelInstances dataset;
	private int numlabels;
	private Instances test;
	private String xmlFilename;
	private BinaryRelevance br;
	

	public Multi_Classifier(String[] arg) {
		try {
			String arffFilename = Utils.getOption("arff", arg);
			xmlFilename = Utils.getOption("xml", arg);
			dataset = new MultiLabelInstances(arffFilename, xmlFilename);
			numlabels = dataset.getNumLabels();
			if (arg.length == 6) {
				String testFilename = Utils.getOption("unlabeled", arg);
				test = new Instances(new FileReader(testFilename));
			} else
				test = null;
		} catch (Exception e) {
			System.out.println("");
		}

	}

	public void set_dataset(MultiLabelInstances dataset) {
		this.dataset = dataset;

	}

	public void set_test(Instances test) {
		this.test = test;

	}

	public void add_sample(List<Instance> samples) {
		Instances in = dataset.getDataSet();
		in.addAll(samples);
		try {
			dataset = new MultiLabelInstances(in, xmlFilename);
		} catch (InvalidDataFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public Statistics statics() throws InvalidDataFormatException {
		// System.out.println(dataset.getLabelAttributes());
		Statistics stat = new Statistics();
		stat.calculateStats(dataset);
		System.out.println("traning sample: " + stat);
		if (test != null) {
			stat.calculateStats(new MultiLabelInstances(test, xmlFilename));
			System.out.println("statisic test sample: " + stat);
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
				Logistic classifier=new Logistic();
				System.out.println("the label is : "+ labelNames[i]);
				Instances instrain=brt.transformInstances(i);
				classifier.buildClassifier(instrain);
				//System.out.println("gobalinfo: \n"+classifier.globalInfo());
				System.out.println("classifier : \n"+classifier);
				//测试数据
				Instances instest=brt1.transformInstances(i);
				Evaluation ev=new Evaluation(instest);
			    ev.evaluateModel(classifier, instest);
				System.out.println("F-measure: "+ev.fMeasure(0)+","+ev.fMeasure(1));
				//System.out.println(classifier.coefficients());
				
			}
			brt=null;
			brt1=null;
		
		}
		
	public void printStatics(MultiLabelInstances d) throws InvalidDataFormatException {
		// System.out.println(dataset.getLabelAttributes());
		Statistics stat = new Statistics();
		stat.calculateStats(d);
		System.out.println("traning sample: " + stat);
	}

	// 评估方法
	public List<Measure> setMeasures(int numOfLabels) {
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
		//measures.add(new MicroAUC(numOfLabels));
		// add ranking based measures
		measures.add(new AveragePrecision());
		measures.add(new Coverage());
		measures.add(new OneError());
		measures.add(new IsError());
		measures.add(new ErrorSetSize());
		measures.add(new RankingLoss());

		ThresholdCurve tc = new ThresholdCurve();
		//tc.getCurve(null, arg1)
		

		return measures;
	}

	public void run_br(Classifier baseClassifier) throws Exception {
		br = new BinaryRelevance(baseClassifier);
		Evaluator eval = new Evaluator();

		br.build(dataset);
		System.out.println(eval.evaluate(br, new MultiLabelInstances(test, xmlFilename), setMeasures(numlabels)));
		
		/*LabelPowerset lp = new LabelPowerset(baseClassifier);
		lp.build(dataset);
		Evaluator eval = new Evaluator();
		System.out.println(eval.evaluate(br, new MultiLabelInstances(test, xmlFilename), setMeasures(numlabels)));*/
	}

	public void br() throws Exception {
		Classifier baseClassifier;
		// randomforest
		//System.out.println("-----------------------------------BR-RandomForest-------------------------------");
		//baseClassifier = new RandomForest();
		//run_br(baseClassifier);
		// logistic
		System.out.println("-----------------------------------BR-Logistic-------------------------------");
		baseClassifier = new Logistic();
		run_br(baseClassifier);
		// j48
		System.out.println("-----------------------------------BR-j48-------------------------------");
		baseClassifier = new J48();
		run_br(baseClassifier);
		// svm
		System.out.println("-----------------------------------BR-SVM(SMO）-------------------------------");
		baseClassifier = new SMO();
		run_br(baseClassifier);

	}
	
	public void run_HOMER() throws Exception{
		System.out.println("------------------------run_HOMER-------------------------");
		//HOMER homer=new HOMER();//默认基分类器为 J48
		HOMER homer=new HOMER(new BinaryRelevance(new Logistic()),3,HierarchyBuilder.Method.BalancedClustering);
		//HOMER homer=new HOMER(new BinaryRelevance(new SMO()),3,HierarchyBuilder.Method.BalancedClustering);
		Evaluator eval=new Evaluator();
		homer.build(dataset);
		System.out.println(eval.evaluate(homer, new MultiLabelInstances(test, xmlFilename), setMeasures(numlabels)));
		
	}
	
	public void run_RAKEL() throws Exception {
		System.out.println("------------------------run_RAKEL-------------------------");
		//RAkEL rakel=new RAkEL(new BinaryRelevance(new J48()),9,3);//默认基分类器为 J48
		//RAkEL rakel=new RAkEL(new BinaryRelevance(new Logistic()),9,3);
		RAkEL rakel=new RAkEL(new BinaryRelevance(new SMO()));
		Evaluator eval=new Evaluator();
		
		rakel.build(dataset);
		System.out.println(eval.evaluate(rakel, new MultiLabelInstances(test, xmlFilename), setMeasures(numlabels)));
		
	}
	
	public void run_MLKNN() throws InvalidDataFormatException, Exception {
		System.out.println("------------------------run_MLKNN-------------------------");
		MLkNN   mlknn=new MLkNN();
		Evaluator eval=new Evaluator();
		//mlknn.build(dataset);
		//Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        results = eval.crossValidate(mlknn, dataset, 5);
        System.out.println(results);
		//System.out.println(eval.evaluate(mlknn, new MultiLabelInstances(test, xmlFilename), setMeasures(numlabels)));
	
	}

	public void run_Ada() throws Exception {
		System.out.println("------------------------run_ada-------------------------");
		// J48(c4.5)
		Classifier baseClassifier = new J48();
		AdaBoostMH adb = new AdaBoostMH();        
		System.out.println(adb.getBaseClassifier());

		Evaluator eval = new Evaluator();
		MultipleEvaluation results;
	
		
		adb.build(dataset);
		System.out.println(eval.evaluate(adb, new MultiLabelInstances(test, xmlFilename), setMeasures(numlabels)));

	}

	public void ensemblePrediction() throws InvalidDataException, ModelInitializationException, Exception {
		MLkNN mlknn = new MLkNN();
		mlknn.build(dataset);
	
		int numInstances = dataset.getNumInstances();
		int[] label_position = dataset.getLabelIndices();

		// 每个小类标签被误分类的次数及在整个训练集中被误分类的概率
		int[] statisticEveryclassification = new int[label_position.length];
		int[] misclassification = new int[label_position.length];
		double[] misclassificationPro = new double[label_position.length];

		Instances instances = dataset.getDataSet();
		for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
			Instance instance = instances.get(instanceIndex);
			boolean[] predict = mlknn.makePrediction(instance).getBipartition();
			for (int i = 0; i < label_position.length; i++) {
				if (instance.value(label_position[i]) == 1) {
					statisticEveryclassification[i]++;
					if (!predict[i])
						misclassification[i]++;
				}
			}
		}
		for (int j = 0; j < label_position.length; j++) {
			misclassificationPro[j] = misclassification[j] * 1.0 / statisticEveryclassification[j];
			System.out.println(misclassificationPro[j]);
		}
		
		double[][] rakel=ensembleRAKEL();
		
		List<Measure> measures = setMeasures(numlabels);
		int numInstancesTest = test.numInstances(); 
		double[] output = new double[label_position.length];// 标签预测值 
		boolean[] truth = new boolean[label_position.length];// 标签真实值
		for (int i = 0; i < numInstancesTest; i++) {
			Instance instance = test.get(i);
			double[] output1 = mlknn.makePrediction(instance).getConfidences();
			double[] output2=rakel[i];
			for (int j = 0; j < output.length; j++) {
				output[j] = (1 - misclassificationPro[j]) * output1[j] +(misclassificationPro[j]+0.08) * output2[j];
				truth[j] = instance.value(label_position[j]) == 1 ? true : false;
			}
			Iterator<Measure> it = measures.iterator();
			while (it.hasNext()) { 
				Measure m= it.next();
				m.update(new MultiLabelOutput(output, 0.5), new GroundTruth(truth)); 
				}
			}
		 //System.out.println(new Evaluation(measures, new MultiLabelInstances(test, xmlFilename)));
		
	}
	
	public double[][] ensembleRAKEL() throws InvalidDataFormatException,  Exception{
		FindSmallLabels fs = new FindSmallLabels(dataset);// 获取小类样本的标签集合
		fs.between_labels();// 获取小类样本的标签集合
		Instances smallSample = fs.getSmallSample();
		Instances maxSample = fs.getMaxSample();
		
		//System.out.println("the number of small sample: "+smallSample.size()+" the number of max Sample:"+maxSample.size());
		/**
		 * 对现有的数据集统计大小类样本占的比例，小样本为8294 ，大样本为11602，为了使用RAKEL算法，需要将训练集数目控制在7000到7200之间
		 * 同时为了加大小样本的数目，缓解数据不均衡，将划分后的大小样本比例设为约2:1，即子数据集中 小样本数目约需要4100左右，大样本约需要2300左右。
		 * 由此，对现有的8294个小样本将其划分为2部分，而大样本11602则划分为5部分，划分后的子样本集进行混洗，得到10份数据集用于训练 
		 * 
		 */
		Instances[] smallSampleSplit=new Instances[2];//小样本集划分
		splitInstances(smallSample,smallSampleSplit,2);
		Instances[] maxSampleSplit=new Instances[4];//大样本集划分
		splitInstances(maxSample,maxSampleSplit,4);
		
		//smallSample=null;
		//maxSample=null;
		
		RAkEL rakel;;
		double[][] val=new double[test.size()][numlabels];
		for(int i=0;i<smallSampleSplit.length;i++)
			for(int j=0;j<maxSampleSplit.length;j++) {
				Instances small=new Instances(smallSampleSplit[i]);
				small.addAll(maxSampleSplit[j]);
				MultiLabelInstances trainingSet=new MultiLabelInstances(small,xmlFilename);
				//printStatics(trainingSet);
				rakel=new RAkEL(); 
				//rakel = new RAkEL(new BinaryRelevance(new SMO())); 
				//rakel=new RAkEL(new BinaryRelevance(new Logistic())); 
			    rakel.build(trainingSet);
				//homer=new HOMER(new BinaryRelevance(new Logistic()),3,HierarchyBuilder.Method.BalancedClustering);
				//homer.build(trainingSet);
			    for (int t = 0; t < test.size(); t++) {
					Instance instance = test.get(t);
					double[] output=rakel.makePrediction(instance).getConfidences();
					for(int r=0;r<output.length;r++) {
						val[t][r]+=output[r];
					}
			    }		
			 }
		for(int i=0;i<val.length;i++)
			for(int j=0;j<val[i].length;j++)
				val[i][j]=val[i][j]/(smallSampleSplit.length*maxSampleSplit.length);	
		return val;

	}
	


public void splitInstances(Instances instances, Instances[] sample,int k) {
    //初始化
	for(int i=0;i< k;i++) {
      sample[i]=new Instances(instances);
      sample[i].clear();			
      }
		 
	int number=instances.size();
	int eval=number/k;
	for(int i=0;i<number;i++) {
		int j=i/eval;
		if(j>=k)
			j=k-1;
	    sample[j].add(instances.get(i));
	}
}
	

	/*
	 * 划分测试集、训练集,参数为划分比例
	 */
	public void split_arff(double percentage) throws InvalidDataFormatException {
		Instances new_dataset = dataset.getDataSet().resample(new Random());// 重排样本集
		int testLen = (int) ((1 - percentage) * dataset.getNumInstances());
		if (test == null)
			test = new Instances(new_dataset, 0);
		Iterator<Instance> it = new_dataset.iterator();
		while (testLen > 0 && it.hasNext()) {
			//test.add(it.next());
			it.next();
			it.remove();
			testLen--;
		}
		dataset = new MultiLabelInstances(new_dataset, xmlFilename);
		System.out.println("training: " + dataset.getNumInstances() + ", testing: " + test.numInstances());
	}

	/*
	 * 保存训练集，测试集到指定的参数路径中
	 */
	public void save_arff(String... output) throws IOException {
		ArffSaver saver = new ArffSaver();
		if (output.length >= 1) {
			saver.setInstances(dataset.getDataSet());
			saver.setFile(new File(output[0]));
			saver.writeBatch();
		}

		if (output.length == 2) {
			saver.setInstances(test);
			saver.setFile(new File(output[1]));
			saver.writeBatch();
		}
	}

	public void resample_simple() throws Exception {
		/*
		 * 仅仅欠采样大类别的数据量 ，可记作原始数据集
		 * 
		 */
		System.out.println("underSamples dataset statistic: ");
		LabelSet labelset = new LabelSet(new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0 }); // 大类别样本欠采样
		UnbalancedSamples.Undersampling(dataset, labelset, 0.1);// 训练集大类样本欠采样，采样后的比重为80%
		LabelSet labelset1 = new LabelSet(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1 });
		UnbalancedSamples.Undersampling(dataset, labelset1, 0.1);
		LabelSet labelset2 = new LabelSet(new double[] { 0, 0, 0, 0, 0, 0, 1, 1, 1 });
		UnbalancedSamples.Undersampling(dataset, labelset2, 0.1);
		statics();

	}

	public void resample_RUS() throws Exception {
		double p = 0.1;//p取0.05  0.1 0.2

		UnbalancedSamples.calML_RUS(dataset, p);
		System.out.println("-----------------------采样后数据集统计-------------------------------");
		//statics();
		//save_arff("training_simple" + p + ".arff");
        br();
        run_Ada();
        run_RAKEL();
        run_HOMER();
        run_MLKNN();
	}

	public void resample_MLSMOTE() throws Exception {
		MLSMOTE smote = new MLSMOTE(dataset);
		smote.doMLSMOTE();
		//System.out.println("-----------------------采样后数据集统计-------------------------------");
		statics();
		//save_arff("training_simple0.05_mlsmote.arff");
        //br();
        //run_Ada();
        run_RAKEL();
        run_HOMER();
        //run_MLKNN();
	}

	public void resample_MLBBS() throws Exception {
		//ML_BBS.dobbs(dataset);
		//System.out.println("-----------------------采样后数据集统计-------------------------------");
		//statics();
	   // run_HOMER();
		//split_arff(0.4);
		//save_arff("training_simpleBBS_5.arff");
	    //br();
	    //run_Ada();
      //  run_RAKEL();
         // run_MLKNN();
	//	ensemblePrediction();
		run_single();

	}

	// -arff dataset_2.arff -xml output_2.xml
	// -arff training_simple.arff -xml output_2.xml -unlabeled testing_sample.arff
	// -arff ../mimic2/holdout-1/training_part1.arff -xml ../mimic2/labels.xml   -unlabeled ../mimic2/holdout-1/testval_part2.arff

	public static void main(String[] args) throws InvalidDataException, ModelInitializationException, Exception {
		Multi_Classifier br = new Multi_Classifier(args);
		System.out.println("-----------------------采样前数据集统计-------------------------------");
		//br.split_arff(0.9); //按照70%比例划分训练集测试集
		//br.statics();
		//br.resample_RUS();
		//br.resample_MLSMOTE();
		br.resample_MLBBS();
	}

}
