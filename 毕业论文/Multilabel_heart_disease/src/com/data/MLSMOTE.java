package com.data;

/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * MLSMOTE.java
 * 
 * Copyright (C) 2008 Ryan Lichtenwalter 
 * Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 */

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import mulan.data.MultiLabelInstances;

public class MLSMOTE extends Filter implements SupervisedFilter, OptionHandler, TechnicalInformationHandler {

	/** for serialization. */
	static final long serialVersionUID = -1653880819059250364L;

	/** the number of neighbors to use. */
	protected int m_NearestNeighbors = 5;

	/** the random seed to use. */
	protected int m_RandomSeed = 1;

	/** the percentage of SMOTE instances to create. */
	protected double m_Percentage = 100.0;


	/** whether to detect the minority class automatically. */
	protected boolean m_DetectMinorityClass = false;

	/** multi-label samples */
	public MultiLabelInstances dataset;

	/**
	 * Returns a string describing this classifier.
	 * 
	 * @return a description of the classifier suitable for displaying in the
	 *         explorer/experimenter gui
	 */

	public MLSMOTE(MultiLabelInstances dataset) {
		this.dataset = dataset;
	}

	public String globalInfo() {
		return "Resamples a dataset by applying the Multilabel Synthetic Minority Oversampling TEchnique (MLSMOTE)."
				+ " The original dataset must fit entirely in memory."
				+ " The amount of SMOTE and number of nearest neighbors may be specified."
				+ " For more information, see \n\n" + getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);

		result.setValue(Field.AUTHOR, "chengjing");
		result.setValue(Field.TITLE, "Multilabel Synthetic Minority Over-sampling Technique");
		result.setValue(Field.YEAR, "2017");

		return result;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 8108 $");
	}

	

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String randomSeedTipText() {
		return "The seed used for random sampling.";
	}

	/**
	 * Gets the random number seed.
	 *
	 * @return the random number seed.
	 */
	public int getRandomSeed() {
		return m_RandomSeed;
	}

	/**
	 * Sets the random number seed.
	 *
	 * @param value
	 *            the new random number seed.
	 */
	public void setRandomSeed(int value) {
		m_RandomSeed = value;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String percentageTipText() {
		return "The percentage of SMOTE instances to create.";
	}

	/**
	 * Sets the percentage of SMOTE instances to create.
	 * 
	 * @param value
	 *            the percentage to use
	 */
	public void setPercentage(double value) {
		if (value >= 0)
			m_Percentage = value;
		else
			System.err.println("Percentage must be >= 0!");
	}

	/**
	 * Gets the percentage of SMOTE instances to create.
	 * 
	 * @return the percentage of SMOTE instances to create
	 */
	public double getPercentage() {
		return m_Percentage;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String nearestNeighborsTipText() {
		return "The number of nearest neighbors to use.";
	}

	/**
	 * Sets the number of nearest neighbors to use.
	 * 
	 * @param value
	 *            the number of nearest neighbors to use
	 */
	public void setNearestNeighbors(int value) {
		if (value >= 1)
			m_NearestNeighbors = value;
		else
			System.err.println("At least 1 neighbor necessary!");
	}

	/**
	 * Gets the number of nearest neighbors to use.
	 * 
	 * @return the number of nearest neighbors to use
	 */
	public int getNearestNeighbors() {
		return m_NearestNeighbors;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String classValueTipText() {
		return "The index of the class value to which SMOTE should be applied. "
				+ "Use a value of 0 to auto-detect the non-empty minority class.";
	}

	
	/*
	 * compose minority class dataset also push all dataset instances 原始数据集中
	 * 提取出现小类标签的所有实例
	 */
	public Instances getLabel(int label) {
		Instances sample=dataset.clone().getDataSet();
		Iterator<Instance> it = sample.iterator();
		while (it.hasNext()) {
			Instance instance = it.next();
			int v=(int) instance.value(label);
			if (v!= 1) {
				it.remove();
			}
		}
		return sample;
	}
	
	
	

	/**
	 * The procedure implementing the SMOTE algorithm. The output instances are
	 * pushed onto the output queue for collection.
	 * 
	 * @throws Exception
	 *             if provided options cannot be executed on input instances
	 */
	public void doMLSMOTE() throws Exception {

		FindSmallLabels fsl = new FindSmallLabels(dataset);
		double meanIR = fsl.getMEANIR();// 标签间的不均衡度
		List<Integer> smalllabels = fsl.getsmalllabels();// 小标签集合

		int nearestNeighbors = getNearestNeighbors();

		for (int label : smalllabels) {
			Instances sample = getLabel(label);
            System.out.println("sample: "+sample.size());
			// compute Value Distance Metric matrices for nominal features
			Map vdmMap = new HashMap();
			Set<Attribute> attrEnum = dataset.getFeatureAttributes();
			int[] labeldices = dataset.getLabelIndices();
			for (Attribute attr : attrEnum) {
				if (attr.isNominal() || attr.isString()) {
					double[][] vdm = new double[attr.numValues()][attr.numValues()];
					vdmMap.put(attr, vdm);
					int[] featureValueCounts = new int[attr.numValues()];
					int[][] featureValueCountsByClass = new int[dataset.getNumLabels()][attr.numValues()];
					Iterator<Instance> it1 = sample.iterator();
					while (it1.hasNext()) {
						Instance instance = it1.next();
						int value = (int) instance.value(attr);
						featureValueCounts[value]++;// 统计特征中每个值出现的次数
						for (int i = 0; i < labeldices.length; i++)
							if (instance.value(labeldices[i]) == 1)
								featureValueCountsByClass[i][value]++;// 统计特征值对应出现的类标号出现的次数
					}
					for (int valueIndex1 = 0; valueIndex1 < attr.numValues(); valueIndex1++) {
						for (int valueIndex2 = 0; valueIndex2 < attr.numValues(); valueIndex2++) {
							double sum = 0;
							for (int classValueIndex = 0; classValueIndex < labeldices.length; classValueIndex++) {
								double c1i = (double) featureValueCountsByClass[classValueIndex][valueIndex1];
								double c2i = (double) featureValueCountsByClass[classValueIndex][valueIndex2];
								double c1 = (double) featureValueCounts[valueIndex1];
								double c2 = (double) featureValueCounts[valueIndex2];
								double term1 = c1i / c1;
								double term2 = c2i / c2;
								sum += Math.abs(term1 - term2);
							}
							vdm[valueIndex1][valueIndex2] = sum;
						}
					}
				}
			}

			// use this random source for all required randomness
			Random rand = new Random(getRandomSeed());

			// find the set of extra indices to use if the percentage is not evenly
			// divisible by 100
			List extraIndices = new LinkedList();
			double percentageRemainder = (getPercentage() / 100) - Math.floor(getPercentage() / 100.0);
			int extraIndicesCount = (int) (percentageRemainder * sample.numInstances());
			if (extraIndicesCount >= 1) {
				for (int i = 0; i < sample.numInstances(); i++) {
					extraIndices.add(i);
				}
			}
			Collections.shuffle(extraIndices, rand);
			extraIndices = extraIndices.subList(0, extraIndicesCount);
			Set extraIndexSet = new HashSet(extraIndices);

			// the main loop to handle computing nearest neighbors and generating SMOTE
			// examples from each instance in the original minority class data
			Instance[] nnArray = new Instance[nearestNeighbors];
			for (int i = 0; i < sample.numInstances(); i++) {
				Instance instanceI = sample.instance(i);
				// find k nearest neighbors for each instance
				List distanceToInstance = new LinkedList();
				for (int j = 0; j < sample.numInstances(); j++) {
					Instance instanceJ = sample.instance(j);
					if (i != j) {
						double distance = 0;
						Iterator<Attribute> it2 = attrEnum.iterator();
						while (it2.hasNext()) {
							Attribute attr = it2.next();
							double iVal = instanceI.value(attr);
							double jVal = instanceJ.value(attr);
							if (attr.isNumeric()) {
								distance += Math.pow(iVal - jVal, 2);
							} else {
								distance += ((double[][]) vdmMap.get(attr))[(int) iVal][(int) jVal];
							}
						}
						distance = Math.pow(distance, .5);
						distanceToInstance.add(new Object[] { distance, instanceJ });
					}
				}

				// sort the neighbors according to distance
				Collections.sort(distanceToInstance, new Comparator() {
					public int compare(Object o1, Object o2) {
						double distance1 = (Double) ((Object[]) o1)[0];
						double distance2 = (Double) ((Object[]) o2)[0];
						return Double.compare(distance1, distance2);
					}
				});

				// populate the actual nearest neighbor instance array
				Iterator entryIterator = distanceToInstance.iterator();
				int j = 0;
				while (entryIterator.hasNext() && j < nearestNeighbors) {
					nnArray[j] = (Instance) ((Object[]) entryIterator.next())[1];
					j++;
				}

				// create synthetic examples
				int n = (int) Math.floor(getPercentage() / 100);
				while (n > 0 || extraIndexSet.remove(i)) {
					double[] values = new double[attrEnum.size()+dataset.getNumLabels()];//要插入的新样本
					int nn = rand.nextInt(nearestNeighbors);
					Iterator<Attribute> it3 = attrEnum.iterator();
					while (it3.hasNext()) {
						Attribute attr = it3.next();
						if (attr.isNumeric()) {
							double dif = nnArray[nn].value(attr) - instanceI.value(attr);
							double gap = rand.nextDouble();
							values[attr.index()] = (double) (instanceI.value(attr) + gap * dif);
						} else if (attr.isDate()) {
							double dif = nnArray[nn].value(attr) - instanceI.value(attr);
							double gap = rand.nextDouble();
							values[attr.index()] = (long) (instanceI.value(attr) + gap * dif);
						} else {
							int[] valueCounts = new int[attr.numValues()];
							int iVal = (int) instanceI.value(attr);
							valueCounts[iVal]++;
							for (int nnEx = 0; nnEx < nearestNeighbors; nnEx++) {
								int val = (int) nnArray[nnEx].value(attr);
								valueCounts[val]++;
							}
							int maxIndex = 0;
							int max = Integer.MIN_VALUE;
							for (int index = 0; index < attr.numValues(); index++) {
								if (valueCounts[index] > max) {
									max = valueCounts[index];
									maxIndex = index;
								}
							}
							values[attr.index()] = maxIndex;// nominal属性使用出现频度最大值填充
						}
					}
					// 标签集生成方法，基于邻居投票
					int[] indices=dataset.getLabelIndices();
					int[] labelcounts=new int[indices.length];
					for(int t=0;t<indices.length;t++)
						labelcounts[t]=(int) instanceI.value(indices[t]);
					for(int r=0;r<labelcounts.length;r++) {
					    for(int t=0;t<nnArray.length;t++) 
						{
							labelcounts[r]+=nnArray[t].value(indices[r]);	
						}	
					    
					    if(labelcounts[r]>(nearestNeighbors+1)/2)
					    	values[indices[r]] = 1;
					    else values[indices[r]]=0;
					}
				// 生成新样本
					Instance synthetic = new DenseInstance(1.0, values);
					dataset.getDataSet().add(synthetic);
					n--;
				}
			}
		}

	}

}
