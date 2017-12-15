package com.data;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.*;

/**
 * Created by Administrator on 2017/3/29.
 */
public class _csvfile {

	/**
	 * 瀵煎嚭
	 *
	 * @param file
	 *            csv鏂囦欢(璺緞+鏂囦欢鍚�)锛宑sv鏂囦欢涓嶅瓨鍦ㄤ細鑷姩鍒涘缓
	 * @param dataList
	 *            鏁版嵁
	 * @return
	 * @throws IOException 
	 */
	public static void csvtoarff(String input, String output) throws IOException {

		// load CSV
	    CSVLoader loader = new CSVLoader();
	    loader.setFieldSeparator("|");
	    loader.setSource(new File(input));
	    Instances data = loader.getDataSet();
	 
	    // save ARFF
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(output));
	    //saver.setDestination(new File(output));
	    saver.writeBatch();
	 }
	
	

	 

	public static void main(String[] args) {
		
		try {
			_csvfile.csvtoarff("../Dataset/dataset_2.csv", "../Dataset/dataset_2.arff");
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
}
