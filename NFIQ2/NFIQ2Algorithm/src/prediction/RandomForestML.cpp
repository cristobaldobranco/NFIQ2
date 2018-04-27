#include "RandomForestML.h"
#include "RandomForestTrainedParams.h"
#include "include/NFIQException.h"

#if defined WINDOWS || defined WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <math.h>
#include <float.h>
#include <time.h>
#include <numeric>  // std::accumulate

using namespace NFIQ;
using namespace cv;

std::string RandomForestML::joinRFTrainedParamsString()
{
	unsigned int size = sizeof(g_strRandomForestTrainedParams) / sizeof(g_strRandomForestTrainedParams[0]);
	std::string result = "";
	for (unsigned int i = 0; i < size; i++)
		result.append(g_strRandomForestTrainedParams[i]);
	return result;
}

RandomForestML::RandomForestML()
{
}

RandomForestML::~RandomForestML()
{
	if (m_pTrainedRF)
	{
		m_pTrainedRF->clear();
	}
}
void RandomForestML::initModule()
{
	try
	{
		// get parameters from string
		// accumulate it to a single string
		// and decode base64
		NFIQ::Data data;
		std::string params = joinRFTrainedParamsString();
		unsigned int size = params.size();
		data.fromBase64String(params);
		params = "";
		params.assign((const char*)data.data(), data.size());

		// create file storage with parameters in memory
		FileStorage fs(params.c_str(), FileStorage::READ | FileStorage::MEMORY | FileStorage::FORMAT_YAML);
		m_pTrainedRF = Algorithm::read<cv::ml::RTrees>(fs["my_random_trees"]);
	}
	catch (cv::Exception &e)
	{
		throw e;
	}
	catch (...)
	{
		throw;
	}
}

void RandomForestML::evaluate(
							  const std::list<NFIQ::QualityFeatureData> & featureVector,
							  const double & utilityValue,
							  double & qualityValue,
							  double & deviation)
{
	if (!m_pTrainedRF)
		throw NFIQ::NFIQException(NFIQ::e_Error_InvalidConfiguration, "The trained network could not be loaded for prediction!");

	deviation = 0.0; // ignore deviation here

	// copy data to structure
	Mat sample_data = Mat(1, featureVector.size(), CV_32FC1);
	std::list<NFIQ::QualityFeatureData>::const_iterator it_feat;
	unsigned int counterFeatures = 0;
	for (it_feat = featureVector.begin(); it_feat != featureVector.end(); it_feat++)
	{
		if (it_feat->featureDataType == e_QualityFeatureDataTypeDouble)
			sample_data.at<float>(0, counterFeatures) = (float)it_feat->featureDataDouble;
		else
			sample_data.at<float>(0, counterFeatures) = 0.0f;
		counterFeatures++;
	}

	// replacement of the predict_prob(...) method of OpenCV 2.4 CvRTrees
	cv::Mat votes;
	m_pTrainedRF->getVotes(sample_data, votes, 0);

	int a = votes.at<int>(1, 0);
	int b = votes.at<int>(1, 1);

	// returns probability that between 0 and 1 that result belongs to second class
	float prob = b / ((float) a + b);

	// return quality value
	qualityValue = (int)((prob * 100) + 0.5);
}

std::string RandomForestML::getModuleID()
{
	return "NFIQ2_RandomForest";
}

