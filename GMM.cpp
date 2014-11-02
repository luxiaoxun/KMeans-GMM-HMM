/***************************************************************************

Module Name:

	Gaussian Mixture Model with Diagonal Covariance Matrix

History:

	2003/11/01	Fei Wang
	2013 luxiaoxun
***************************************************************************/
#include <math.h>
#include <iostream>
#include "GMM.h"
#include "KMeans.h"
using namespace std;

//double M_PI=3.14159;

GMM::GMM(int dimNum, int mixNum)
{
	m_dimNum = dimNum;
	m_mixNum = mixNum;

	m_maxIterNum = 100;
	m_endError = 0.001;

	Allocate();

	for (int i = 0; i < m_mixNum; i++)
	{
		m_priors[i] = 1.0 / m_mixNum;

		for (int d = 0; d < m_dimNum; d++)
		{
			m_means[i][d] = 0;
			m_vars[i][d] = 1;
		}
	}
}

GMM::~GMM()
{
	Dispose();
}

void GMM::Allocate()
{
	m_priors = new double[m_mixNum];
	m_means = new double*[m_mixNum];
	m_vars = new double*[m_mixNum];

	for (int i = 0; i < m_mixNum; i++)
	{
		m_means[i] = new double[m_dimNum];
		m_vars[i] = new double[m_dimNum];
	}

	m_minVars = new double[m_dimNum];
}

void GMM::Dispose()
{
	delete[] m_priors;

	for (int i = 0; i < m_mixNum; i++)
	{
		delete[] m_means[i];
		delete[] m_vars[i];
	}
	delete[] m_means;
	delete[] m_vars;

	delete[] m_minVars;
}

void GMM::Copy(GMM* gmm)
{
	assert(m_mixNum == gmm->m_mixNum && m_dimNum == gmm->m_dimNum);

	for (int i = 0; i < m_mixNum; i++)
	{
		m_priors[i] = gmm->Prior(i);
		memcpy(m_means[i], gmm->Mean(i), sizeof(double) * m_dimNum);
		memcpy(m_vars[i], gmm->Variance(i), sizeof(double) * m_dimNum);
	}
	memcpy(m_minVars, gmm->m_minVars, sizeof(double) * m_dimNum);
}

double GMM::GetProbability(const double* sample)
{
	double p = 0;
	for (int i = 0; i < m_mixNum; i++)
	{
		p += m_priors[i] * GetProbability(sample, i);
	}
	return p;
}

double GMM::GetProbability(const double* x, int j)
{
	double p = 1;
	for (int d = 0; d < m_dimNum; d++)
	{
		p *= 1 / sqrt(2 * 3.14159 * m_vars[j][d]);
		p *= exp(-0.5 * (x[d] - m_means[j][d]) * (x[d] - m_means[j][d]) / m_vars[j][d]);
	}
	return p;
}

void GMM::Train(const char* sampleFileName)
{
	//DumpSampleFile(sampleFileName);
	Init(sampleFileName);

	ifstream sampleFile(sampleFileName, ios_base::binary);
	assert(sampleFile);

	int size = 0;
	sampleFile.seekg(0, ios_base::beg);
	sampleFile.read((char*)&size, sizeof(int));

	// Reestimation
	bool loop = true;
	double iterNum = 0;
	double lastL = 0;
	double currL = 0;
	int unchanged = 0;
	double* x = new double[m_dimNum];	// Sample data
	double* next_priors = new double[m_mixNum];
	double** next_vars = new double*[m_mixNum];
	double** next_means = new double*[m_mixNum];

	for (int i = 0; i < m_mixNum; i++)
	{
		next_means[i] = new double[m_dimNum];
		next_vars[i] = new double[m_dimNum];
	}

	while (loop)
	{
		// Clear buffer for reestimation
		memset(next_priors, 0, sizeof(double) * m_mixNum);
		for (int i = 0; i < m_mixNum; i++)
		{
			memset(next_vars[i], 0, sizeof(double) * m_dimNum);
			memset(next_means[i], 0, sizeof(double) * m_dimNum);
		}

		lastL = currL;
		currL = 0;

		// Predict
		sampleFile.seekg(2 * sizeof(int), ios_base::beg);
		for (int k = 0; k < size; k++)
		{
			sampleFile.read((char*)x, sizeof(double) * m_dimNum);
			double p = GetProbability(x);

			for (int j = 0; j < m_mixNum; j++)
			{
				double pj = GetProbability(x, j) * m_priors[j] / p;
				next_priors[j] += pj;
				for (int d = 0; d < m_dimNum; d++)
				{
					next_means[j][d] += pj * x[d];
					next_vars[j][d] += pj* x[d] * x[d];
				}
			}
			currL += (p > 1E-20) ? log10(p) : -20;
		}
		currL /= size;

		// Reestimation: generate new priors, means and variances.
		for (int j = 0; j < m_mixNum; j++)
		{
			m_priors[j] = next_priors[j] / size;

			if (m_priors[j] > 0)
			{
				for (int d = 0; d < m_dimNum; d++)
				{
					m_means[j][d] = next_means[j][d] / next_priors[j];
					m_vars[j][d] = next_vars[j][d] / next_priors[j] - m_means[j][d] * m_means[j][d];
					if (m_vars[j][d] < m_minVars[d])
					{
						m_vars[j][d] = m_minVars[d];
					}
				}
			}
		}

		// Terminal conditions
		iterNum++;
		if (fabs(currL - lastL) < m_endError * fabs(lastL))
		{
			unchanged++;
		}
		if (iterNum >= m_maxIterNum || unchanged >= 3)
		{
			loop = false;
		}
		//--- Debug ---
		//cout << "Iter: " << iterNum << ", Average Log-Probability: " << currL << endl;
	}

	sampleFile.close();
	delete[] next_priors;
	for (int i = 0; i < m_mixNum; i++)
	{
		delete[] next_means[i];
		delete[] next_vars[i];
	}
	delete[] next_means;
	delete[] next_vars;
	delete[] x;
}

void GMM::Train(double *data, int N)
{
	Init(data,N);

	int size = N;

	// Reestimation
	bool loop = true;
	double iterNum = 0;
	double lastL = 0;
	double currL = 0;
	int unchanged = 0;
	double* x = new double[m_dimNum];	// Sample data
	double* next_priors = new double[m_mixNum];
	double** next_vars = new double*[m_mixNum];
	double** next_means = new double*[m_mixNum];

	for (int i = 0; i < m_mixNum; i++)
	{
		next_means[i] = new double[m_dimNum];
		next_vars[i] = new double[m_dimNum];
	}

	while (loop)
	{
		// Clear buffer for reestimation
		memset(next_priors, 0, sizeof(double) * m_mixNum);
		for (int i = 0; i < m_mixNum; i++)
		{
			memset(next_vars[i], 0, sizeof(double) * m_dimNum);
			memset(next_means[i], 0, sizeof(double) * m_dimNum);
		}

		lastL = currL;
		currL = 0;

		// Predict
		for (int k = 0; k < size; k++)
		{
			for(int j=0;j<m_dimNum;j++)
				x[j]=data[k*m_dimNum+j];
			double p = GetProbability(x);

			for (int j = 0; j < m_mixNum; j++)
			{
				double pj = GetProbability(x, j) * m_priors[j] / p;

				next_priors[j] += pj;

				for (int d = 0; d < m_dimNum; d++)
				{
					next_means[j][d] += pj * x[d];
					next_vars[j][d] += pj* x[d] * x[d];
				}
			}

			currL += (p > 1E-20) ? log10(p) : -20;
		}
		currL /= size;

		// Reestimation: generate new priors, means and variances.
		for (int j = 0; j < m_mixNum; j++)
		{
			m_priors[j] = next_priors[j] / size;

			if (m_priors[j] > 0)
			{
				for (int d = 0; d < m_dimNum; d++)
				{
					m_means[j][d] = next_means[j][d] / next_priors[j];
					m_vars[j][d] = next_vars[j][d] / next_priors[j] - m_means[j][d] * m_means[j][d];
					if (m_vars[j][d] < m_minVars[d])
					{
						m_vars[j][d] = m_minVars[d];
					}
				}
			}
		}

		// Terminal conditions
		iterNum++;
		if (fabs(currL - lastL) < m_endError * fabs(lastL))
		{
			unchanged++;
		}
		if (iterNum >= m_maxIterNum || unchanged >= 3)
		{
			loop = false;
		}

		//--- Debug ---
		//cout << "Iter: " << iterNum << ", Average Log-Probability: " << currL << endl;
	}
	delete[] next_priors;
	for (int i = 0; i < m_mixNum; i++)
	{
		delete[] next_means[i];
		delete[] next_vars[i];
	}
	delete[] next_means;
	delete[] next_vars;
	delete[] x;
}

void GMM::Init(double *data, int N)
{
	const double MIN_VAR = 1E-10;

	KMeans* kmeans = new KMeans(m_dimNum, m_mixNum);
	kmeans->SetInitMode(KMeans::InitUniform);
	int *Label;
	Label=new int[N];
	kmeans->Cluster(data,N,Label);

	int* counts = new int[m_mixNum];
	double* overMeans = new double[m_dimNum];	// Overall mean of training data
	for (int i = 0; i < m_mixNum; i++)
	{
		counts[i] = 0;
		m_priors[i] = 0;
		memcpy(m_means[i], kmeans->GetMean(i), sizeof(double) * m_dimNum);
		memset(m_vars[i], 0, sizeof(double) * m_dimNum);
	}
	memset(overMeans, 0, sizeof(double) * m_dimNum);
	memset(m_minVars, 0, sizeof(double) * m_dimNum);

	int size = 0;
	size=N;

	double* x = new double[m_dimNum];
	int label = -1;

	for (int i = 0; i < size; i++)
	{
		for(int j=0;j<m_dimNum;j++)
			x[j]=data[i*m_dimNum+j];
		label=Label[i];

		// Count each Gaussian
		counts[label]++;
		double* m = kmeans->GetMean(label);
		for (int d = 0; d < m_dimNum; d++)
		{
			m_vars[label][d] += (x[d] - m[d]) * (x[d] - m[d]);
		}

		// Count the overall mean and variance.
		for (int d = 0; d < m_dimNum; d++)
		{
			overMeans[d] += x[d];
			m_minVars[d] += x[d] * x[d];
		}
	}

	// Compute the overall variance (* 0.01) as the minimum variance.
	for (int d = 0; d < m_dimNum; d++)
	{
		overMeans[d] /= size;
		m_minVars[d] = max(MIN_VAR, 0.01 * (m_minVars[d] / size - overMeans[d] * overMeans[d]));
	}

	// Initialize each Gaussian.
	for (int i = 0; i < m_mixNum; i++)
	{
		m_priors[i] = 1.0 * counts[i] / size;

		if (m_priors[i] > 0)
		{
			for (int d = 0; d < m_dimNum; d++)
			{
				m_vars[i][d] = m_vars[i][d] / counts[i];

				// A minimum variance for each dimension is required.
				if (m_vars[i][d] < m_minVars[d])
				{
					m_vars[i][d] = m_minVars[d];
				}
			}
		}
		else
		{
			memcpy(m_vars[i], m_minVars, sizeof(double) * m_dimNum);
			cout << "[WARNING] Gaussian " << i << " of GMM is not used!\n";
		}
	}
	delete kmeans;
	delete[] x;
	delete[] counts;
	delete[] overMeans;
	delete[] Label;

}

void GMM::Init(const char* sampleFileName)
{
	const double MIN_VAR = 1E-10;

	KMeans* kmeans = new KMeans(m_dimNum, m_mixNum);
	kmeans->SetInitMode(KMeans::InitUniform);
	kmeans->Cluster(sampleFileName, "gmm_init.tmp");

	int* counts = new int[m_mixNum];
	double* overMeans = new double[m_dimNum];	// Overall mean of training data
	for (int i = 0; i < m_mixNum; i++)
	{
		counts[i] = 0;
		m_priors[i] = 0;
		memcpy(m_means[i], kmeans->GetMean(i), sizeof(double) * m_dimNum);
		memset(m_vars[i], 0, sizeof(double) * m_dimNum);
	}
	memset(overMeans, 0, sizeof(double) * m_dimNum);
	memset(m_minVars, 0, sizeof(double) * m_dimNum);

	// Open the sample and label file to initialize the model
	ifstream sampleFile(sampleFileName, ios_base::binary);
	assert(sampleFile);

	ifstream labelFile("gmm_init.tmp", ios_base::binary);
	assert(labelFile);

	int size = 0;
	sampleFile.read((char*)&size, sizeof(int));
	sampleFile.seekg(2 * sizeof(int), ios_base::beg);
	labelFile.seekg(sizeof(int), ios_base::beg);

	double* x = new double[m_dimNum];
	int label = -1;

	for (int i = 0; i < size; i++)
	{
		sampleFile.read((char*)x, sizeof(double) * m_dimNum);
		labelFile.read((char*)&label, sizeof(int));

		// Count each Gaussian
		counts[label]++;
		double* m = kmeans->GetMean(label);
		for (int d = 0; d < m_dimNum; d++)
		{
			m_vars[label][d] += (x[d] - m[d]) * (x[d] - m[d]);
		}

		// Count the overall mean and variance.
		for (int d = 0; d < m_dimNum; d++)
		{
			overMeans[d] += x[d];
			m_minVars[d] += x[d] * x[d];
		}
	}

	// Compute the overall variance (* 0.01) as the minimum variance.
	for (int d = 0; d < m_dimNum; d++)
	{
		overMeans[d] /= size;
		m_minVars[d] = max(MIN_VAR, 0.01 * (m_minVars[d] / size - overMeans[d] * overMeans[d]));
	}

	// Initialize each Gaussian.
	for (int i = 0; i < m_mixNum; i++)
	{
		m_priors[i] = 1.0 * counts[i] / size;

		if (m_priors[i] > 0)
		{
			for (int d = 0; d < m_dimNum; d++)
			{
				m_vars[i][d] = m_vars[i][d] / counts[i];

				// A minimum variance for each dimension is required.
				if (m_vars[i][d] < m_minVars[d])
				{
					m_vars[i][d] = m_minVars[d];
				}
			}
		}
		else
		{
			memcpy(m_vars[i], m_minVars, sizeof(double) * m_dimNum);
			cout << "[WARNING] Gaussian " << i << " of GMM is not used!\n";
		}
	}

	delete kmeans;
	delete[] x;
	delete[] counts;
	delete[] overMeans;

	sampleFile.close();
	labelFile.close();
}

void GMM::DumpSampleFile(const char* fileName)
{
	ifstream sampleFile(fileName, ios_base::binary);
	assert(sampleFile);

	int size = 0;
	sampleFile.read((char*)&size, sizeof(int));
	cout << size << endl;

	int dim = 0;
	sampleFile.read((char*)&dim, sizeof(int));
	cout << dim << endl;

	double* f = new double[dim];
	for (int i = 0; i < size; i++)
	{
		sampleFile.read((char*)f, sizeof(double) * dim);

		cout << i << ":";
		for (int j = 0; j < dim; j++)
		{
			cout << " " << f[j];
		}
		cout << endl;
	}

	delete[] f;
	sampleFile.close();
}

ostream& operator<<(ostream& out, GMM& gmm)
{
	out << "<GMM>" << endl;
	out << "<DimNum> " << gmm.m_dimNum << " </DimNum>" << endl;
	out << "<MixNum> " << gmm.m_mixNum << " </MixNum>" << endl;

	out << "<Prior> ";
	for (int i = 0; i < gmm.m_mixNum; i++)
	{
		out << gmm.m_priors[i] << " ";
	}
	out << "</Prior>" << endl;

	out << "<Mean>" << endl;
	for (int i = 0; i < gmm.m_mixNum; i++)
	{
		for (int d = 0; d < gmm.m_dimNum; d++)
		{
			out << gmm.m_means[i][d] << " ";
		}
		out << endl;
	}
	out << "</Mean>" << endl;

	out << "<Variance>" << endl;
	for (int i = 0; i < gmm.m_mixNum; i++)
	{
		for (int d = 0; d < gmm.m_dimNum; d++)
		{
			out << gmm.m_vars[i][d] << " ";
		}
		out << endl;
	}
	out << "</Variance>" << endl;

	out << "</GMM>" << endl;

	return out;
}

istream& operator>>(istream& in, GMM& gmm)
{
	char label[50];
	in >> label; // "<GMM>"
	assert(strcmp(label, "<GMM>") == 0);

	gmm.Dispose();

	in >> label >> gmm.m_dimNum >> label; // "<DimNum>"
	in >> label >> gmm.m_mixNum >> label; // "<MixNum>"

	gmm.Allocate();

	in >> label; // "<Prior>"
	for (int i = 0; i < gmm.m_mixNum; i++)
	{
		in >> gmm.m_priors[i];
	}
	in >> label;

	in >> label; // "<Mean>"
	for (int i = 0; i < gmm.m_mixNum; i++)
	{
		for (int d = 0; d < gmm.m_dimNum; d++)
		{
			in >> gmm.m_means[i][d];
		}
	}
	in >> label;

	in >> label; // "<Variance>"
	for (int i = 0; i < gmm.m_mixNum; i++)
	{
		for (int d = 0; d < gmm.m_dimNum; d++)
		{
			in >> gmm.m_vars[i][d];
		}
	}
	in >> label;

	in >> label; // "</GMM>"
	return in;
}
