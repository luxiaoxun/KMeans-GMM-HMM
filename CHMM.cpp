/***************************************************************************
Module Name:
	Continuous Observation Hidden Markov Model with Gaussian Mixture

History:
	2003/12/13	Fei Wang
	2013 luxiaoxun
***************************************************************************/
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <strstream>
#include "CHMM.h"
using namespace std;


CHMM::CHMM(int stateNum, int dimNum, int mixNum)
{
	m_stateNum = stateNum;
	m_maxIterNum = 100;
	m_endError = 0.001;

	Allocate(stateNum, dimNum, mixNum);

	for (int i = 0; i < m_stateNum; i++)
	{
		// The initial probabilities
		m_stateInit[i] = 1.0 / m_stateNum;

		// The transition probabilities
		for (int j = 0; j <= m_stateNum; j++)
		{
			m_stateTran[i][j] = 1.0 / (m_stateNum + 1);
		}
	}
}

CHMM::~CHMM()
{
	Dispose();
}

void CHMM::Allocate(int state, int dim, int mix)
{
	m_stateModel = new GMM*[state];
	m_stateInit = new double[state];
	m_stateTran = new double*[state];

	for (int i = 0; i < state; i++)
	{
		m_stateModel[i] = new GMM(dim, mix);
		m_stateTran[i] = new double[state + 1]; // Add a final state
	}
}

void CHMM::Dispose()
{
	for (int i = 0; i < m_stateNum; i++)
	{
		delete m_stateModel[i];
		delete[] m_stateTran[i];
	}
	delete[] m_stateModel;
	delete[] m_stateTran;
	delete[] m_stateInit;
}

void CHMM::Zero()
{
	for (int i = 0; i < m_stateNum; i++)
	{
		// The initial probabilities
		m_stateInit[i] = 0;

		// The transition probabilities
		for (int j = 0; j < m_stateNum + 1; j++)
		{
			m_stateTran[i][j] = 0;
		}
	}
}

void CHMM::Norm()
{
	double count = 0;
	int i,j;
	for ( j = 0; j < m_stateNum; j++)
	{
		count += m_stateInit[j];
	}
	for ( j = 0; j < m_stateNum; j++)
	{
		m_stateInit[j] /= count;
	}

	for (i = 0; i < m_stateNum; i++)
	{
		count = 0;
		for ( j = 0; j < m_stateNum; j++)
		{
			count += m_stateTran[i][j];
		}
		if (count > 0)
		{
			for ( j = 0; j < m_stateNum + 1; j++)
			{
				m_stateTran[i][j] /= count;
			}
		}
	}
}

double CHMM::GetStateInit(int i)
{
	assert(i >= 0 && i < m_stateNum);
	return m_stateInit[i];
}

double CHMM::GetStateFinal(int i)
{
	assert(i >= 0 && i < m_stateNum);
	return m_stateTran[i][m_stateNum];
}

double CHMM::GetStateTrans(int i, int j)
{
	assert(i >= 0 && i < m_stateNum && j >= 0 && j < m_stateNum);
	return m_stateTran[i][j];
}

GMM* CHMM::GetStateModel(int i)
{
	assert(i >= 0 && i < m_stateNum);
	return m_stateModel[i];
}

double CHMM::GetProbability(std::vector<double*>& seq)
{
	vector<int> state;
	return Decode(seq, state);
}

//Viterbi Decode
//vector state: save the best state seqence to generate the seq
double CHMM::Decode(vector<double*>& seq, vector<int>& state)
{
	// Viterbi
	int size = (int)seq.size();
	double* lastLogP = new double[m_stateNum];
	double* currLogP = new double[m_stateNum];
	int** path = new int*[size];
	int i,j,t;

	// Init
	path[0] = new int[m_stateNum];
	for ( i = 0; i < m_stateNum; i++)
	{
		currLogP[i] = LogProb(m_stateInit[i]) + LogProb(m_stateModel[i]->GetProbability(seq[0]));
		path[0][i] = -1;
	}

	// Recursion
	for ( t = 1; t < size; t++)  //对每一个观测，求属于每个状态的当前最大累加概率
	{
		path[t] = new int[m_stateNum];
		double* temp = lastLogP;
		lastLogP = currLogP;
		currLogP = temp;

		for ( i = 0; i < m_stateNum; i++)
		{
			currLogP[i] = -1e308;
			// Searching the max for last state.
			for ( j = 0; j < m_stateNum; j++)
			{
				double l = lastLogP[j] + LogProb(m_stateTran[j][i]);
				if (l > currLogP[i])
				{
					currLogP[i] = l;
					path[t][i] = j;
				}
			}
			currLogP[i] += LogProb(m_stateModel[i]->GetProbability(seq[t]));
		}
	}

	// Termination
	int finalState = 0;
	double prob = -1e308;
	for ( i = 0; i < m_stateNum; i++)
	{
		if (currLogP[i] > prob)
		{
			prob = currLogP[i];
			finalState = i;
		}
	}

	// Decode
	state.push_back(finalState);
	for ( t = size - 2; t >=0; t--)
	{
		int stateIndex = path[t+1][state.back()];
		state.push_back(stateIndex);
	}

	// Reverse the state list
	reverse(state.begin(), state.end());

	// Clean up
	delete[] lastLogP;
	delete[] currLogP;
	for ( i = 0; i < size; i++)
	{
		delete[] path[i];
	}
	delete[] path;

	prob = exp(prob / size);
	return prob;
}

/*	SampleFile: <size><dim><seq_size><seq_data>...<seq_size>...*/
void CHMM::Init(const char* sampleFileName)
{
	//--- Debug ---//
	//DumpSampleFile(sampleFileName);

	// Check the sample file
	ifstream sampleFile(sampleFileName, ios_base::binary);
	assert(sampleFile);

	int i,j;
	int size = 0;
	int dim = 0;
	sampleFile.read((char*)&size, sizeof(int));  //读样本数
	sampleFile.read((char*)&dim, sizeof(int));   //读取特征维数
	assert(size >= 3);
	assert(dim == m_stateModel[0]->GetDimNum());

	//这里为从左到右型，第一个状态的初始概率为0.5, 其他状态的初始概率之和为0.5,
	//每个状态到自身的转移概率为0.5, 到下一个状态的转移概率为0.5.
	//此处的初始化主要是对混合高斯模型进行初始化
	for ( i = 0; i < m_stateNum; i++)
	{
		// The initial probabilities
		if(i == 0)
			m_stateInit[i] = 0.5;
		else
            m_stateInit[i] = 0.5 / float(m_stateNum-1);

		// The transition probabilities
		for ( j = 0; j <= m_stateNum; j++)
		{
			if((i == j)||( j == i+1))
				m_stateTran[i][j] = 0.5;
		}
	}

	vector<double*> *gaussseq;
	gaussseq= new vector<double*>[m_stateNum];

	for ( i = 0; i < size; i++)//处理每个样本产生的特征序列
	{
		int seq_size = 0;
		sampleFile.read((char*)&seq_size, sizeof(int));  //序列的长度

		double r = float(seq_size)/float(m_stateNum); //每个状态有r个dim维的特征向量
		for ( j = 0; j < seq_size; j++)
		{
			double* x = new double[dim];
			sampleFile.read((char*)x, sizeof(double) * dim);
			//把特征序列平均分配给每个状态
			gaussseq[int(j/r)].push_back(x);
		}
	}

	char** stateFileName = new char*[m_stateNum];
	ofstream* stateFile = new ofstream[m_stateNum];
	int* stateDataSize = new int[m_stateNum];

	for ( i = 0; i < m_stateNum; i++)
	{
		stateFileName[i] = new char[20];
		ostrstream str(stateFileName[i], 20);
		str << "chmm_s" << i << ".tmp" << '\0';
	}
	//将每个状态的特征序列保存到文件中，并初始化GMM
	for ( i = 0; i < m_stateNum; i++)
	{
		stateFile[i].open(stateFileName[i], ios_base::binary);
		stateDataSize[i] = gaussseq[i].size();
		stateFile[i].write((char*)&stateDataSize[i], sizeof(int));
		stateFile[i].write((char*)&dim, sizeof(int));
		double* x = new double[dim];
		for( j = 0; j < stateDataSize[i]; j++)
		{
			x = (double*)gaussseq[i].at(j);
            stateFile[i].write((char*)x, sizeof(double) * dim);
		}
		delete x;
		stateFile[i].close();
		//使用Kmeans算法初始化状态的每个GMM
		m_stateModel[i]->Train(stateFileName[i]);
		gaussseq[i].clear();
	}

	for ( i = 0; i < m_stateNum; i++)
		delete[] stateFileName[i];

	delete[] stateFileName;
	delete[] stateFile;
	delete[] stateDataSize;
	delete[] gaussseq;
}

/*	SampleFile: <size><dim><seq_size><seq_data>...<seq_size>...*/
void CHMM::Train(const char* sampleFileName)
{
	Init(sampleFileName);

	//--- Debug ---//
	//DumpSampleFile(sampleFileName);

	// Check the sample file
	ifstream sampleFile(sampleFileName, ios_base::binary);
	assert(sampleFile);
	int i,j;

	int size = 0;
	int dim = 0;
	sampleFile.read((char*)&size, sizeof(int));
	sampleFile.read((char*)&dim, sizeof(int));
	assert(size >= 3);
	assert(dim == m_stateModel[0]->GetDimNum());

	// Buffer for new model
	int* stateInitNum = new int[m_stateNum];
	int** stateTranNum = new int*[m_stateNum];
	char** stateFileName = new char*[m_stateNum];
	ofstream* stateFile = new ofstream[m_stateNum];
	int* stateDataSize = new int[m_stateNum];

	for ( i = 0; i < m_stateNum; i++)
	{
		stateTranNum[i] = new int[m_stateNum + 1];
		stateFileName[i] = new char[20];
		ostrstream str(stateFileName[i], 20);
		str << "chmm_s" << i << ".tmp" << '\0';
	}

	bool loop = true;
	double currL = 0;
	double lastL = 0;
	int iterNum = 0; //迭代次数
	int unchanged = 0;
	vector<int> state;
	vector<double*> seq;

	while (loop)
	{
		lastL = currL;
		currL = 0;

		// Clear buffer and open temp data files
		for ( i = 0; i < m_stateNum; i++)
		{
			stateDataSize[i] = 0;
			stateFile[i].open(stateFileName[i], ios_base::binary);
			stateFile[i].write((char*)&stateDataSize[i], sizeof(int));
			stateFile[i].write((char*)&dim, sizeof(int));
			memset(stateTranNum[i], 0, sizeof(int) * (m_stateNum + 1));
		}
		memset(stateInitNum, 0, sizeof(int) * m_stateNum);

		// Predict: obtain the best path
		sampleFile.seekg(sizeof(int) * 2, ios_base::beg);
		for ( i = 0; i < size; i++)
		{
			int seq_size = 0;
			sampleFile.read((char*)&seq_size, sizeof(int));

			for ( j = 0; j < seq_size; j++)
			{
				double* x = new double[dim];
				sampleFile.read((char*)x, sizeof(double) * dim);
				seq.push_back(x);
			}

			currL += LogProb(Decode(seq, state)); //Viterbi解码

			stateInitNum[state[0]]++;
			for ( j = 0; j < seq_size; j++)
			{
				stateFile[state[j]].write((char*)seq[j], sizeof(double) * dim);
				stateDataSize[state[j]]++;
				if (j > 0)
				{
					stateTranNum[state[j-1]][state[j]]++;
				}
			}
			stateTranNum[state[j-1]][m_stateNum]++; // Final state

			for ( j = 0; j < seq_size; j++)
			{
				delete[] seq[j];
			}
			state.clear();
			seq.clear();
		}
		currL /= size;

		// Close temp data files
		for ( i = 0; i < m_stateNum; i++)
		{
			stateFile[i].seekp(0, ios_base::beg);
			stateFile[i].write((char*)&stateDataSize[i], sizeof(int));
			stateFile[i].close();
		}

		// Reestimate: stateModel, stateInit, stateTran
		int count = 0;
		for ( j = 0; j < m_stateNum; j++)
		{
			if (stateDataSize[j] > m_stateModel[j]->GetMixNum() * 2)
			{
				//m_stateModel[j]->DumpSampleFile(stateFileName[j]);
				m_stateModel[j]->Train(stateFileName[j]);
			}
			count += stateInitNum[j];
		}
		for ( j = 0; j < m_stateNum; j++)
		{
			m_stateInit[j] = 1.0 * stateInitNum[j] / count;
		}
		for ( i = 0; i < m_stateNum; i++)
		{
			count = 0;
			for ( j = 0; j < m_stateNum + 1; j++)
			{
				count += stateTranNum[i][j];
			}
			if (count > 0)
			{
				for ( j = 0; j < m_stateNum + 1; j++)
				{
					m_stateTran[i][j] = 1.0 * stateTranNum[i][j] / count;
				}
			}
		}
		// Terminal conditions
		iterNum++;
		unchanged = (currL - lastL < m_endError * fabs(lastL)) ? (unchanged + 1) : 0;
		if (iterNum >= m_maxIterNum || unchanged >= 3)
		{
			loop = false;
		}
		//DEBUG
		//cout << "Iter: " << iterNum << ", Average Log-Probability: " << currL << endl;
	}

	for ( i = 0; i < m_stateNum; i++)
	{
		delete[] stateTranNum[i];
		delete[] stateFileName[i];
	}
	delete[] stateTranNum;
	delete[] stateFileName;
	delete[] stateFile;
	delete[] stateInitNum;
	delete[] stateDataSize;
}

double CHMM::getTransProb(int i,int j)
{
	if( i < 0 || i > m_stateNum || j < 0 || j > m_stateNum)
		return -100;
	return LogProb(m_stateTran[i][j]);
}

/*	SampleFile: <size><dim><seq_size><seq_data>...<seq_size><seq_data>...*/
void CHMM::DumpSampleFile(const char* fileName)
{
	ifstream sampleFile(fileName, ios_base::binary);
	assert(sampleFile);

	int size = 0;
	int i,j;
	sampleFile.read((char*)&size, sizeof(int));
	cout << size << endl;

	int dim = 0;
	sampleFile.read((char*)&dim, sizeof(int));
	cout << dim << endl;

	double* f = new double[dim];

	for ( i = 0; i < size; i++)
	{
		int seq_size = 0;
		sampleFile.read((char*)&seq_size, sizeof(int));

		cout << seq_size << endl;
		for ( j = 0; j < seq_size; j++)
		{
			sampleFile.read((char*)f, sizeof(double) * dim);
			for (int d = 0; d < dim; d++)
			{
				cout << f[d] << " ";
			}
			cout << endl;
		}
	}
	sampleFile.close();

	delete[] f;
}

double CHMM::LogProb(double p)
{
	return (p > 1e-20) ? log10(p) : -20;
}

ostream& operator<<(ostream& out, CHMM& hmm)
{
	int i,j;
	out << "<CHMM>" << endl;
	out << "<StateNum> " << hmm.m_stateNum << " </StateNum>" << endl;

	for (i = 0; i < hmm.m_stateNum; i++)
	{
		out << *hmm.m_stateModel[i];
	}

	out << "<Init> ";
	for ( i = 0; i < hmm.m_stateNum; i++)
	{
		out << hmm.m_stateInit[i] << " ";
	}
	out << "</Init>" << endl;

	out << "<Tran>" << endl;
	for ( i = 0; i < hmm.m_stateNum; i++)
	{
		for ( j = 0; j < hmm.m_stateNum + 1; j++)
		{
			out << hmm.m_stateTran[i][j] << " ";
		}
		out << endl;
	}
	out << "</Tran>" << endl;

	out << "</CHMM>" << endl;
	return out;
}

istream& operator>>(istream& in, CHMM& hmm)
{
	char label[20];
	int i,j;
	in >> label;
	assert(strcmp(label, "<CHMM>") == 0);

	hmm.Dispose();

	in >> label >> hmm.m_stateNum >> label; // "<StateNum>"

	hmm.Allocate(hmm.m_stateNum);

	for ( i = 0; i < hmm.m_stateNum; i++)
	{
		in >> *hmm.m_stateModel[i];
	}

	in >> label; // "<Init>"
	for ( i = 0; i < hmm.m_stateNum; i++)
	{
		in >> hmm.m_stateInit[i];
	}
	in >> label;

	in >> label; // "<Tran>"
	for ( i = 0; i < hmm.m_stateNum; i++)
	{
		for ( j = 0; j < hmm.m_stateNum + 1; j++)
		{
			in >> hmm.m_stateTran[i][j];
		}
	}
	in >> label;

	in >> label; // "</CHMM>"

	return in;
}

void CHMM::TextTransform(const char* InputText, const char * OutputBinaryText)
{
	ifstream Input(InputText);
	ofstream Output(OutputBinaryText,ios_base::binary);
	int seq_num = 0;  //总序列长度，int型
	int dim = 0;      //特征维数，int型
    int seq_size = 0; //各个序列包含的特征数，int型

	Input>>seq_num;
	Input>>dim;
	Output.write((char*)&seq_num,sizeof(int));
	Output.write((char*)&dim,sizeof(int));

    double *pt_feature;
	pt_feature = new double[dim];   //别忘了释放内存！！！

	for(int i = 0; i < seq_num; i++)
	{
		Input>>seq_size;
        Output.write((char*)&seq_size,sizeof(int));
		for(int j = 0; j < seq_size; j++)
		{
		    for(int k = 0; k < dim; k++)
			{
			    Input>>pt_feature[k];
			}
            for(int t = 0; t < seq_size; t++)
			{
			    Output<<pt_feature[t];
				pt_feature[t] = 0;
			}
		}
	}

    delete []pt_feature;  //勿忘我！！！
}
