#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

inline bool is_number(const std::string& s)
{
	return (strspn(s.c_str(), "-.0123456789") == s.size());
}

inline std::filesystem::path AbsolutePath()
{
	char path[FILENAME_MAX];
	ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
	return std::filesystem::path(std::string(path, (count > 0) ? count : 0));
}

inline std::filesystem::path ProjectHomePath()
{
	return AbsolutePath().parent_path() / ".." / "..";
}

inline Matrix DataFromFile(const std::filesystem::path& path, int& factorDim, int& sampleDim, Matrix& X, Vector& Y)
{
	X.clear();
	Y.clear();
	Matrix zline;
	factorDim = 1; // dimension of (Y,X)
	sampleDim = 0; // number of samples
	std::ifstream openFile(path);
	if (!openFile.is_open())
	{
		std::cerr << "Error: File '" << path.string() << "' failed to open.\n";
		exit(1);
	}

	std::string line, temp;

	while (getline(openFile, line))
	{
		if (factorDim == 1)
		{
			for (size_t i = 0; i < line.length(); ++i)
			{
				if (line[i] == ',')
				{
					factorDim++;
				}
			}
		}

		size_t pos = line.find_first_of(',', 0);
		std::string first_word = line.substr(0, pos);

		if (!is_number(first_word))
		{
			getline(openFile, line);
		}

		size_t start = 0, end = 0;
		Vector vecline;
		do
		{
			end = line.find_first_of(',', start);
			temp = line.substr(start, end);
			vecline.push_back(atof(temp.c_str()));
			start = end + 1;
		} while (start);
		zline.push_back(vecline);
		sampleDim++;
	}

	Matrix zData(sampleDim); // double** zData = new double*[sampleDim];
	X.resize(sampleDim);     // X = new double*[sampleDim];
	Y.resize(sampleDim);     // Y = new double[sampleDim];

	bool flagY01 = true;
	for (int i = 0; i < sampleDim; ++i)
	{
		if (zline[i][0] == -1)
		{
			flagY01 = false;
			break;
		}
	}

	for (int j = 0; j < sampleDim; ++j)
	{
		Vector zj(factorDim); // double* zj = new double[factorDim];
		Vector xj(factorDim); // double* xj = new double[factorDim];

		Y[j] = zline[j][0];
		if (flagY01)
		{
			zj[0] = 2 * zline[j][0] - 1; // change class label Y{0,1} to Y{-1,+1}
			Y[j] = 2 * Y[j] - 1;
		}

		xj[0] = 1.0;

		for (int i = 1; i < factorDim; ++i)
		{
			zj[i] = zj[0] * zline[j][i];
			xj[i] = zline[j][i];
		}

		zData[j] = zj;
		X[j] = xj;
	}

	return zData;
}

inline void NormalizeZData(Matrix& zData, int factorDim, int sampleDim)
{
	for (int i = 0; i < factorDim; ++i)
	{
		double colmin = zData[0][i];
		double colmax = zData[0][i];

		for (int j = 0; j < sampleDim; ++j)
		{
			colmax = std::max(colmax, zData[j][i]);
			colmin = std::min(colmin, zData[j][i]);
		}

		for (int j = 0; j < sampleDim; ++j)
		{
			if (colmax - colmin < 1e-10)
			{
				if (colmin <= 0)
				{
					zData[j][i] = 0.0;
				}
				if (1 <= colmax)
				{
					zData[j][i] = 1.0;
				}
			}
			else
			{
				zData[j][i] = float(zData[j][i] - colmin) / (colmax - colmin);
			}
		}
	}
}

inline Matrix zInvBFromFile(Matrix zDataTrain, int& factorDim, int& sampleDim, bool isfirst = true,
                            double epsilon = 1e-8)
{
	assert(isfirst);

	Matrix zline;
	Matrix zData(sampleDim);
	Matrix zInvB(sampleDim);

	for (int j = 0; j < sampleDim; ++j)
	{
		Vector zj(factorDim);
		Vector Bj(factorDim);
		for (int i = 0; i < factorDim; ++i)
		{
			zj[i] = std::abs(zDataTrain[j][i]);
			Bj[i] = std::abs(zDataTrain[j][i]);
		}
		zData[j] = zj;
		zInvB[j] = Bj;
	}

	for (int j = 0; j < sampleDim; ++j)
	{
		for (int i = 1; i < factorDim; ++i)
		{
			zInvB[j][0] += zInvB[j][i];
		}
	}

	for (int j = 0; j < sampleDim; ++j)
	{
		for (int i = 1; i < factorDim; ++i)
		{
			zInvB[j][i] = zInvB[j][0];
		}
	}

	for (int j = 0; j < sampleDim; ++j)
	{
		for (int i = 0; i < factorDim; ++i)
		{
			zInvB[j][i] *= zData[j][i];
		}
	}

	for (int i = 0; i < factorDim; ++i)
	{
		for (int j = 1; j < sampleDim; ++j)
		{
			zInvB[0][i] += zInvB[j][i];
		}
	}

	for (int i = 0; i < factorDim; ++i)
	{
		for (int j = 1; j < sampleDim; ++j)
		{
			zInvB[j][i] = zInvB[0][i];
		}
	}

	for (int i = 0; i < factorDim; ++i)
	{
		for (int j = 0; j < sampleDim; ++j)
		{
			zInvB[j][i] = 1.0 / (epsilon + .25 * zInvB[j][i]);
		}
	}

	return zInvB;
}

inline double TrueIP(const Vector& a, size_t a_start, const Vector& b, int b_start, int len)
{
	assert(a_start + len <= a.size());
	assert(b_start + len <= b.size());

	double res = 0.0;
	for (int i = 0; i < len; ++i)
	{
		res += a[a_start + i] * b[b_start + i];
	}
	return res;
}

double CalculateAUCandACC(const Matrix& zData, const Vector& wData, int factorDim, int sampleDim, double& correctness,
                          double& auc)
{
	long TN = 0, FP = 0;
	Vector thetaTN;
	Vector thetaFP;

	for (int i = 0; i < sampleDim; ++i)
	{
		const auto& z = zData.at(i);
		double ip_full = TrueIP(z, 0, wData, 0, factorDim);

		if (z.at(0) > 0)
		{
			if (ip_full < 0)
			{
				TN++;
			}

			double val = z.at(0) * TrueIP(z, 1, wData, 1, factorDim - 1);
			thetaTN.push_back(val);
		}
		else
		{
			if (ip_full < 0)
			{
				FP++;
			}

			double val = z.at(0) * TrueIP(z, 1, wData, 1, factorDim - 1);
			thetaFP.push_back(val);
		}
	}

	correctness = 100.0 - (100.0 * (FP + TN) / sampleDim);
	auc = 0.0;

	if (thetaFP.empty() || thetaTN.empty())
	{
		std::cout << "n_test_yi = 0 : cannot compute AUC\n";
	}
	else
	{
		for (double tn_val : thetaTN)
		{
			for (double fp_val : thetaFP)
			{
				if (fp_val <= tn_val)
				{
					auc++;
				}
			}
		}
		auc /= static_cast<double>(thetaTN.size()) * thetaFP.size();
	}

	return correctness;
}
