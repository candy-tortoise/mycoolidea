#include "file.cuh"

#include <openfhe.h>

#include <CKKS/Bootstrap.cuh>
#include <CKKS/BootstrapPrecomputation.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/Parameters.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <FIDESlib/CKKS/Context.cuh>
#include <FIDESlib/CKKS/RNSPoly.cuh>
#include <FIDESlib/CKKS/openfhe-interface/RawCiphertext.cuh>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>

using lbcrypto::Plaintext;
using Ciphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

bool useSLOTHE = false;
constexpr int log_ring = 1 << 17;
constexpr int slot_count = 1 << 15;
constexpr int digits_hks = 2;
constexpr int dcrtBits = 50;
constexpr int firstMod = 52;
constexpr uint32_t levelsUsedBeforeBootstrap = 12;
std::chrono::high_resolution_clock::time_point start;
std::chrono::high_resolution_clock::time_point stop;
std::chrono::seconds duration;

FIDESlib::CKKS::Parameters params;
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context = nullptr;

FIDESlib::CKKS::Ciphertext move_ciphertext(FIDESlib::CKKS::Context& cc_gpu,
                                           const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct);
std::vector<FIDESlib::CKKS::Ciphertext> move_ciphertext(
    FIDESlib::CKKS::Context& cc_gpu, const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& cts);

int main(int argc, char* argv[])
{
	std::string trainingFile, testingFile;
	Vector degree3(3);
	int choice = 0, EpochCount = 0;

	std::cout << "Select a dataset:\n";
	std::cout << "1. MNIST dataset\n";
	std::cout << "2. Credit dataset\n";
	std::cin >> choice;

	while (true)
	{
		if (choice == 1)
		{
			std::cout << "You selected the MNIST dataset.\n";
			trainingFile = "MNIST_train.csv";
			testingFile = "MNIST_test.csv";

			degree3.at(0) = 0.5;
			degree3.at(1) = -0.0843;
			degree3.at(2) = 0.0002;

			break;
		}
		else if (choice == 2)
		{
			std::cout << "You selected the Credit dataset.\n";
			trainingFile = "Credit_train.csv";
			testingFile = "Credit_test.csv";

			if (useSLOTHE)
			{
				degree3.at(0) = 0.5;
				degree3.at(1) = 0.15583087751130215;
				degree3.at(2) = -0.0016336772377525508;
			}
			else
			{
				degree3.at(0) = 0.5;
				degree3.at(1) = -0.15012;
				degree3.at(2) = 0.001593;
			}

			break;
		}
		else
		{
			std::cout << "Invalid choice. Please try again.\n";
			return -1;
		}
	}

	std::cout << "Enter the number of epochs to trian: ";
	while (true)
	{
		std::cin >> EpochCount;

		if (std::cin.fail())
		{
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::cout << "Invalid input. Please enter a positive integer: ";
		}
		else
		{
			break;
		}
	}

	double auc = 0.0, acc = 0.0;
	int batch = 1 << 5;
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;

	parameters.SetSecretKeyDist(lbcrypto::SPARSE_TERNARY);
	parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
	parameters.SetNumLargeDigits(digits_hks);
	parameters.SetRingDim(log_ring);
	parameters.SetBatchSize(slot_count);

	parameters.SetScalingModSize(dcrtBits);
	parameters.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);
	parameters.SetFirstModSize(firstMod);

	uint32_t circuit_depth =
	    levelsUsedBeforeBootstrap + lbcrypto::FHECKKSRNS::GetBootstrapDepth({4, 4}, lbcrypto::SPARSE_TERNARY);

	parameters.SetMultiplicativeDepth(circuit_depth);

	context = GenCryptoContext(parameters);

	std::cout << "Context built, generating keys...\n";

	context->Enable(lbcrypto::PKE);
	context->Enable(lbcrypto::KEYSWITCH);
	context->Enable(lbcrypto::LEVELEDSHE);
	context->Enable(lbcrypto::ADVANCEDSHE);
	context->Enable(lbcrypto::FHE);

	auto key_pair = context->KeyGen();
	context->EvalMultKeyGen(key_pair.secretKey);

	context->EvalBootstrapSetup({4, 4}, {0, 0}, batch);
	context->EvalBootstrapKeyGen(key_pair.secretKey, batch);
	context->EvalBootstrapPrecompute(batch);

	std::cout << "Generated.\n";

	std::filesystem::path dataPath = AbsolutePath().parent_path() / ".." / "data";
	std::filesystem::path trainFile = dataPath / trainingFile;
	std::filesystem::path testFile = dataPath / testingFile;

	int trainFactorDim = 0, trainSampleDim = 0, testFactorDim = 0, testSampleDim = 0;
	Matrix trainDataSet, testDataSet;
	Vector trainDataLabel, testDataLabel;

	start = std::chrono::high_resolution_clock::now();
	Matrix zData = DataFromFile(trainFile, trainFactorDim, trainSampleDim, trainDataSet, trainDataLabel);
	Matrix zDate = DataFromFile(testFile, testFactorDim, testSampleDim, testDataSet, testDataLabel);
	NormalizeZData(trainDataSet, trainFactorDim, trainSampleDim);
	NormalizeZData(testDataSet, testFactorDim, testSampleDim);

	Matrix trainData(trainSampleDim), testData;
	Vector trainLabel(trainSampleDim), testLabel;

	testData = testDataSet;
	testLabel = testDataLabel;

	for (int i = 0; i < trainSampleDim; ++i)
	{
		trainData[i] = trainDataSet[i];
		trainLabel[i] = trainDataLabel[i];
	}

	int minbatchsize = slot_count / batch;
	int sBits = std::ceil(std::log2(slot_count));
	int fdimBits = std::ceil(std::log2(trainFactorDim));
	int bBits = std::ceil(std::log2(batch));
	int cnum = std::ceil(double(trainFactorDim) / batch);
	int rnum = std::ceil((double)trainSampleDim / minbatchsize);
	int lognslots = std::ceil(std::log2(batch));

	assert(cnum <= (1 << fdimBits));

	std::vector<int> shift_key;
	for (int l = 0; l < bBits; ++l)
	{
		shift_key.push_back(1 << l);
		shift_key.push_back(-(1 << l));
	}
	for (int l = bBits; l < sBits; ++l)
	{
		shift_key.push_back(1 << l);
	}

	std::sort(shift_key.begin(), shift_key.end());
	auto it = std::unique(shift_key.begin(), shift_key.end());
	shift_key.erase(it, shift_key.end());

	context->EvalRotateKeyGen(key_pair.secretKey, shift_key);

	Vector cwData(trainFactorDim);
	Vector cvData(trainFactorDim);

	std::vector<Ciphertext> encXyZdata(rnum * cnum);
	std::vector<Ciphertext> encBinv(rnum * cnum);

	std::vector<Ciphertext> encWData(cnum);
	std::vector<Ciphertext> encVData(cnum);

	Plaintext tempPlaintext;

	std::cout << "Setting up GPU parameters, context, etc...\n";
	auto raw_params = FIDESlib::CKKS::GetRawParams(context);
	auto adapted_params = params.adaptTo(raw_params);
	FIDESlib::CKKS::Context cc_gpu(adapted_params, {0});

	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(key_pair);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_gpu);
	eval_key_gpu.Initialize(cc_gpu, eval_key);
	FIDESlib::CKKS::Context::AddEvalKey(std::move(eval_key_gpu));
	FIDESlib::CKKS::AddBootstrapPrecomputation(context, key_pair, batch, cc_gpu);
	std::cout << "Successfully generated!\n";

	// - - - - - - - - - - - - - - - - - - - - Client and Server - - - - - - - - - - - - - - - - - - - -

	std::cout << "Encrypting training data...\n";
	for (int r = 0; r < rnum - 1; ++r)
	{
		for (int i = 0; i < cnum - 1; ++i)
		{
			Vector pzData(slot_count, 0.0);
			for (int j = 0; j < minbatchsize; ++j)
			{
				for (int l = 0; l < batch; ++l)
				{
					pzData[batch * j + l] = trainLabel[r * minbatchsize + j] *
					                        trainData[r * minbatchsize + j][batch * i + l];
				}
			}

			tempPlaintext = context->MakeCKKSPackedPlaintext(pzData);
			encXyZdata[r * cnum + i] = context->Encrypt(key_pair.publicKey, tempPlaintext);
		}

		Vector pzData(slot_count, 0.0);
		for (int j = 0; j < minbatchsize; ++j)
		{
			int rest = trainFactorDim - batch * (cnum - 1);
			for (int l = 0; l < rest; ++l)
			{
				pzData[batch * j + l] = trainLabel[r * minbatchsize + j] *
				                        trainData[r * minbatchsize + j][batch * (cnum - 1) + l];
			}
		}

		tempPlaintext = context->MakeCKKSPackedPlaintext(pzData);
		encXyZdata[r * cnum + cnum - 1] = context->Encrypt(key_pair.publicKey, tempPlaintext);
	}
	std::cout << "Success!\n";

	int restrownum = trainSampleDim - minbatchsize * (rnum - 1);
	for (int i = 0; i < cnum - 1; ++i)
	{
		Vector pzData(slot_count, 0.0);
		for (int j = 0; j < restrownum; ++j)
		{
			for (int l = 0; l < batch; ++l)
			{
				pzData[batch * j + l] = trainLabel[(rnum - 1) * minbatchsize + j] *
				                        trainData[(rnum - 1) * minbatchsize + j][batch * i + l];
			}
		}

		tempPlaintext = context->MakeCKKSPackedPlaintext(pzData);
		encXyZdata[(rnum - 1) * cnum + i] = context->Encrypt(key_pair.publicKey, tempPlaintext);
	}

	Vector pzDatra(slot_count, 0.0);
	for (int j = 0; j < restrownum; ++j)
	{
		int rest = trainFactorDim - batch * (cnum - 1);
		for (int l = 0; l < rest; ++l)
		{
			pzDatra[batch * j + l] = trainLabel[(rnum - 1) * minbatchsize + j] *
			                         trainData[(rnum - 1) * minbatchsize + j][batch * (cnum - 1) + l];
		}
	}

	tempPlaintext = context->MakeCKKSPackedPlaintext(pzDatra);
	encXyZdata[(rnum - 1) * cnum + cnum - 1] = context->Encrypt(key_pair.publicKey, tempPlaintext);

	// -------------------------------------------------------------

	std::cout << "Computing Hessian...\n";
	Matrix Binv = zInvBFromFile(trainData, trainFactorDim, trainSampleDim);
	for (int r = 0; r < rnum - 1; ++r)
	{
		Matrix zInvB(minbatchsize);
		for (int i = 0; i < minbatchsize; ++i)
		{
			Vector Bj(trainFactorDim);
			for (int j = 0; j < trainFactorDim; ++j)
			{
				Bj[j] = trainData[r * minbatchsize + i][j];
			}
			zInvB[i] = Bj;
		}

		Matrix zTemp = zInvBFromFile(zInvB, trainFactorDim, minbatchsize);

		for (int i = 0; i < minbatchsize; ++i)
		{
			for (int j = 0; j < trainFactorDim; ++j)
			{
				Binv[r * minbatchsize + i][j] = std::min(zTemp[i][j], 8.);
			}
		}

		zInvB.clear();
		zTemp.clear();
	}

	int restrows = trainSampleDim - minbatchsize * (rnum - 1);
	Matrix zInvB(restrows);
	for (int i = 0; i < restrows; ++i)
	{
		Vector Bj(trainFactorDim);
		for (int j = 0; j < trainFactorDim; ++j)
		{
			Bj[j] = trainData[(rnum - 1) * minbatchsize + i][j];
		}
		zInvB[i] = Bj;
	}

	auto zTemp = zInvBFromFile(zInvB, trainFactorDim, restrows);

	for (int i = 0; i < restrows; ++i)
	{
		for (int j = 0; j < trainFactorDim; ++j)
		{
			Binv[(rnum - 1) * minbatchsize + i][j] = std::min(zTemp[i][j], 8.);
		}
	}

	zInvB.clear();
	zTemp.clear();

	std::cout << "Encrypting inv...\n";
	for (int r = 0; r < rnum - 1; ++r)
	{
		for (int i = 0; i < cnum - 1; ++i)
		{
			Vector pzData(slot_count, 0.0);
			for (int j = 0; j < minbatchsize; ++j)
			{
				for (int l = 0; l < batch; ++l)
				{
					pzData[batch * j + l] = Binv[r * minbatchsize + j][batch * i + l];
				}
			}

			tempPlaintext = context->MakeCKKSPackedPlaintext(pzData);
			encBinv[r * cnum + i] = context->Encrypt(key_pair.publicKey, tempPlaintext);
		}

		Vector pzData3(slot_count, 0.0);
		for (int j = 0; j < minbatchsize; ++j)
		{
			int rest = trainFactorDim - batch * (cnum - 1);
			for (int l = 0; l < rest; ++l)
			{
				pzData3[batch * j + l] = Binv[r * minbatchsize + j][batch * (cnum - 1) + l];
			}
		}

		tempPlaintext = context->MakeCKKSPackedPlaintext(pzData3);
		encBinv[r * cnum + cnum - 1] = context->Encrypt(key_pair.publicKey, tempPlaintext);
	}

	restrownum = trainSampleDim - minbatchsize * (rnum - 1);
	for (int i = 0; i < cnum - 1; ++i)
	{
		Vector pzData(slot_count, 0.0);
		for (int j = 0; j < restrownum; ++j)
		{
			for (int l = 0; l < batch; ++l)
			{
				pzData[batch * j + l] = Binv[(rnum - 1) * minbatchsize + j][batch * i + l];
			}
		}

		tempPlaintext = context->MakeCKKSPackedPlaintext(pzData);
		encBinv[(rnum - 1) * cnum + i] = context->Encrypt(key_pair.publicKey, tempPlaintext);
	}

	Vector pzData7(slot_count, 0.0);
	for (int j = 0; j < restrownum; ++j)
	{
		int rest = trainFactorDim - batch * (cnum - 1);
		for (int l = 0; l < rest; ++l)
		{
			pzData7[batch * j + l] = Binv[(rnum - 1) * minbatchsize + j][batch * (cnum - 1) + l];
		}
	}

	tempPlaintext = context->MakeCKKSPackedPlaintext(pzData7);
	encBinv[(rnum - 1) * cnum + cnum - 1] = context->Encrypt(key_pair.publicKey, tempPlaintext);

	std::cout << "Computing zDataTrain and zDataTest\n";

	// zDataTrain and zDataTest are used for testing in the plaintext environment
	// zData = (Y,Y@X)
	Matrix zDataTrain(trainSampleDim);
	for (int i = 0; i < trainSampleDim; ++i)
	{
		Vector zi(trainFactorDim, 0.0);
		zi[0] = trainLabel[i];
		for (int j = 1; j < trainFactorDim; ++j)
		{
			zi[j] = zi[0] * trainData[i][j];
		}
		zDataTrain[i] = zi;
	}

	// zDataTest is only used for Cross-Validation test, not necessary for training LG model.
	// zData = (Y,Y@X)
	Matrix zDataTest(testSampleDim);
	for (int i = 0; i < testSampleDim; ++i)
	{
		Vector zi(trainFactorDim);
		zi[0] = testLabel[i];
		for (int j = 1; j < trainFactorDim; ++j)
		{
			zi[j] = zi[0] * testData[i][j];
		}
		zDataTest[i] = zi;
	}

	std::cout << "Encrypting wData and vData...\n";
	for (int i = 0; i < cnum; ++i)
	{
		Vector data(slot_count, 0.0);
		tempPlaintext = context->MakeCKKSPackedPlaintext(data);
		encWData[i] = context->Encrypt(key_pair.publicKey, tempPlaintext);
		encVData[i] = encWData[i];
	}

	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
	std::cout << "Data encryption took " << duration.count() << " seconds!\n";
	std::cout << "Data encryption completed successfully!\n";

	assert(cnum <= (1 << fdimBits));
	double min_lr = 1.0, max_lr = 2.0, total_steps = EpochCount * rnum;
	double exp_gamma = 2.5, eta = 0.0;
	double alpha0 = 0.01;
	double alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;

	Vector pvals(slot_count, 0.0);
	for (int j = 0; j < slot_count; j += batch)
	{
		pvals[j] = 1.0;
	}

	Vector zero(slot_count, 0.0);
	tempPlaintext = context->MakeCKKSPackedPlaintext(zero);
	Ciphertext zeroes = context->Encrypt(key_pair.publicKey, tempPlaintext);

	FIDESlib::CKKS::Ciphertext ctBinv_gpu(cc_gpu);
	FIDESlib::CKKS::Ciphertext ctmpw_gpu(cc_gpu);
	FIDESlib::CKKS::Ciphertext test(cc_gpu);
	FIDESlib::CKKS::Ciphertext encIP_gpu(cc_gpu);
	FIDESlib::CKKS::Ciphertext encIP2_gpu(cc_gpu);
	FIDESlib::CKKS::Ciphertext rot_gpu(cc_gpu);
	FIDESlib::CKKS::Ciphertext tmp_gpu(cc_gpu);
	FIDESlib::CKKS::Ciphertext ctIP_gpu(cc_gpu);
	FIDESlib::CKKS::Ciphertext ctIP2_gpu(cc_gpu);

	std::vector<FIDESlib::CKKS::Ciphertext> encIPvec_gpu;
	std::vector<FIDESlib::CKKS::Ciphertext> encGrad_gpu;
	std::vector<FIDESlib::CKKS::Ciphertext> encZData_gpu;
	for (int i = 0; i < cnum; ++i)
	{
		encIPvec_gpu.push_back(move_ciphertext(cc_gpu, zeroes));
		encGrad_gpu.push_back(move_ciphertext(cc_gpu, zeroes));
		encZData_gpu.push_back(move_ciphertext(cc_gpu, zeroes));
	}
	assert(encIPvec_gpu.size() == cnum);
	assert(encGrad_gpu.size() == cnum);
	assert(encZData_gpu.size() == cnum);

	std::vector<FIDESlib::CKKS::Ciphertext> encXyZdata_gpu;
	std::vector<FIDESlib::CKKS::Ciphertext> encBinv_gpu;
	std::vector<FIDESlib::CKKS::Ciphertext> encWData_gpu;
	std::vector<FIDESlib::CKKS::Ciphertext> encVData_gpu;
	FIDESlib::CKKS::RawCipherText raw_ct;

	std::cout << "Moving ciphertexts to the GPU...\n";
	encXyZdata_gpu = move_ciphertext(cc_gpu, encXyZdata);
	encBinv_gpu = move_ciphertext(cc_gpu, encBinv);
	encWData_gpu = move_ciphertext(cc_gpu, encWData);
	encVData_gpu = move_ciphertext(cc_gpu, encVData);
	std::cout << "Success!\n";

	FIDESlib::CKKS::Plaintext tempPlaintextGPU(cc_gpu);

	auto fullStart = std::chrono::high_resolution_clock::now();
	for (int iter = 0; iter < EpochCount; ++iter)
	{
		std::cout << "NesterovWithG : " + std::to_string(iter + 1) + "-th iteration\n";
		start = std::chrono::high_resolution_clock::now();
		for (int r = 0; r < rnum; ++r)
		{
			eta = (1 - alpha0) / alpha1;

			int iterations = iter * rnum + r;
			double gamma = max_lr - (max_lr - min_lr) * std::pow(iterations / total_steps, exp_gamma);

			for (int i = 0; i < cnum; ++i)
			{
				encZData_gpu.at(i).copy(encXyZdata_gpu.at(r * cnum + i));
			}

			// For each batch, sum itself
			for (int i = 0; i < cnum; ++i)
			{
				encIPvec_gpu.at(i).copy(encZData_gpu.at(i));
				encIPvec_gpu.at(i).mult(encVData_gpu.at(i), FIDESlib::CKKS::Context::GetEvalKey());
			}

			encIP_gpu.copy(encIPvec_gpu.at(0));

			// Sum all batches
			for (int i = 1; i < cnum; ++i)
			{
				encIP_gpu.add(encIP_gpu, encIPvec_gpu.at(i));
			}

			for (int l = 0; l < bBits; ++l)
			{
				FIDESlib::CKKS::KeySwitchingKey kskEval(cc_gpu);
				auto rawKskEval = FIDESlib::CKKS::GetRotationKeySwitchKey(key_pair, (1 << l), context);
				kskEval.Initialize(cc_gpu, rawKskEval);
				rot_gpu.rotate(encIP_gpu, (1 << l), kskEval);
				encIP_gpu.add(rot_gpu);
			}

			// Sum this batch to get the inner product
			tempPlaintext = context->MakeCKKSPackedPlaintext(pvals);
			assert(tempPlaintext != nullptr);
			auto tpg = FIDESlib::CKKS::GetRawPlainText(context, tempPlaintext);
			FIDESlib::CKKS::Plaintext tempPlaintextGPU0(cc_gpu, tpg);

			encIP_gpu.multPt(tempPlaintextGPU0);

			for (int l = 0; l < bBits; ++l)
			{
				FIDESlib::CKKS::KeySwitchingKey kskEval(cc_gpu);
				auto rawKskEval = FIDESlib::CKKS::GetRotationKeySwitchKey(key_pair, -(1 << l), context);
				kskEval.Initialize(cc_gpu, rawKskEval);
				tmp_gpu.rotate(encIP_gpu, -(1 << l), kskEval);
				encIP_gpu.add(tmp_gpu);
			}

			encIP2_gpu.copy(encIP_gpu);
			encIP2_gpu.mult(encIP_gpu, FIDESlib::CKKS::Context::GetEvalKey());

			Vector cnst_vec(slot_count, degree3[1] / degree3[2]);
			tempPlaintext = context->MakeCKKSPackedPlaintext(cnst_vec);
			tpg = FIDESlib::CKKS::GetRawPlainText(context, tempPlaintext);
			FIDESlib::CKKS::Plaintext tempPlaintextGPU(cc_gpu, tpg);

			encIP2_gpu.addPt(tempPlaintextGPU);

			for (int i = 0; i < cnum; ++i)
			{
				Vector gamma3(slot_count);
				tempPlaintext = context->MakeCKKSPackedPlaintext(gamma3);
				auto tpg2 = FIDESlib::CKKS::GetRawPlainText(context, tempPlaintext);
				FIDESlib::CKKS::Plaintext tempPlaintextGPU2(cc_gpu, tpg2);
				encGrad_gpu.at(i).multPt(encZData_gpu.at(i), tempPlaintextGPU2);

				ctIP_gpu.copy(encIP_gpu);
				encGrad_gpu.at(i).mult(ctIP_gpu, FIDESlib::CKKS::Context::GetEvalKey());

				ctIP2_gpu.copy(encIP2_gpu);

				encGrad_gpu.at(i).mult(ctIP2_gpu, FIDESlib::CKKS::Context::GetEvalKey());

				Vector gammaDeg3(slot_count, (gamma)*degree3[0]);
				tempPlaintext = context->MakeCKKSPackedPlaintext(gammaDeg3);
				auto tpg3 = FIDESlib::CKKS::GetRawPlainText(context, tempPlaintext);
				FIDESlib::CKKS::Plaintext tempPlaintextGPU3(cc_gpu, tpg3);

				tmp_gpu.multPt(encZData_gpu.at(i), tempPlaintextGPU3);

				encGrad_gpu.at(i).add(tmp_gpu);
			}

			// Sum Each Column of encGrad[i] To Get the Final gradient : (1 - sigm(yWTx)) * Y.T @ X
			for (int i = 0; i < cnum; ++i)
			{
				for (int l = bBits; l < sBits; ++l)
				{
					FIDESlib::CKKS::KeySwitchingKey kskEval(cc_gpu);
					auto rawKskEval =
					    FIDESlib::CKKS::GetRotationKeySwitchKey(key_pair, (1 << l), context);
					kskEval.Initialize(cc_gpu, rawKskEval);
					tmp_gpu.rotate(encGrad_gpu.at(i), (1 << l), kskEval);

					encGrad_gpu.at(i).add(tmp_gpu);
				}

				ctBinv_gpu.copy(encBinv_gpu.at(r * cnum + i));

				encGrad_gpu.at(i).mult(ctBinv_gpu, FIDESlib::CKKS::Context::GetEvalKey());
			}

			for (int i = 0; i < cnum; ++i)
			{
				ctmpw_gpu.add(encVData_gpu.at(i), encGrad_gpu.at(i));

				Vector _1minusETA(slot_count, 1. - eta);
				tempPlaintext = context->MakeCKKSPackedPlaintext(_1minusETA);

				auto tpg4 = FIDESlib::CKKS::GetRawPlainText(context, tempPlaintext);
				FIDESlib::CKKS::Plaintext tempPlaintextGPU4(cc_gpu, tpg4);
				encVData_gpu.at(i).multPt(ctmpw_gpu, tempPlaintextGPU4);

				Vector ETA(slot_count, eta);
				tempPlaintext = context->MakeCKKSPackedPlaintext(ETA);

				auto tpg5 = FIDESlib::CKKS::GetRawPlainText(context, tempPlaintext);
				FIDESlib::CKKS::Plaintext tempPlaintextGPU5(cc_gpu, tpg5);
				encWData_gpu.at(i).multPt(tempPlaintextGPU5);

				encVData_gpu.at(i).add(encWData_gpu.at(i));
				encWData_gpu.at(i).copy(ctmpw_gpu);
			}

			// Testing Accuracy
			for (int i = 0; i < (cnum - 1); ++i)
			{
				encVData_gpu.at(i).store(cc_gpu, raw_ct);
				FIDESlib::CKKS::GetOpenFHECipherText(encVData.at(i), raw_ct);
				context->Decrypt(encVData.at(i), key_pair.secretKey, &tempPlaintext);
				for (int j = 0; j < batch; ++j)
				{
					cvData.at(batch * i + j) = tempPlaintext->GetCKKSPackedValue().at(j).real();
				}
			}
			encVData_gpu.at(cnum - 1).store(cc_gpu, raw_ct);

			FIDESlib::CKKS::GetOpenFHECipherText(encVData.at(cnum - 1), raw_ct);
			context->Decrypt(encVData.at(cnum - 1), key_pair.secretKey, &tempPlaintext);
			int rest = trainFactorDim - batch * (cnum - 1);
			for (int j = 0; j < rest; ++j)
			{
				cvData[batch * (cnum - 1) + j] = tempPlaintext->GetCKKSPackedValue().at(j).real();
			}

			CalculateAUCandACC(zDataTrain, cvData, trainFactorDim, trainSampleDim, auc, acc);
			std::cout << "NewTrainACC : " << acc << "\n";
			std::cout << "NewTrainAUC : " << auc << "\n";

			CalculateAUCandACC(zDataTest, cvData, trainFactorDim, testSampleDim, auc, acc);
			std::cout << "NewTest ACC : " << acc << "\n";
			std::cout << "NewTestAUC : " << auc << "\n";

			std::cout << "\n---------- TEST : THE " << iter * rnum + r + 1
			          << "-th ITERATION-- -- -- -- --\n";

			if (encVData[0]->GetLevel() > 10)
			{
				auto bootStart = std::chrono::high_resolution_clock::now();
				std::cout << "Beginning bootstrapping...\n";

				for (size_t z = 0; z < encVData.size(); ++z)
				{
					FIDESlib::CKKS::Bootstrap(encVData_gpu.at(z), batch);
				}
				for (size_t z = 0; z < encWData.size(); ++z)
				{
					FIDESlib::CKKS::Bootstrap(encWData_gpu.at(z), batch);
				}

				auto bootStop = std::chrono::high_resolution_clock::now();
				auto bootDur = std::chrono::duration_cast<std::chrono::seconds>(bootStop - bootStart);
				std::cout << "Bootstrapping took " << bootDur.count() << " seconds\n";
				std::cout << "Finished bootstrapping!\n";
			}

			alpha0 = alpha1;
			alpha1 = (1. + std::sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;

			std::cout << "End of Iteration " << iter + 1 << "\n\n ";
		}

		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
		std::cout << "Iteration took " << duration.count() << " seconds\n";
	}

	auto fullStop = std::chrono::high_resolution_clock::now();
	auto fullDuration = std::chrono::duration_cast<std::chrono::seconds>(fullStop - fullStart);
	std::cout << "Total runtime was " << fullDuration.count() << " seconds\n";

	return 0;
}

FIDESlib::CKKS::Ciphertext move_ciphertext(FIDESlib::CKKS::Context& cc_gpu,
                                           const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct)
{
	const FIDESlib::CKKS::RawCipherText raw_ct = FIDESlib::CKKS::GetRawCipherText(context, ct);
	FIDESlib::CKKS::Ciphertext ct_gpu(cc_gpu, raw_ct);
	return ct_gpu;
}

std::vector<FIDESlib::CKKS::Ciphertext> move_ciphertext(
    FIDESlib::CKKS::Context& cc_gpu, const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& cts)
{
	std::vector<FIDESlib::CKKS::Ciphertext> cts_gpu;
	for (auto& ct : cts)
	{
		cts_gpu.push_back(move_ciphertext(cc_gpu, ct));
	}
	return cts_gpu;
}
