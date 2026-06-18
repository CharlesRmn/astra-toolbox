/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/

/** \file astra_mex_direct_c.cpp
 *
 *  \brief Utility functions for low-overhead FP and BP calls.
 */
#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexCopyDataHelpFunctions.h"
#include "mexDataManagerHelpFunctions.h"

#include <list>
#include <vector>

#include "astra/Globals.h"

#include "astra/AstraObjectManager.h"

#include "astra/CudaProjector3D.h"
#include "astra/Projector3D.h"
#include "astra/Data3D.h"

#include "astra/CompositeGeometryManager.h"

using namespace std;
using namespace astra;


class CDataMemory_simple : public astra::CDataMemory<float> {
public:
	CDataMemory_simple(float *ptr) { m_pfData = ptr; }
	~CDataMemory_simple() { m_pfData = nullptr; }
};

#ifdef ASTRA_CUDA

template<typename Type>
void clean_vector(std::vector<Type *> & vec, const size_t limit)
{
	for (size_t itClean = 0; itClean < limit; itClean++) {
		delete vec[itClean];
	}
}

// Look up and check the projectors referenced by a vector (or scalar) of
// projector ids. Returns false (and raises a mex error) if any of the
// projectors could not be found, is not initialized, or is not CUDA-based.
bool get_projectors(const mxArray * const projs,
                     std::vector<astra::CProjector3D *> & pProjectors)
{
	const double * const dPids = (const double *)mxGetData(projs);
	const size_t num_projectors = pProjectors.size();
	for (size_t itPid = 0; itPid < num_projectors; itPid++) {
		int iPid = (int)(dPids[itPid]);
		astra::CProjector3D * pProjector = astra::CProjector3DManager::getSingleton().get(iPid);
		if (!pProjector) {
			mexErrMsgTxt("One of the projectors was not found.");
			return false;
		}
		if (!pProjector->isInitialized()) {
			mexErrMsgTxt("One of the projectors was not initialized.");
			return false;
		}
		if (!dynamic_cast<astra::CCudaProjector3D*>(pProjector)) {
			mexErrMsgTxt("Only CUDA projectors are currently supported.");
			return false;
		}
		pProjectors[itPid] = pProjector;
	}
	return true;
}

static size_t getDataSize(const astra::CVolumeGeometry3D & geom) {
	return (size_t)geom.getGridColCount() * geom.getGridRowCount() * geom.getGridSliceCount();
}
static size_t getDataSize(const astra::CProjectionGeometry3D & geom) {
	return (size_t)geom.getDetectorColCount() * geom.getProjectionCount() * geom.getDetectorRowCount();
}

static void getOutputDims(const astra::CVolumeGeometry3D & geom, mwSize dims[3]) {
	dims[0] = geom.getGridColCount();
	dims[1] = geom.getGridRowCount();
	dims[2] = geom.getGridSliceCount();
}
static void getOutputDims(const astra::CProjectionGeometry3D & geom, mwSize dims[3]) {
	dims[0] = geom.getDetectorColCount();
	dims[1] = geom.getProjectionCount();
	dims[2] = geom.getDetectorRowCount();
}

template<typename DataType, typename GeomType>
DataType * loadData(const mxArray * const data, const GeomType & geom)
{
	if (!checkDataType(data)) {
		mexErrMsgTxt("Data must be single or double.");
		return nullptr;
	}
	if (!checkDataSize(data, &geom)) {
		mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry.");
		return nullptr;
	}
	if (mxIsSingle(data)) {
		astra::CDataStorage* m = new CDataMemory_simple((float *)mxGetData(data));
		return new DataType(geom, m);
	} else {
		size_t dataSize = getDataSize(geom);
		astra::CDataStorage* pStorage = new astra::CDataMemory<float>(dataSize);
		DataType * pData = new DataType(geom, pStorage);
		copyMexToCFloat32Array(data, pData->getFloat32Memory(), dataSize);
		return pData;
	}
}

// Allocate output data. If the corresponding input is single, the output is
// single too, and pOutputMx is set to the (uninitialized) mxArray backing it
// directly. Otherwise the output is computed into internal storage, and
// pOutputMx is left untouched; it will be created from the result afterwards.
template<typename DataType, typename GeomType>
DataType * allocateOutput(const mxArray * const data, const GeomType & geom,
                           mxArray * & pOutputMx)
{
	if (mxIsSingle(data)) {
		mwSize dims[3];
		getOutputDims(geom, dims);
		const mwSize zero_dims[2] = {0, 0};
		pOutputMx = mxCreateNumericArray(2, zero_dims, mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(pOutputMx, dims, 3);
		const mwSize num_elems = mxGetNumberOfElements(pOutputMx);
		const mwSize elem_size = mxGetElementSize(pOutputMx);
		mxSetData(pOutputMx, mxMalloc(elem_size * num_elems));
		astra::CDataStorage* m = new CDataMemory_simple((float *)mxGetData(pOutputMx));
		return new DataType(geom, m);
	} else {
		pOutputMx = nullptr;
		astra::CDataStorage* pStorage = new astra::CDataMemory<float>(getDataSize(geom));
		return new DataType(geom, pStorage);
	}
}

// Assemble the final output mxArray(s) from the computed output data. If
// useCell is true, a cell array with one entry per projector is returned.
// Otherwise the single output is returned directly.
template<typename DataType>
mxArray * produce_output(std::vector<mxArray *> & pOutputMxs,
                          const std::vector<DataType *> & pOutput,
                          const std::vector<const mxArray *> & data,
                          const bool useCell)
{
	const size_t num_projectors = data.size();
	mxArray * pOutputMx;

	if (useCell) {
		pOutputMx = mxCreateCellMatrix(num_projectors, 1);

		for (size_t itPid = 0; itPid < num_projectors; itPid++) {
			if (!mxIsSingle(data[itPid])) {
				pOutputMxs[itPid] = createEquivMexArray<mxDOUBLE_CLASS>(pOutput[itPid]);
				copyCFloat32ArrayToMex(pOutput[itPid]->getFloat32Memory(), pOutputMxs[itPid]);
			}
			mxSetCell(pOutputMx, itPid, pOutputMxs[itPid]);
		}
	} else if (mxIsSingle(data[0])) {
		pOutputMx = pOutputMxs[0];
	} else {
		pOutputMx = createEquivMexArray<mxDOUBLE_CLASS>(pOutput[0]);
		copyCFloat32ArrayToMex(pOutput[0]->getFloat32Memory(), pOutputMx);
	}

	return pOutputMx;
}

//-----------------------------------------------------------------------------------------
/**
 * projection = astra_mex_direct_c('FP3D', projector_id, volume);
 * Both 'projection' and 'volume' are Matlab arrays.
 *
 * projections = astra_mex_direct_c('FP3D', projector_ids, volumes);
 * 'projector_ids' is a vector of projector ids, and 'volumes' is a cell
 * array with as many entries. 'projections' is a cell array with the
 * corresponding forward projections.
 */
void astra_mex_direct_fp3d(int& nlhs, mxArray* plhs[], int& nrhs, const mxArray* prhs[])
{
	// TODO: Add an optional way of specifying extra options

	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments. Syntax: astra_mex_direct_c('FP3D', projector_id, data);");
	}

	const mxArray * const mxProjectors = prhs[1];
	const mxArray * const mxVolData = prhs[2];

	const size_t num_projectors = mxGetNumberOfElements(mxProjectors);
	if (num_projectors == 0) {
		mexErrMsgTxt("No projectors specified.");
	}
	const bool useCell = mxIsCell(mxVolData);
	if (num_projectors > 1 && (!useCell || mxGetNumberOfElements(mxVolData) != num_projectors)) {
		mexErrMsgTxt("When using multiple projectors, a cell array with an equal number of volumes should be passed.");
	}

	std::vector<astra::CProjector3D *> pProjectors(num_projectors);
	if (!get_projectors(mxProjectors, pProjectors))
		return;

	std::vector<const mxArray *> data(num_projectors);
	for (size_t itPid = 0; itPid < num_projectors; itPid++) {
		data[itPid] = useCell ? mxGetCell(mxVolData, itPid) : mxVolData;
	}

	std::vector<astra::CFloat32VolumeData3D *> pInput(num_projectors);
	for (size_t itPid = 0; itPid < num_projectors; itPid++) {
		pInput[itPid] = loadData<astra::CFloat32VolumeData3D>(data[itPid], pProjectors[itPid]->getVolumeGeometry());
		if (!pInput[itPid]) {
			clean_vector(pInput, itPid);
			return;
		}
	}

	std::vector<astra::CFloat32ProjectionData3D *> pOutput(num_projectors);
	std::vector<mxArray *> pOutputMxs(num_projectors);
	for (size_t itPid = 0; itPid < num_projectors; itPid++) {
		pOutput[itPid] = allocateOutput<astra::CFloat32ProjectionData3D>(data[itPid], pProjectors[itPid]->getProjectionGeometry(), pOutputMxs[itPid]);
	}

	// Perform FP
	astra::CCompositeGeometryManager cgm;
	astra::CCompositeGeometryManager::TJobList jobs;
	for (size_t itPid = 0; itPid < num_projectors; itPid++) {
		jobs.push_back(cgm.createJobFP(pProjectors[itPid], pInput[itPid], pOutput[itPid], astra::CCompositeGeometryManager::MODE_SET));
	}

	if (!cgm.doJobs(jobs)) {
		clean_vector(pOutput, num_projectors);
		clean_vector(pInput, num_projectors);
		mexErrMsgWithAstraLog("Error running FP.");
	}

	plhs[0] = produce_output(pOutputMxs, pOutput, data, useCell);

	clean_vector(pOutput, num_projectors);
	clean_vector(pInput, num_projectors);
}
//-----------------------------------------------------------------------------------------
/**
 * volume = astra_mex_direct_c('BP3D', projector_id, projection);
 * Both 'projection' and 'volume' are Matlab arrays.
 *
 * volumes = astra_mex_direct_c('BP3D', projector_ids, projections);
 * 'projector_ids' is a vector of projector ids, and 'projections' is a cell
 * array with as many entries. 'volumes' is a cell array with the
 * corresponding back projections.
 */
void astra_mex_direct_bp3d(int& nlhs, mxArray* plhs[], int& nrhs, const mxArray* prhs[])
{
	// TODO: Add an optional way of specifying extra options

	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments. Syntax: astra_mex_direct_c('BP3D', projector_id, data);");
	}

	const mxArray * const mxProjectors = prhs[1];
	const mxArray * const mxProjData = prhs[2];

	const size_t num_projectors = mxGetNumberOfElements(mxProjectors);
	if (num_projectors == 0) {
		mexErrMsgTxt("No projectors specified.");
	}
	const bool useCell = mxIsCell(mxProjData);
	if (num_projectors > 1 && (!useCell || mxGetNumberOfElements(mxProjData) != num_projectors)) {
		mexErrMsgTxt("When using multiple projectors, a cell array with an equal number of projection data sets should be passed.");
	}

	std::vector<astra::CProjector3D *> pProjectors(num_projectors);
	if (!get_projectors(mxProjectors, pProjectors))
		return;

	std::vector<const mxArray *> data(num_projectors);
	for (size_t itPid = 0; itPid < num_projectors; itPid++) {
		data[itPid] = useCell ? mxGetCell(mxProjData, itPid) : mxProjData;
	}

	std::vector<astra::CFloat32ProjectionData3D *> pInput(num_projectors);
	for (size_t itPid = 0; itPid < num_projectors; itPid++) {
		pInput[itPid] = loadData<astra::CFloat32ProjectionData3D>(data[itPid], pProjectors[itPid]->getProjectionGeometry());
		if (!pInput[itPid]) {
			clean_vector(pInput, itPid);
			return;
		}
	}

	std::vector<astra::CFloat32VolumeData3D *> pOutput(num_projectors);
	std::vector<mxArray *> pOutputMxs(num_projectors);
	for (size_t itPid = 0; itPid < num_projectors; itPid++) {
		pOutput[itPid] = allocateOutput<astra::CFloat32VolumeData3D>(data[itPid], pProjectors[itPid]->getVolumeGeometry(), pOutputMxs[itPid]);
	}

	// Perform BP
	astra::CCompositeGeometryManager cgm;
	astra::CCompositeGeometryManager::TJobList jobs;
	for (size_t itPid = 0; itPid < num_projectors; itPid++) {
		jobs.push_back(cgm.createJobBP(pProjectors[itPid], pOutput[itPid], pInput[itPid], astra::CCompositeGeometryManager::MODE_SET));
	}

	if (!cgm.doJobs(jobs)) {
		clean_vector(pOutput, num_projectors);
		clean_vector(pInput, num_projectors);
		mexErrMsgWithAstraLog("Error running BP.");
	}

	plhs[0] = produce_output(pOutputMxs, pOutput, data, useCell);

	clean_vector(pOutput, num_projectors);
	clean_vector(pInput, num_projectors);
}

#endif

//-----------------------------------------------------------------------------------------

static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("Valid modes: FP3D, BP3D\n");
}


//-----------------------------------------------------------------------------------------
/**
 * ... = astra_mex_direct_c(mode,...);
 */
void mexFunction(int nlhs, mxArray* plhs[],
				 int nrhs, const mxArray* prhs[])
{

	// INPUT: Mode
	std::string sMode;
	if (1 <= nrhs) {
		sMode = mexToString(prhs[0]);
	} else {
		printHelp();
		return;
	}

#ifndef ASTRA_CUDA
	mexErrMsgTxt("Only CUDA projectors are currently supported.");
#else

	// 3D data
	if (sMode == "FP3D") {
		astra_mex_direct_fp3d(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "BP3D") {
		astra_mex_direct_bp3d(nlhs, plhs, nrhs, prhs);
	} else {
		printHelp();
	}
#endif

	return;
}

