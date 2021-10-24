#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <stdio.h>
#include <string.h>
#include <fstream>

#include <npp.h>
#include <helper_cuda.h>

#define NUMBER_OF_IMAGES 3

    Npp8u  * pInputImageDev[NUMBER_OF_IMAGES];
    Npp8u  * pInputImageHost[NUMBER_OF_IMAGES];
    Npp8u  * pSegmentationScratchBufferDev[NUMBER_OF_IMAGES];
    Npp8u * pSegmentsDev[NUMBER_OF_IMAGES];
    Npp8u * pSegmentsHost[NUMBER_OF_IMAGES];
    Npp32u * pSegmentLabelsOutputBufferDev[NUMBER_OF_IMAGES];
    Npp32u * pSegmentLabelsOutputBufferHost[NUMBER_OF_IMAGES];

void tearDown() // Clean up and tear down
{
    for (int j = 0; j < NUMBER_OF_IMAGES; j++)
    {
        if (pSegmentLabelsOutputBufferDev[j] != 0)
            cudaFree(pSegmentLabelsOutputBufferDev[j]);
        if (pSegmentationScratchBufferDev[j] != 0)
            cudaFree(pSegmentationScratchBufferDev[j]);
        if (pSegmentsDev[j] != 0)
            cudaFree(pSegmentsDev[j]);
        if (pInputImageDev[j] != 0)
            cudaFree(pInputImageDev[j]);
        if (pSegmentLabelsOutputBufferHost[j] != 0)
            free(pSegmentLabelsOutputBufferHost[j]);
        if (pSegmentsHost[j] != 0)
            free(pSegmentsHost[j]);
        if (pInputImageHost[j] != 0)
            free(pInputImageHost[j]);
    }
}

const std::string& SegmentsOutputFile0 = "Lena_Segments_8Way_512x512_8u.raw";
const std::string& SegmentsOutputFile1 = "CT_skull_Segments_8Way_512x512_8u.raw";
const std::string& SegmentsOutputFile2 = "Rocks_Segments_8Way_512x512_8u.raw";

const std::string& SegmentBoundariesOutputFile0 = "Lena_SegmentBoundaries_8Way_512x512_8u.raw";
const std::string& SegmentBoundariesOutputFile1 = "CT_skull_SegmentBoundaries_8Way_512x512_8u.raw";
const std::string& SegmentBoundariesOutputFile2 = "Rocks_SegmentBoundaries_8Way_512x512_8u.raw";

const std::string& SegmentsWithContrastingBoundariesOutputFile0 = "Lena_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw";
const std::string& SegmentsWithContrastingBoundariesOutputFile1 = "CT_skull_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw";
const std::string& SegmentsWithContrastingBoundariesOutputFile2 = "Rocks_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw";

const std::string& CompressedSegmentLabelsOutputFile0 = "Lena_CompressedSegmentLabels_8Way_512x512_32u.raw";
const std::string& CompressedSegmentLabelsOutputFile1 = "CT_skull_CompressedSegmentLabels_8Way_512x512_32u.raw";
const std::string& CompressedSegmentLabelsOutputFile2 = "Rocks_CompressedSegmentLabels_8Way_512x512_32u.raw";

void
savePBMHeader(FILE* file, int width, int height, bool i32)
{
    fprintf(file, "P5\n");
    if (i32)
    {
	fprintf(file, "# PixelLength: 32\n");
	fprintf(file, "# DataType: Int\n");
    }
    fprintf(file, "%d %d\n255\n", width, height);
}

int
loadRaw8BitImage(Npp8u * pImage, int nWidth, int nHeight, int nImage)
{
    FILE * bmpFile;
    size_t nSize;

    if (nImage == 0)
    {
        if (nWidth != 512 || nHeight != 512)
            return -1;
        const char* fileName = "Lena_512x512_8u_Gray.raw";
        const char* InputFile = sdkFindFilePath(fileName, ".");
        if (InputFile == NULL)
        {
          printf("%s file not found.. exiting\n", fileName);
          exit(EXIT_WAIVED);
        }

        FOPEN(bmpFile, InputFile, "rb");
    }
    else if (nImage == 1)
    {
        if (nWidth != 512 || nHeight != 512)
            return -1;
        const char* fileName = "CT_skull_512x512_8u_Gray.raw";
        const char* InputFile = sdkFindFilePath(fileName, ".");
        if (InputFile == NULL)
        {
          printf("%s file not found.. exiting\n", fileName);
          exit(EXIT_WAIVED);
        }

        FOPEN(bmpFile, InputFile, "rb");
    }
    else if (nImage == 2)
    {
        if (nWidth != 512 || nHeight != 512)
            return -1;
        const char* fileName = "Rocks_512x512_8u_Gray.raw";
        const char* InputFile = sdkFindFilePath(fileName, ".");
        if (InputFile == NULL)
        {
          printf("%s file not found.. exiting\n", fileName);
          exit(EXIT_WAIVED);
        }

        FOPEN(bmpFile, InputFile, "rb");
    }
    else
    {
        printf ("Input file load failed.\n");
        return -1;
    }

    if (bmpFile == NULL)
    {
        printf ("Input file load failed.\n");
        return -1;
    }
    nSize = fread(pImage, 1, nWidth * nHeight, bmpFile);
    if (nSize < nWidth * nHeight)
    {
        printf ("Input file load failed.\n");
        fclose(bmpFile);
        return -1;
    }
    fclose(bmpFile);

    printf ("Input file load succeeded.\n");

    return 0;
}

int
main( int argc, char** argv )
{

    int      aSegmentationScratchBufferSize[NUMBER_OF_IMAGES];
    int      aSegmentLabelsOutputBufferSize[NUMBER_OF_IMAGES];

    cudaError_t cudaError;
    NppStatus nppStatus;
    NppStreamContext nppStreamCtx;
    FILE * bmpFile;
    NppiNorm eNorm = nppiNormInf; // default to 8 way neighbor search

    for (int j = 0; j < NUMBER_OF_IMAGES; j++)
    {
        pInputImageDev[j] = 0;
        pInputImageHost[j] = 0;
        pSegmentationScratchBufferDev[j] = 0;
        pSegmentLabelsOutputBufferDev[j] = 0;
        pSegmentLabelsOutputBufferHost[j] = 0;
        pSegmentsDev[j] = 0;
        pSegmentsHost[j] = 0;
    }

    nppStreamCtx.hStream = 0; // The NULL stream by default, set this to whatever your stream ID is if not the NULL stream.

    cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
    {
        printf("CUDA error: no devices supporting CUDA.\n");
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;
    }

    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver  Version: %d.%d\n",
	   driverVersion/1000, (driverVersion%100)/10);
    printf("CUDA Runtime Version: %d.%d\n\n",
	   runtimeVersion/1000, (runtimeVersion%100)/10);

    cudaError = cudaDeviceGetAttribute(
			&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
			cudaDevAttrComputeCapabilityMajor,
			nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

    cudaError = cudaDeviceGetAttribute(
			&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
			cudaDevAttrComputeCapabilityMinor,
			nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

    cudaError = cudaStreamGetFlags(nppStreamCtx.hStream,
				   &nppStreamCtx.nStreamFlags);

    cudaDeviceProp oDeviceProperties;

    cudaError = cudaGetDeviceProperties(&oDeviceProperties,
					nppStreamCtx.nCudaDeviceId);

    nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor
	= oDeviceProperties.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;

    NppiSize oSizeROI[NUMBER_OF_IMAGES];

    for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++)
    {
        if (nImage == 0)
        {
            oSizeROI[nImage].width = 512;
            oSizeROI[nImage].height = 512;
        }
        else if (nImage == 1)
        {
            oSizeROI[nImage].width = 512;
            oSizeROI[nImage].height = 512;
        }
        else if (nImage == 2)
        {
            oSizeROI[nImage].width = 512;
            oSizeROI[nImage].height = 512;
        }

        // cudaMallocPitch OR cudaMalloc can be used here, in this sample case width == pitch.

        cudaError = cudaMalloc(
			(void**)&pInputImageDev[nImage],
			oSizeROI[nImage].width * sizeof(Npp8u) *
			oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc (
			(void**)&pSegmentsDev[nImage],
			oSizeROI[nImage].width * sizeof(Npp32u) *
			oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pInputImageHost[nImage] =
	    reinterpret_cast<Npp8u *>(
		malloc(oSizeROI[nImage].width * sizeof(Npp8u) *
		       oSizeROI[nImage].height));
        pSegmentsHost[nImage] =
	    reinterpret_cast<Npp8u *>(
		malloc(oSizeROI[nImage].width * sizeof(Npp32u) *
		       oSizeROI[nImage].height));

        nppStatus = nppiSegmentWatershedGetBufferSize_8u_C1R(
			oSizeROI[nImage],
			&aSegmentationScratchBufferSize[nImage]);

        cudaError = cudaMalloc((void **)&pSegmentationScratchBufferDev[nImage],
			       aSegmentationScratchBufferSize[nImage]);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        // Output label marker buffers are only needed if you want to same the generated segmentation labels, they ARE compatible with NPP UF generated labels.
        // Requesting segmentation output may slightly decrease segmentation function performance.  Regardless of the pitch of the segmentation image
        // the segment labels output buffer will have a pitch of oSizeROI[nImage].width * sizeof(Npp32u).

        aSegmentLabelsOutputBufferSize[nImage] =
	    oSizeROI[nImage].width * sizeof(Npp32u) * oSizeROI[nImage].height;

        cudaError = cudaMalloc(
			(void **)&pSegmentLabelsOutputBufferDev[nImage],
			aSegmentLabelsOutputBufferSize[nImage]);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pSegmentLabelsOutputBufferHost[nImage] =
	    reinterpret_cast<Npp32u *>(
		malloc(oSizeROI[nImage].width * sizeof(Npp32u) *
		       oSizeROI[nImage].height));

        if (loadRaw8BitImage(pInputImageHost[nImage],
			     oSizeROI[nImage].width * sizeof(Npp8u),
			     oSizeROI[nImage].height, nImage) == 0)
        {
            cudaError = cudaMemcpy2DAsync(
				pInputImageDev[nImage],
				oSizeROI[nImage].width * sizeof(Npp8u),
				pInputImageHost[nImage],
				oSizeROI[nImage].width * sizeof(Npp8u),
				oSizeROI[nImage].width * sizeof(Npp8u),
				oSizeROI[nImage].height,
				cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            // Make a second copy of the unaltered input image since this function works in place and we want to reuse the input image multiple times.
            cudaError = cudaMemcpy2DAsync(
				pSegmentsDev[nImage],
				oSizeROI[nImage].width * sizeof(Npp8u),
				pInputImageHost[nImage],
				oSizeROI[nImage].width * sizeof(Npp8u),
				oSizeROI[nImage].width * sizeof(Npp8u),
				oSizeROI[nImage].height,
				cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            nppStatus = nppiSegmentWatershed_8u_C1IR_Ctx(
				pSegmentsDev[nImage],
				oSizeROI[nImage].width * sizeof(Npp8u),
				pSegmentLabelsOutputBufferDev[nImage],
				oSizeROI[nImage].width * sizeof(Npp32u),
				eNorm, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE,
				oSizeROI[nImage],
				pSegmentationScratchBufferDev[nImage],
				nppStreamCtx);

            if (nppStatus != NPP_SUCCESS)
            {
                if (nImage == 0)
                    printf("Lena segments 8Way 512x512 8u failed.\n");
                else if (nImage == 1)
                    printf("CT skull segments 8Way 512x512 8u failed.\n");
                else if (nImage == 2)
                    printf("Rocks segments 8Way 512x512 8u failed.\n");
                tearDown();
                return -1;
            }

            // Now compress the label markers output to make them easier to view.
            int nCompressedLabelsScratchBufferSize;
            Npp8u * pCompressedLabelsScratchBufferDev;

            nppStatus = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
			    oSizeROI[nImage].width * oSizeROI[nImage].height,
			    &nCompressedLabelsScratchBufferSize);
            if (nppStatus != NPP_NO_ERROR)
                return nppStatus;

            cudaError = cudaMalloc((void **)&pCompressedLabelsScratchBufferDev,
				   nCompressedLabelsScratchBufferSize);
            if (cudaError != cudaSuccess)
                return NPP_MEMORY_ALLOCATION_ERR;

            int nCompressedLabelCount = 0;

            nppStatus = nppiCompressMarkerLabelsUF_32u_C1IR(
			    pSegmentLabelsOutputBufferDev[nImage],
			    oSizeROI[nImage].width * sizeof(Npp32u),
			    oSizeROI[nImage],
			    oSizeROI[nImage].width * oSizeROI[nImage].height,
			    &nCompressedLabelCount,
			    pCompressedLabelsScratchBufferDev);

            if (nppStatus != NPP_SUCCESS)
            {
                if (nImage == 0)
                    printf("Lena_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
                else if (nImage == 1)
                    printf("CT_Skull_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
                else if (nImage == 2)
                    printf("Rocks_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
                tearDown();
                return -1;
            }

            // Copy segmented image to host
            cudaError = cudaMemcpy2DAsync(pSegmentsHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u),
                                          pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Copy segment labels image to host
            cudaError = cudaMemcpy2DAsync(
			    pSegmentLabelsOutputBufferHost[nImage],
			    oSizeROI[nImage].width * sizeof(Npp32u),
			    pSegmentLabelsOutputBufferDev[nImage],
			    oSizeROI[nImage].width * sizeof(Npp32u),
			    oSizeROI[nImage].width * sizeof(Npp32u),
			    oSizeROI[nImage].height,
			    cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) !=
		cudaSuccess)
            {
                printf ("Post segmentation cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }

            // Free single image scratch buffer
            cudaFree(pCompressedLabelsScratchBufferDev);

            // Save default segments file.
            if (nImage == 0)
                FOPEN(bmpFile, SegmentsOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                FOPEN(bmpFile, SegmentsOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                FOPEN(bmpFile, SegmentsOutputFile2.c_str(), "wb");

            if (bmpFile == NULL)
                return -1;

	    savePBMHeader(bmpFile, oSizeROI[nImage].width,
			  oSizeROI[nImage].height, false);
            size_t nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSegmentsHost[nImage][j * oSizeROI[nImage].width],
				sizeof(Npp8u), oSizeROI[nImage].width,
				bmpFile);
            }
            fclose(bmpFile);

            if (nImage == 0)
                printf("Lena_Segments_8Way_512x512_8u succeeded.\n");
            else if (nImage == 1)
                printf("CT_Skull_Segments_8Way_512x512_8u succeeded.\n");
            else if (nImage == 2)
                printf("Rocks_Segments_8Way_512x512_8u succeeded.\n");

            // Save segment labels file.
            if (nImage == 0)
                FOPEN(bmpFile, CompressedSegmentLabelsOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                FOPEN(bmpFile, CompressedSegmentLabelsOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                FOPEN(bmpFile, CompressedSegmentLabelsOutputFile2.c_str(), "wb");

            if (bmpFile == NULL)
                return -1;

	    savePBMHeader(bmpFile, oSizeROI[nImage].width,
			  oSizeROI[nImage].height, true);
            nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSegmentLabelsOutputBufferHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            if (nImage == 0)
                printf("Lena_CompressedSegmentLabels_8Way_512x512_32u succeeded.\n");
            else if (nImage == 1)
                printf("CT_Skull_CompressedSegmentLabels_8Way_512x512_32u succeeded.\n");
            else if (nImage == 2)
                printf("Rocks_CompressedSegmentLabels_8Way_512x512_32u succeeded.\n");

            // Now generate a segment boundaries only output image

            // Make a second copy of the unaltered input image since this function works in place and we want to reuse the input image multiple times.
            cudaError = cudaMemcpy2DAsync(
			    pSegmentsDev[nImage],
			    oSizeROI[nImage].width * sizeof(Npp8u),
			    pInputImageHost[nImage],
			    oSizeROI[nImage].width * sizeof(Npp8u),
			    oSizeROI[nImage].width * sizeof(Npp8u),
			    oSizeROI[nImage].height,
			    cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            // We already generated segment labels images to skip that this time
            nppStatus = nppiSegmentWatershed_8u_C1IR_Ctx(
			    pSegmentsDev[nImage],
			    oSizeROI[nImage].width * sizeof(Npp8u),
			    0, 0, eNorm,
			    NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY,
			    oSizeROI[nImage],
			    pSegmentationScratchBufferDev[nImage],
			    nppStreamCtx);

            if (nppStatus != NPP_SUCCESS)
            {
                if (nImage == 0)
                    printf("Lena segment boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 1)
                    printf("CT skull segment boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 2)
                    printf("Rocks segment boundaries 8Way 512x512 8u failed.\n");
                tearDown();
                return -1;
            }

            // Copy segment boundaries image to host
            cudaError = cudaMemcpy2DAsync(pSegmentsHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u),
                                          pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess)
            {
                printf ("Post segmentation cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }

            if (nImage == 0)
                FOPEN(bmpFile, SegmentBoundariesOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                FOPEN(bmpFile, SegmentBoundariesOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                FOPEN(bmpFile, SegmentBoundariesOutputFile2.c_str(), "wb");

            if (bmpFile == NULL)
                return -1;

	    savePBMHeader(bmpFile, oSizeROI[nImage].width,
			  oSizeROI[nImage].height, false);
            nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSegmentsHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp8u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            if (nImage == 0)
                printf("Lena_SegmentBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 1)
                printf("CT_Skull_SegmentBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 2)
                printf("Rocks_SegmentBoundaries_8Way_512x512_8u succeeded.\n");

            // Now generate a segmented with contrasting boundaries output image

            // Make a second copy of the unaltered input image since this function works in place and we want to reuse the input image multiple times.
            cudaError = cudaMemcpy2DAsync(pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), pInputImageHost[nImage],
                                          oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height,
                                          cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            // We already generated segment labels images to skip that this time
            nppStatus = nppiSegmentWatershed_8u_C1IR_Ctx(pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u),
                                                         0, 0, eNorm,
                                                         NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST, oSizeROI[nImage], pSegmentationScratchBufferDev[nImage], nppStreamCtx);

            if (nppStatus != NPP_SUCCESS)
            {
                if (nImage == 0)
                    printf("Lena segments with contrasting boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 1)
                    printf("CT skull segments with contrasting boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 2)
                    printf("Rocks segments with contrasting boundaries 8Way 512x512 8u failed.\n");
                tearDown();
                return -1;
            }

            // Copy segment boundaries image to host
            cudaError = cudaMemcpy2DAsync(pSegmentsHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u),
                                          pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess)
            {
                printf ("Post segmentation cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }

            if (nImage == 0)
                FOPEN(bmpFile, SegmentsWithContrastingBoundariesOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                FOPEN(bmpFile, SegmentsWithContrastingBoundariesOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                FOPEN(bmpFile, SegmentsWithContrastingBoundariesOutputFile2.c_str(), "wb");

            if (bmpFile == NULL)
                return -1;

	    savePBMHeader(bmpFile, oSizeROI[nImage].width,
			  oSizeROI[nImage].height, false);
            nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSegmentsHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp8u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            if (nImage == 0)
                printf("Lena_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 1)
                printf("CT_Skull_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 2)
                printf("Rocks_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.\n");
        }
    }

    tearDown();

    return 0;
}
