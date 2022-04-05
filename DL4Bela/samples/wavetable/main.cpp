/*
 * default_main.cpp
 *
 *  Created on: Oct 24, 2014
 *      Author: parallels
 */

#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <libgen.h>
#include <signal.h>
#include <getopt.h>
#include <string.h>
#include <functional>
#include <Bela.h>

#include "AppOptions.h"

// Handle Ctrl-C by requesting that the audio rendering stop
void interrupt_handler(int var)
{
	gShouldStop = true;
}

// Print usage information
void usage(const char *processName)
{
	std::cerr << "Usage: " << processName << " [options]" << std::endl;

	Bela_usage();

	std::cerr << "   --onnx [-n] filename:               Name of the file to load\n";
	std::cerr << "   --useArmnn [-a]					  					Use ArmmNN Delegate\n";
	std::cerr << "   --help [-h]:                        Print this menu\n";
}

int main(int argc, char *argv[])
{

	// ////////////////
	BelaInitSettings *settings = Bela_InitSettings_alloc(); // Standard audio settings

	struct option customOptions[] =
			{
					{"help", 0, NULL, 'h'},
					{"model", 1, NULL, 'm'},
					{"frontend", 0, NULL, 'f'},
					{NULL, 0, NULL, 0}};

	// Set default settings
	Bela_defaultSettings(settings);
	settings->setup = setup;
	settings->render = render;
	settings->cleanup = cleanup;
	if (argc > 0 && argv[0])
	{
		settings->projectName = strrchr(argv[0], '/') + 1;
	}

	AppOptions opts;
	while (1)
	{
		int c = Bela_getopt_long(argc, argv, "hm:f:", customOptions, settings);
		if (c < 0)
		{
			break;
		}
		int ret = -1;
		switch (c)
		{
		case 'h':
			usage(basename(argv[0]));
			ret = 0;
			break;
		case 'm':
			opts.modelPath = (char *)optarg;
			break;
		case 'f':
			opts.frontend = (int)optarg;
			break;
		default:
			usage(basename(argv[0]));
			ret = 1;
			break;
		}
		if (ret >= 0)
		{
			Bela_InitSettings_free(settings);
			return ret;
		}
	}

	// Initialise the PRU audio device
	if (Bela_initAudio(settings, &opts, 48000.0, 3.0) != 0)
	{
		Bela_InitSettings_free(settings);
		fprintf(stderr, "Error: unable to initialise audio\n");
		return 1;
	}
	Bela_InitSettings_free(settings);

	// Start the audio device running
	if (Bela_startAudio())
	{
		fprintf(stderr, "Error: unable to start real-time audio\n");
		// Stop the audio device
		Bela_stopAudio();
		// Clean up any resources allocated for audio
		Bela_cleanupAudio();
		return 1;
	}

	// Set up interrupt handler to catch Control-C and SIGTERM
	signal(SIGINT, interrupt_handler);
	signal(SIGTERM, interrupt_handler);

	// Run until told to stop
	while (!gShouldStop)
	{
		usleep(100000);
	}

	// Stop the audio device
	Bela_stopAudio();

	// Clean up any resources allocated for audio
	Bela_cleanupAudio();

	// All done!
	return 0;
}
