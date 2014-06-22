/*
 * Copyright(C) 2014 Pedro H. Penna <pedrohenriquepenna@gmail.com>
 * 
 * slave.h -  Private slave library.
 */

#ifndef _SLAVE_H_
#define _SLAVE_H_

	#include <stdlib.h>
	
	/*
	 * Maximum thread num per cluster.
	 */
	#define MAX_THREADS (16)
	
	/*
	 * Maximum chunk size.
	 */
	#define CHUNK_SIZE (1024)
	
	/*
	 * Maximum mask size.
	 */
	#define MASK_SIZE (50)
	
	/*
	 * Mask radius.
	 */
	#define MASK_RADIUS (3)
	
	/*
	 * Halo size for chunk.
	 */
	#define HALO_SIZE ((2*CHUNK_SIZE*MASK_RADIUS) + (4*MASK_RADIUS*MASK_RADIUS))
	
	/*
	 * Maximum image size.
	 */
	#define IMG_SIZE (32768)
	
	/*
	 * Threshold value between central pixel and neighboor pixel.
	 */
	#define THRESHOLD (20)
	
	/* Type of messages. */
	#define MSG_CHUNK 1
	#define MSG_DIE   0

	/*
	 * Synchronizes with master process.
	 */
	extern void sync_master(void);
	
	/*
	 * Opens NoC connectors.
	 */
	extern void open_noc_connectors(void);
	
	/*
	 * Closes NoC connectors.
	 */
	extern void close_noc_connectors(void);

	/* Inter process communication. */
	extern int rank;  /* Process rank.   */
	extern int infd;  /* Input channel.  */
	extern int outfd; /* Output channel. */
	
#endif /* _SLAVE_H_ */