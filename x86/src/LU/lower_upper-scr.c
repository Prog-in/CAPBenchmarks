/*
 * Copyright(C) 2014 Pedro H. Penna <pedrohenriquepenna@gmail.com>
 * 
 * lu.c - Lower Upper Benchmark Kerkernl.
 */

#include <assert.h>
//#include <global.h>
#include <math.h>
#include <mpi.h>
//#include <scr.h>
#include <util.h>
//#include "../../libsrc/util.h"
#include "lu.h"

static int size;
static int rank;

/*
 * Arguments sanity check.
 */
#define SANITY_CHECK() \
	assert(m != NULL); \
	assert(l != NULL); \
	assert(u != NULL); \

/*
 * Swaps two rows of a matrix.
 */
static void _swap_rows(struct matrix *m, int i1, int i2)
{
	int j;     /* Loop index.      */
	float tmp; /* Temporary value. */
	
	/* Swap columns. */
	for (j = 0; j < m->width; j++)
	{
		tmp = MATRIX(m, i1, j);
		MATRIX(m, i1, j) = MATRIX(m, i2, j);
		MATRIX(m, i2, j) = tmp;
	}
}

/*
 * Swaps two columns in a matrix.
 */
static void _swap_columns(struct matrix *m, int j1, int j2)
{
	int i;     /* Loop index.      */
	float tmp; /* Temporary value. */

	/* Swap columns. */
	for (i = 0; i < m->height; i++)
	{
		tmp = MATRIX(m, i, j1);
		MATRIX(m, i, j1) = MATRIX(m, i, j2);
		MATRIX(m, i, j2) = tmp;
	}
}

/*
 * Finds the pivot element.
 */
static float _find_pivot(struct matrix *m, int i0, int j0)
{
	int i, j;         /* Loop indexes.          */
	int ipvt, jpvt;   /* Pivot indexes.         */
	int pipvt, pjpvt; /* Private pivot indexes. */
	
	ipvt = i0;
	jpvt = j0;
	
	pipvt = i0;
	pjpvt = j0;
	
	int local_rows = (m->height - i0) / size;
	int remainder = (m->height - i0) % size; // Calculate the remainder for padding

	/* If rank is less than remainder, add one more row for padding */
	if (rank < remainder) {
		//MPI_Scatter(NULL, m->width, MPI_FLOAT, m[local_rows], m->width, MPI_FLOAT, 0, MPI_COMM_WORLD);
		local_rows++; /* Increment local_rows for processes with padding */
	}

	/* Find pivot element. */
	int flag = rank-1 < remainder ? 1 : 0;
	for (int i = i0 + rank*local_rows + flag; i < i0 + (rank+1)*local_rows; i++) {
		for (int j = j0; j < m->width; j++) {
			/* Found. */
			if (fabs(MATRIX(m, i, j)) < fabs(MATRIX(m,pipvt,pjpvt)))
			{
				pipvt = i;
				pjpvt = j;
			}
		}
	}

	int *pivots_i, *pivots_j;
	/* Gather the results */
	if (0 == rank) {
		pivots_i = smalloc(sizeof(int) * size);
		pivots_j = smalloc(sizeof(int) * size);
	}

	MPI_Gather(&pipvt, 1, MPI_FLOAT, pivots_i, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Gather(&pjpvt, 1, MPI_FLOAT, pivots_j, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	/* Reduct. */
	if (0 == rank) {
		for (int i = 0; i < size; i++) {
			if (fabs(MATRIX(m, pivots_i[i], pivots_j[i]) > fabs(MATRIX(m, ipvt, jpvt))))
			{
				ipvt = pivots_i[i];
				jpvt = pivots_j[i];
			}
		}
		free(pivots_i);
		free(pivots_j);
	}
	
	//MPI_Bcast(ipvt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//MPI_Bcast(jpvt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	// todo: implement parallelism
	if (0 == rank) {
		_swap_rows(m, i0, ipvt);
		_swap_columns(m, j0, jpvt);
	}
	
	return (MATRIX(m, ipvt, jpvt));
}

/*
 * Applies the row reduction algorithm in a matrix.
 */
static void _row_reduction(struct matrix *m, int i0, float pivot)
{
	int j0;      /* Starting column. */
	int i, j;    /* Loop indexes.    */
	float mult;  /* Row multiplier.  */
	
	j0 = i0;

	int local_rows = (m->height - i0) / size;
	int remainder = (m->height - i0) % size; // Calculate the remainder for padding

	/* If rank is less than remainder, add one more row for padding */
	if (rank < remainder) {
		local_rows++; /* Increment local_rows for processes with padding */
	}

	int flag = rank-1 < remainder ? 1 : 0;
	int local_start = i0 + rank*local_rows + flag;
	int local_end = i0 + (rank+1)*local_rows;

	int qtd_elements = (local_end - local_start + 1) * (m->height - j0 + 1);

	MPI_Datatype local_columns;

	int bigsizes[2]  = {m->height, m->width};
	int subsizes[2]  = {local_end - local_start + 1, m->height - j0 + 1};
	int starts[2] = {local_start, j0};

	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &local_columns);
	MPI_Type_commit(&local_columns);

	// do I need to malloc?
	local_columns = smalloc(qtd_elements);

	for (int i = local_start; i < local_end; i++) {
		mult = MATRIX(m, i, j0)/pivot;
	
		/* Store multiplier. */
		//MATRIX(m, i, j0) = mult;
		MATRIX(local_columns, i - local_start, 0) = mult;
	
		/* Iterate over columns. */
		for (j = j0 + 1; j < m->width; j++) {
			//MATRIX(m, i, j) = MATRIX(m, i, j) - mult*MATRIX(m, i0, j);
			MATRIX(local_columns, i - local_start, j - j0) = MATRIX(m, i, j) - mult*MATRIX(m, i0, j);
		}
	}

	MPI_Allgather(local_columns, qtd_elements, MPI_FLOAT, m->elements, qtd_elements, MPI_FLOAT, MPI_COMM_WORLD);

	MPI_Type_free(&local_columns);
}

/*
 * Performs LU factorization in a matrix.
 */
int lower_upper(struct matrix *m, struct matrix *l, struct matrix *u)
{
	int i, j;    /* Loop indexes. */
	float pivot; /* Pivot.        */
	
	MPI_Init(NULL, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* Apply elimination on all rows. */
	for (i = 0; i < m->height - 1; i++)
	{
		pivot = _find_pivot(m, i, i);	
	
		/* Impossible to solve. */
		if (pivot == 0.0)
		{
			warning("cannot factorize matrix");
			return (-1);
		}
		
		_row_reduction(m, i, pivot);
	}
	
	int local_rows = m->height / size;
	int remainder = m->height % size; // Calculate the remainder for padding

	/* If rank is less than remainder, add one more row for padding */
	if (rank < remainder) {
		local_rows++; /* Increment local_rows for processes with padding */
	}

	int flag = rank-1 < remainder ? 1 : 0;
	int local_start = rank*local_rows + flag;
	int local_end = (rank+1)*local_rows;

	int qtd_elements = (local_end - local_start + 1) * m->height;

	MPI_Datatype local_columns;

	int bigsizes[2]  = {m->height, m->width};
	int subsizes[2]  = {local_end - local_start + 1, m->height};
	int starts[2] = {local_start, 0};

	MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &local_columns);
	MPI_Type_commit(&local_columns);

	float *local_columns_l = smalloc(qtd_elements);
	float *local_columns_u = smalloc(qtd_elements);

	/* Build upper and lower matrixes.  */
	for (i = local_start; i < local_end; i++) {
		for (j = 0; j < m->width; j++)
		{
			if (i > j)
			{
				MATRIX(local_columns_l, i, j) = 0.0;
				MATRIX(local_columns_u, i, j) = MATRIX(m, i, j);
			}
			
			else if (i < j)
			{	
				MATRIX(local_columns_l, i, j) = MATRIX(m, i, j);
				MATRIX(local_columns_u, i, j) = 0.0;
			}
			
			else
			{
				MATRIX(local_columns_l, i, j) = 1.0;
				MATRIX(local_columns_u, i, j) = 1.0;
			}
		}
	}

	MPI_Allgather(local_columns_l, qtd_elements, MPI_FLOAT, l->elements, qtd_elements, MPI_FLOAT, MPI_COMM_WORLD);
	MPI_Allgather(local_columns_u, qtd_elements, MPI_FLOAT, u->elements, qtd_elements, MPI_FLOAT, MPI_COMM_WORLD);

	MPI_Type_free(&local_columns);

	return (0);
}
