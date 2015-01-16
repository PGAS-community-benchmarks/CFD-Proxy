#ifndef UTIL_H
#define UTIL_H

#define MAX(a, b) ( (a) > (b) ? (a) : (b))
#define MIN(a, b) ( (a) < (b) ? (a) : (b))

void  check_free(void *ptr);
void *check_malloc(size_t bytes);
void *check_realloc(void *old, size_t bytes);

void sort_median(double *begin, double *end);

void sort_faces(int pm[]
		, int fpoint[][2]
		, int ttype[]
		, int nfaces
		);

double now();

#endif
