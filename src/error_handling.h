#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#define ERRCODE 2

#define ERR(e)                                                           \
  {                                                                      \
    printf("Error: %s\n", nc_strerror(e));                               \
    exit(ERRCODE);                                                       \
  }

#define SUCCESS_OR_DIE(f...)                                             \
  do                                                                     \
    {                                                                    \
      const gaspi_return_t r = f;				  	 \
      if (r != GASPI_SUCCESS)						 \
	{								 \
	  gaspi_printf ("Error: '%s' [%s:%i]: %i\n", #f, __FILE__, __LINE__, r); \
          								 \
	  exit (EXIT_FAILURE);						 \
	}								 \
    } while (0)


#define ASSERT(x...)                                                     \
  if (!(x))                                                              \
    {									 \
      fprintf (stderr, "Error: '%s' [%s:%i]\n", #x, __FILE__, __LINE__); \
      exit (EXIT_FAILURE);						 \
    }



#define ASSERT_INT(x, y)							\
  if ((x) != (y))							\
    {									 \
      fprintf (stderr, "Error: '%s' != '%s' %d != %d [%s:%i]\n", #x, #y, (x), (y), __FILE__, __LINE__); \
      exit (EXIT_FAILURE);						 \
    }


int f_exist(char *fname);

#endif
