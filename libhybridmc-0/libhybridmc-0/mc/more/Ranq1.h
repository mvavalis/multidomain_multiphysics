#ifndef _RANQ1_H_
#define _RANQ1_H_

/**
 * Random number generator Ranq1
 *
 * Numerical Recipes 3rd edition:
 * It has a period of “only” 1.8*10^19 , so it should not be used by an
 * application that makes more than about 10^12 calls.
 */
class Ranq1
{
private:
	unsigned long long v;
public:
	Ranq1(unsigned long long j=0) : v(4101842887655102017LL)
	{
		v ^= j;
		v = int64();
	}
	inline unsigned long long int64()
	{
		v ^= v >> 21; v ^= v << 35; v ^= v >> 4;
		return v * 2685821657736338717LL;
	}
	inline double doub() {return 5.42101086242752217E-20*int64();}
	inline unsigned int int32() {return (unsigned int)int64();}
};

#endif
