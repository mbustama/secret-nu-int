#include <stdio.h>
#include <math.h>
#include <cassert> 
#include <iostream>
#include <fstream> 
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#define c 29979245800.0e0
#define energybins 400  
double energyarray[energybins];
double omega[energybins];
double y[energybins];
double energymin=2.0e0; 
double energymax=8.0e0;
double nt(double z) // cm^-3
{
    double ans=56.0*pow((1.0+z),3.0);
    return(ans);
}
double L0(double E)
{
    double k=1.0;
    double gamma=2.0;
    double E_max=1.0e7;
    return(k*pow(E,-gamma)*exp(-E/E_max));
}
double W(double z)
{
    double a=3.4;
    double b=-0.3;
    double c1=-3.5;
    double B=5000.0e0;
    double C=9.0e0;
    double eta=-10.0e0;	
    double ans=0.0e0;
    ans=pow((1+z),(a*eta));
    ans=ans+pow(((1+z)/B),(b*eta));
    ans=ans+pow(((1+z)/C),c1*eta);
    ans=pow(ans,(1.0/eta));
    //return(((1+z)**(a*eta)+((1+z)/B)**(b*eta)+((1+z)/C)**(c1*eta))**(1/eta));
    //printf("Wz %e %e",z,ans);
    return(ans);
}
double L(double z,double E)
{
    return(W(z)*L0(E));
}
double H(double z) //  # s^-1
{
    double H0=0.678/(9.777752*3.16*1.e16); 
    double OM=0.308; 
    double OL=0.692;
    return(H0*sqrt(OM*pow((1.+z),3.0) + OL));
}
double sigma(double E,double g,double M,double m)  //# cm^2
{
	double ans=0.0e0;
    	ans=(g*g*g*g/(16.0*M_PI))*(2.0*E*m)/((2.0*E*m-M*M)*(2.0*E*m-M*M)+((M*M*M*M*g*g*g*g)/(16*M_PI*M_PI)))*0.389379e-27;
	return(ans);
}
double dsigma(double Ep,double E,double g,double M,double m) //# cm^2*GeV^-1
{
    double ans=0.0e0;
    if(Ep < E)
        {
		ans=0.0e0;
	}
    else
	{
        	ans=sigma(Ep, g, M, m)*3./Ep*((E/Ep)*(E/Ep)+(1.-(E/Ep))*(1.-(E/Ep)));
	}
	return(ans);
}
int
func (double r, const double y[], double f[],
      void *params)
{
  (void)(r); /* avoid unused parameter warning */
  double theta = *(double *)params;
  (void)(theta);
  for(int i=0;i<energybins;i++)
  {
  	f[i]=0.0e0;
  }
  for(int k=0;k<energybins;k++) 
  {
	f[k]=f[k]+H(fabs(r))*y[k];
	f[k]=f[k]+L(fabs(r), energyarray[k]);
	f[k]=f[k]-c*nt(fabs(r))*sigma(energyarray[k], 0.3, 0.1, 1e-10)*y[k]; // Use this for no numerical integration for the loss term
	//printf("%e %e %e %e\n",fabs(r),energyarray[k],y[k],-c*nt(fabs(r))*sigma(energyarray[k], 0.03, 0.01, 1e-10)*y[k]);
	for(int kp=0;kp<energybins-1;kp++)
	{
		//f[k]=f[k]-0.5*c*nt(fabs(r))*dsigma(energyarray[k],energyarray[kp],0.01, 0.001, 1e-10)*(energyarray[kp+1]-energyarray[kp])*y[k];
                //Use the line above for numerical integration over the loss term
		f[k]=f[k]+c*nt(fabs(r))*dsigma(energyarray[kp],energyarray[k],0.3, 0.1, 1e-10)*(energyarray[kp+1]-energyarray[kp])*y[kp];
	}
	f[k]=f[k]/((1.0+fabs(r))*H(fabs(r)));
  }
  return GSL_SUCCESS;
}

void initarrays()
{
        for(int j=0;j<energybins;j++)
        {
                energyarray[j]=pow(10.0e0,energymin+(double(j)+0.5e0)*(energymax-energymin)/(double)energybins);
        }
        for(int i=0;i<energybins;i++)
        {
        	y[i]=0.0e0;
        }
}

int
main (void)
{
  size_t dimension=energybins;
  double theta = M_PI/4.0; 
  gsl_odeiv2_system sys = {func, NULL, dimension, &theta};
  

  gsl_odeiv2_driver * d = 
    gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk8pd,
				  0.1, 1e-4, 0.0);
  
  double rmin = -6.0e0, rmax = 0.0e0, deltar=0.1e0;
  
  initarrays();

  double r=rmin, r_next;

  printf("%e ",r);
  for(int i=0;i<energybins;i++)
  {
	printf("%e ",y[i]);
  }
  printf("\n");
  for (r_next = rmin + deltar; r_next <= rmax; r_next += deltar)
  {
  	while (r < r_next)
        {
                int status = gsl_odeiv2_driver_apply (d, &r, r_next, y);
                if (status != GSL_SUCCESS)
                        break;
        } 
        //The few lines above evolves the ode
	printf("%e ",r);
        for(int i=0;i<energybins;i++)
  	{
        	printf("%e %e ",energyarray[i],y[i]*energyarray[i]*energyarray[i]);
  	}
	printf("\n");
        //prints the radius and all the values of y 
        //the values of y will be printed at a regular interval
        //determined by deltar
  }
  
 // ofstream write_output("Flux.dat");
 // assert(write_output.is_open());
 // for(int i=0; i<energybins; i++)
 // {
 //   write_output << energyarray[i] << " " << y[i]*energyarray[i]*energyarray[i] << "\n" ; 
 // }
  
 //  write_output.close()

  gsl_odeiv2_driver_free (d);
  return 0;
}


