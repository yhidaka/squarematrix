	subroutine dot(ff,f,u,l,m)
cf2py intent(in) :: ff
cf2py intent(in) :: f
cf2py intent(hide) :: l,m
cf2py intent(out) :: u
          double precision :: f(m)
          double complex :: u(l)
          integer :: l,m
          complex :: ff(l, m)

          do 400 i=1,l
              u(i)=0
400       continue

          do 100 i=1,l
	        do 200 j=1,m
	            u(i)=u(i)+ff(i,j)*f(j)
200	        continue
100       continue
	end subroutine dot
