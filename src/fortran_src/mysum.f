      subroutine sum(x,y,sx,sy,n)
cf2py intent(in) :: x,y
cf2py intent(hide) :: n
cf2py intent(out) :: sx,sy
      double complex :: x(n),y(n),sx,sy
      integer :: n
      sx=0.0
      sy=0.0
        
      do 400 i=1,n
              sx=sx+x(i)
              sy=sy+y(i)
400       continue
      end subroutine sum
