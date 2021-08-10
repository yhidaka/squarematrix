      subroutine ftm(f,g,powerindex,sequencenumber,tol,norder,n,u)
CF2PY INTENT(out) :: u
CF2PY INTENT(hide) :: n
CF2PY INTENT(hide) :: norder
cf2py DOUBLE COMPLEX :: f(n)
cf2py DOUBLE COMPLEX :: g(n)
cf2py INTEGER :: powerindex(n,4)
cf2py double :: tol
cf2py integer :: sequencenumber(norder+1, norder+1, norder+1, norder+1)
      double complex f(n)
      double complex g(n)
      double complex u(n)
      integer norder
      integer n
      integer powerindex(n,4)
      integer index(4)
      integer power
      integer sn
      integer sequencenumber(norder+1, norder+1, norder+1, norder+1)

      do 400 i=1,n
        u(i)=0
400   continue

      do 100 i=1,n
      if(abs(f(i))>tol) then
        do 200 j=1,n
          if(abs(g(j))>tol) then

             power=0
             do 300 k=1,4
                index(k)=powerindex(i,k)+powerindex(j,k) 
                power=power+index(k)
300          continue
             if(power<norder+1) then
                sn=sequencenumber(index(1)+1,index(2)+1,
     $          index(3)+1,index(4)+1)+1
                u(sn)=u(sn)+f(i)*g(j)
             else
             endif

          else
          endif
200     continue
      else
      endif
100   continue


      end
      
      


