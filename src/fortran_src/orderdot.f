      subroutine orderdot(u1,u2,order,powerindex,ypowerorder,m,n,u1u2)
cf2py intent(in) :: u1
cf2py intent(in) :: u2
cf2py intent(in) :: order
cf2py intent(in) :: powerindex
cf2py intent(in) :: ypowerorder
cf2py intent(hide) :: m,n
cf2py intent(out) :: u1u2
      double complex :: u1(m)
      double complex :: u2(m)
      integer :: order,n,m,ypowerorder
      integer :: powerindex(m,n)
      double complex :: u1u2
      
cThis fortran replace the following python module
cdef orderdotnew(u1,u2,order,powerindex,ypowerorder):
c	#see jf26.py section 7 for why we need to modify orderdot to this by adding [:2] to powerindex[i] to correct an old error of using the tatal power including y power
c	v1=array([ u1[i] for i in range(len(u1)) if sum(powerindex[i])==order and sum(powerindex[i][2:4])<=ypowerorder])
c	v2=array([ u2[i] for i in range(len(u2)) if sum(powerindex[i])==order and sum(powerindex[i][2:4])<=ypowerorder])
c	u1u2=np.dot(v1,v2)
c	return u1u2
      u1u2=0
      do 100 i=1,m
          ipower=0
          do 101 j=1,n
101        ipower=ipower+powerindex(i,j)
              if (ipower==order.AND.ipower<=ypowerorder) then
                  u1u2=u1u2+u1(i)*u2(i)
              else
              end if
100     continue
      end subroutine orderdot

